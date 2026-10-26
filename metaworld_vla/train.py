"""Standalone Meta-World trainer; it does not import RoboTwin's dataset or runner."""
from __future__ import annotations
import argparse, json, logging, random
from contextlib import nullcontext
from pathlib import Path
import sys
import time
import numpy as np, torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ``python metaworld_vla/train.py`` puts only ``metaworld_vla`` on
# ``sys.path``.  Add the project root so the shared ``models`` package is
# importable regardless of the caller's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_orientation import IMAGE_ROTATION
from dataset import MetaWorldTaskDataset, collate_fn
from dataset_native import NativeMetaWorldDataset
from dataset_yf import YFMetaWorldDataset
from models.model_runner import ModelFactory, VLAWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def epoch_log_steps(num_steps, updates=4):
    """Evenly spaced progress logs, including the final step, without duplicates."""
    if updates < 1:
        raise ValueError('training.log_updates_per_epoch must be positive')
    if num_steps < 1:
        return set()
    return {(num_steps * i + updates - 1) // updates for i in range(1, updates + 1)}


def format_progress(epoch, epochs, step, steps, total_loss, info, lr, train_cfg):
    # Match the main RoboTwin trainer: Loss is the action flow-matching loss;
    # Total also includes the weighted auxiliary losses.
    message = (f'Epoch [{epoch}/{epochs}] Step [{step}/{steps}] '
               f'Loss: {info.get("loss_mse", 0):.4f} Total: {total_loss:.4f} LR: {lr:.2e}')
    if train_cfg['use_future_feat']:
        message += f' FutureFeat: {info.get("loss_future_feat", 0):.4f}'
    if train_cfg['use_global_intent']:
        message += (f' Intent: {info.get("loss_intent", 0):.4f}'
                    f' TargetNorm: {info.get("intent_target_norm", 0):.3f}'
                    f' PredNorm: {info.get("intent_pred_norm", 0):.3f}'
                    f' Gate: {info.get("intent_gate", 0):.6f}')
    return message


def build_train_config(cfg):
    """Wire the shared causal global-intent loss into the standalone trainer."""
    t, m = cfg.training, cfg.model
    intent = m.get('global_intent', {})
    if m.get('spatial_intent', {}).get('enabled', False):
        raise ValueError('Meta-World spatial-intent training/evaluation is not wired yet; use a separate future experiment.')
    use_intent = bool(intent.get('enabled', False))
    target_layer = int(m.future_feat.target_layer)
    if use_intent and target_layer not in list(m.vision_encoder.feat_layers):
        raise ValueError('Global intent requires future_feat.target_layer in vision_encoder.feat_layers')
    if use_intent and float(t.get('lambda_intent', 0.0)) <= 0:
        raise ValueError('Global intent requires a positive training.lambda_intent')
    horizon = int(cfg.dataset.get('future_frame_offset', cfg.common.action_chunk_size))
    return dict(time_mu=float(t.time_mu), time_sigma=float(t.time_sigma),
                use_vel_weight=bool(t.use_vel_weight), vel_weight_alpha=float(t.vel_weight_alpha),
                vel_weight_sigma=float(t.vel_weight_sigma), use_future_feat=bool(m.future_feat.enabled),
                lambda_future_feat=float(t.lambda_future_feat), use_global_intent=use_intent,
                lambda_intent=float(t.get('lambda_intent', 0.0)),
                intent_cosine_weight=float(t.get('intent_cosine_weight', 0.0)),
                intent_cosine_eps=float(t.get('intent_cosine_eps', 1e-4)),
                intent_horizon=horizon, use_spatial_intent=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--stats", required=True)
    p.add_argument("--save-dir", required=True); p.add_argument("--resume", default=None)
    p.add_argument("--init-from", default=None); p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args(); cfg = OmegaConf.load(args.config)
    seed_everything(int(cfg.training.get("seed", 0)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dcfg, ccfg, mcfg, tcfg = cfg.dataset, cfg.common, cfg.model, cfg.training
    train_cfg = build_train_config(cfg)
    task_cond_dir = dcfg.get("task_cond_dir", None) if mcfg.get("use_task_cond", False) else None
    dataset_format = str(dcfg.get('format', 'auto'))
    dataset_classes = {'native': NativeMetaWorldDataset, 'metaworld_yf': YFMetaWorldDataset, 'auto': MetaWorldTaskDataset}
    if dataset_format not in dataset_classes:
        raise ValueError(f'Unknown dataset.format: {dataset_format}')
    dataset_class = dataset_classes[dataset_format]
    log.info('Dataset format=%s, reader=%s', dataset_format, dataset_class.__name__)
    ds = dataset_class(str(dcfg.dataset_dir), list(dcfg.task_names), int(ccfg.action_chunk_size), int(ccfg.state_dim), tuple(dcfg.image_size), task_cond_dir, train_cfg['use_future_feat'] or train_cfg['use_global_intent'], dcfg.get("future_frame_offset", None), bool(dcfg.get("image_aug", False)), camera_name=str(dcfg.camera_names[0]))
    loader = DataLoader(ds, batch_size=int(tcfg.batch_size), shuffle=True, drop_last=True, num_workers=int(cfg.system.get("num_workers", 0)), pin_memory=bool(cfg.system.get("pin_memory", True)), collate_fn=collate_fn, persistent_workers=int(cfg.system.get("num_workers", 0)) > 0)
    vision, hidden, regs, patch = ModelFactory.create_vision_encoder(mcfg.vision_encoder.checkpoint_path, dtype=dtype, device=str(device))
    # Match the action model precision to the frozen DINO encoder and the
    # tensors prepared by VLAWrapper.  ``.to(device)`` alone leaves Linear
    # weights in float32, which fails on CUDA when qpos/actions are bfloat16.
    action = ModelFactory.create_action_model(cfg, hidden, len(mcfg.vision_encoder.feat_layers), task_cond_dim=hidden if mcfg.get("use_task_cond", False) else None, patch_size=patch, num_register_tokens=regs).to(device=device, dtype=dtype)
    action.keep_control_gates_fp32()
    model = VLAWrapper(vision, action, tcfg.time_sampler, mcfg.vision_encoder.feat_layers, bool(mcfg.vision_encoder.include_cls_register), regs, device, dtype, args.stats, train_cfg, int(mcfg.future_feat.target_layer)).to(device)
    log.info('Global intent=%s injection=%s horizon=%d lambda=%.3g pooling=%s', train_cfg['use_global_intent'], action.inject_global_intent, train_cfg['intent_horizon'], train_cfg['lambda_intent'], getattr(action, 'intent_pooling', 'disabled'))
    optimizer = torch.optim.AdamW(action.parameters(), lr=float(tcfg.learning_rate), betas=tuple(tcfg.betas), weight_decay=float(tcfg.weight_decay))
    total_steps = max(1, int(tcfg.epochs) * len(loader)); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(tcfg.lr_min))
    start_epoch = 0; global_step = 0
    for checkpoint_path in (args.init_from, args.resume):
        if checkpoint_path:
            metadata = torch.load(checkpoint_path, map_location="cpu")
            if metadata.get("image_rotation_degrees") != IMAGE_ROTATION:
                raise ValueError("Checkpoint uses old/unknown image orientation; start a new rotation=180 run.")
    if args.init_from:
        action.load_state_dict(torch.load(args.init_from, map_location="cpu")["model_state"], strict=True)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu"); action.load_state_dict(ck["model_state"]); optimizer.load_state_dict(ck["optimizer_state"]); scheduler.load_state_dict(ck["scheduler_state"]); start_epoch = int(ck["epoch"]); global_step = int(ck.get("global_step", 0))
    run = Path(args.save_dir); run.mkdir(parents=True, exist_ok=True); OmegaConf.save(cfg, run / "config.yaml")
    log_updates = int(tcfg.get('log_updates_per_epoch', 4))
    if log_updates < 1:
        raise ValueError('training.log_updates_per_epoch must be positive')
    for epoch in range(start_epoch, int(tcfg.epochs)):
        action.train(); total = 0.0; epoch_steps = 0
        steps_this_epoch = len(loader)
        if args.max_steps is not None:
            steps_this_epoch = min(steps_this_epoch, max(0, args.max_steps - global_step))
        progress_steps = epoch_log_steps(steps_this_epoch, log_updates)
        epoch_started = time.perf_counter()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            # Match the main RoboTwin trainer: autocast is required for the
            # mixed DINO/action-model path (time embeddings and decoder
            # inputs otherwise remain float32 while CUDA uses bfloat16).
            amp_context = (
                torch.amp.autocast("cuda", dtype=dtype)
                if device.type == "cuda" else nullcontext()
            )
            with amp_context:
                loss, info = model(batch)
            loss.backward(); torch.nn.utils.clip_grad_norm_(action.parameters(), float(tcfg.grad_clip_norm)); optimizer.step(); scheduler.step()
            step_loss = float(loss.detach())
            total += step_loss; global_step += 1; epoch_steps += 1
            if epoch_steps in progress_steps:
                log.info(format_progress(epoch + 1, int(tcfg.epochs), epoch_steps,
                         steps_this_epoch, step_loss, info, scheduler.get_last_lr()[0], train_cfg))
            if args.max_steps and global_step >= args.max_steps: break
        log.info('=== Epoch %d Completed. Avg Loss: %.4f | Time: %.1fs ===',
                 epoch + 1, total / max(1, epoch_steps), time.perf_counter() - epoch_started)
        if (epoch + 1) % int(tcfg.save_interval_epoch) == 0 or (args.max_steps and global_step >= args.max_steps):
            torch.save({"image_rotation_degrees": IMAGE_ROTATION, "model_state": action.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "epoch": epoch + 1, "global_step": global_step}, run / f"checkpoint_epoch_{epoch + 1}.pt")
        if args.max_steps and global_step >= args.max_steps: break

if __name__ == "__main__": main()
