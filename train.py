import os
import sys
import random
import torch
import numpy as np
import logging
import argparse
import time
import math
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataloader.dataset import collate_fn, create_dataset
from utils.train_utils import count_parameters
from models.model_runner import ModelFactory, VLAWrapper


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def seed_everything(seed, deterministic=False):
    """Seed training RNGs; DataLoader shuffling uses its own generator below."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Seed Python/NumPy inside each DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return OmegaConf.load(config_path)


class LossLogger:
    def __init__(self, log_dir="log/loss"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = os.path.join(log_dir, f"train_loss_{timestamp}.csv")
        with open(self.log_file, 'w') as f:
            f.write("Epoch,Step,Global_Step,Loss\n")

    def log(self, epoch, step, global_step, loss):
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{step},{global_step},{loss:.6f}\n")


def build_train_config_from_yaml(cfg):
    """Extract the training-parameter dict needed by VLAWrapper from OmegaConf cfg.training"""
    t = cfg.training
    ff = cfg.model.get('future_feat', {})
    intent = cfg.model.get('global_intent', {})
    spatial = cfg.model.get('spatial_intent', {})
    future_offset = cfg.dataset.get('future_frame_offset', None)
    if future_offset is None:
        future_offset = cfg.common.action_chunk_size
    return {
        'time_mu': t.time_mu,
        'time_sigma': t.time_sigma,
        'use_vel_weight': t.use_vel_weight,
        'vel_weight_alpha': t.vel_weight_alpha,
        'vel_weight_sigma': t.vel_weight_sigma,
        'use_future_feat': ff.get('enabled', False) if ff else False,
        'lambda_future_feat': t.get('lambda_future_feat', 0.0),
        'use_global_intent': intent.get('enabled', False) if intent else False,
        'lambda_intent': t.get('lambda_intent', 0.0),
        'intent_cosine_weight': t.get('intent_cosine_weight', 0.0),
        'intent_cosine_eps': t.get('intent_cosine_eps', 1.0e-4),
        'intent_horizon': future_offset,
        'use_spatial_intent': spatial.get('enabled', False) if spatial else False,
        'lambda_spatial_intent': t.get('lambda_spatial_intent', 0.0),
        'spatial_horizon': future_offset,
    }


def load_init_weights(action_model, checkpoint_state):
    """Load a baseline exactly while allowing only explicitly added intent modules."""
    if not (action_model.use_global_intent or action_model.use_spatial_intent):
        return action_model.load_state_dict(checkpoint_state, strict=True)

    current_state = action_model.state_dict()
    unexpected = sorted(set(checkpoint_state) - set(current_state))
    mismatched = sorted(
        key for key in set(checkpoint_state) & set(current_state)
        if checkpoint_state[key].shape != current_state[key].shape
    )
    missing = sorted(set(current_state) - set(checkpoint_state))
    allowed_prefixes = (
        'intent_predictor.',
        'intent_to_token.',
        'type_emb_intent',
        'intent_gate',
        'spatial_intent_predictor.',
        'spatial_to_tokens.',
        'type_emb_spatial',
        'spatial_gate',
    )
    disallowed_missing = [
        key for key in missing if not key.startswith(allowed_prefixes)
    ]
    if unexpected or mismatched or disallowed_missing:
        raise RuntimeError(
            "Unsafe baseline initialization for intent-augmented model: "
            f"unexpected={unexpected}, shape_mismatch={mismatched}, "
            f"disallowed_missing={disallowed_missing}"
        )

    msg = action_model.load_state_dict(checkpoint_state, strict=False)
    loaded_keys = len(checkpoint_state) - len(unexpected)
    logger.info(
        f"Baseline-compatible partial load: loaded={loaded_keys}, "
        f"new_intent_keys={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
    )
    if msg.missing_keys:
        logger.info(f"New intent parameters initialized from config seed: {msg.missing_keys}")
    return msg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLA Training Script (DINOv3) - two-stage LR version")
    parser.add_argument("--config", type=str, default="./configs/robotwin_all.yaml",
                        help="Path to config file")
    parser.add_argument("--norm_stats_path", type=str, 
                        default="./utils/stat-500-all.json",
                        help="Path to normalization stats")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vla",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from after an interruption "
                             "(restores model + optimizer + scheduler + epoch, continues the "
                             "lr schedule saved in the checkpoint)")
    parser.add_argument("--init_from", type=str,
                        default=None,
                        help="Path to checkpoint to initialize model weights from "
                             "Only model weights "
                             "are loaded; optimizer and lr scheduler start fresh from --config. "
                             "Mutually exclusive with --resume.")
    args = parser.parse_args()

    if args.resume and args.init_from:
        raise ValueError("--resume and --init_from are mutually exclusive: "
                         "--resume continues an interrupted run (inherits its lr schedule), "
                         "--init_from starts a new stage with a fresh lr schedule.")

    # =========================================================================
    # 1. Environment / Config
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info(f"Using device: {device}, Precision: {dtype}")

    config = load_config(args.config)

    train_seed = int(config.training.get('seed', 0))
    deterministic = bool(config.training.get('deterministic', False))
    seed_everything(train_seed, deterministic=deterministic)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(train_seed)
    logger.info(f"Training seed: {train_seed} | Deterministic algorithms: {deterministic}")

    # Training hyperparameters
    epochs = config.training.epochs
    grad_accum_steps = config.training.grad_accum_steps
    save_interval_epoch = config.training.save_interval_epoch
    batch_size = config.training.batch_size
    grad_clip_norm = config.training.grad_clip_norm
    lr = config.training.learning_rate
    lr_min = config.training.lr_min

    if lr_min > lr:
        logger.warning(f"lr_min ({lr_min}) > learning_rate ({lr}): "
                       f"CosineAnnealingLR will RAISE the lr towards lr_min instead of decaying. "
                       f"Check your config.")

    logger.info(f"Epochs: {epochs} | Batch: {batch_size} | "
                f"grad_accum: {grad_accum_steps} | effective_batch: {batch_size * grad_accum_steps} | "
                f"save_every: {save_interval_epoch} ep")
    logger.info(f"LR schedule: CosineAnnealingLR {lr:.2e} -> {lr_min:.2e}")
    logger.info(f"DINOv3 feat_layers: {list(config.model.vision_encoder.feat_layers)} | "
                f"include_cls_register: {config.model.vision_encoder.include_cls_register}")
    logger.info(f"Vel-Weight: {'ENABLED' if config.training.use_vel_weight else 'DISABLED'}")

    # =========================================================================
    # 2. Dataset / DataLoader
    # =========================================================================
    train_dataset = create_dataset(config, val=False)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.system.num_workers,
        pin_memory=config.system.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    logger.info(f"Dataset Size: {len(train_dataset)} | Batches per Epoch: {len(train_dataloader)}")

    # =========================================================================
    # 3. Model
    # =========================================================================
    logger.info(">>> Initializing VLA (DINOv3)")

    vision_encoder, dino_hidden_size, num_register_tokens, patch_size = ModelFactory.create_vision_encoder(
        config.model.vision_encoder.checkpoint_path,
        dtype, device,
    )

    feat_layers = list(config.model.vision_encoder.feat_layers)
    num_dino_layers = len(feat_layers)

    # Task condition vector dim (uses DINOv3 hidden_size; controlled by use_task_cond)
    use_task_cond = config.model.get('use_task_cond', False)
    task_cond_dim = dino_hidden_size if use_task_cond else None
    logger.info(f"Task Condition: {'ENABLED' if use_task_cond else 'DISABLED'}"
                + (f" (dim={task_cond_dim}, dir={config.dataset.get('task_cond_dir', None)})" if use_task_cond else ""))

    # Future feature prediction config (logging)
    ff_cfg = config.model.get('future_feat', {})
    use_future_feat = ff_cfg.get('enabled', False) if ff_cfg else False
    logger.info(f"Future-Feat Pred: {'ENABLED' if use_future_feat else 'DISABLED'}"
                + (f" (target_layer={ff_cfg.get('target_layer', -1)}, "
                   f"lambda={config.training.get('lambda_future_feat', 0.0)})" if use_future_feat else ""))

    intent_cfg = config.model.get('global_intent', {})
    spatial_cfg = config.model.get('spatial_intent', {})
    use_global_intent = bool(intent_cfg.get('enabled', False)) if intent_cfg else False
    logger.info(
        f"Global Intent: {'ENABLED' if use_global_intent else 'DISABLED'}"
        + (f" (inject={intent_cfg.get('inject_into_dit', False)}, "
           f"H={config.dataset.get('future_frame_offset', config.common.action_chunk_size)}, "
           f"lambda={config.training.get('lambda_intent', 0.0)}, "
           f"gate_init={intent_cfg.get('gate_init', 0.0)}, "
           f"gate_trainable={intent_cfg.get('gate_trainable', True)})"
           if use_global_intent else "")
    )
    use_spatial_intent = bool(spatial_cfg.get('enabled', False)) if spatial_cfg else False
    logger.info(
        f"Spatial Intent: {'ENABLED' if use_spatial_intent else 'DISABLED'}"
        + (f" (inject={spatial_cfg.get('inject_into_dit', False)}, "
           f"lambda={config.training.get('lambda_spatial_intent', 0.0)}, "
           f"regions={spatial_cfg.get('region_grid', [2, 4])}, "
           f"gate_init={spatial_cfg.get('gate_init', 0.1)})"
           if use_spatial_intent else "")
    )

    action_model = ModelFactory.create_action_model(
        config,
        dino_hidden_size=dino_hidden_size,
        num_dino_layers=num_dino_layers,
        task_cond_dim=task_cond_dim,
        patch_size=patch_size,
        num_register_tokens=num_register_tokens,
    )

    action_model.to(device, dtype=dtype)
    action_model.keep_control_gates_fp32()
    action_model.train()

    count_parameters(action_model, model_name="Action Model (Trainable)")

    train_config_dict = build_train_config_from_yaml(config)

    model = VLAWrapper(
        vision_encoder=vision_encoder,
        action_model=action_model,
        time_sampler=config.training.time_sampler,
        feat_layers=feat_layers,
        include_cls_register=config.model.vision_encoder.include_cls_register,
        num_register_tokens=num_register_tokens,
        device=device,
        dtype=dtype,
        norm_stats_path=args.norm_stats_path,
        train_config=train_config_dict,
        future_feat_target_layer=ff_cfg.get('target_layer', -1) if ff_cfg else -1,
    )

    # =========================================================================
    # 4. Optimizer / Scheduler
    # =========================================================================
    optimizer = AdamW(
        action_model.parameters(),
        lr=lr,
        betas=tuple(config.training.betas),
        weight_decay=config.training.weight_decay,
    )

    optimizer_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum_steps)
    if optimizer_steps_per_epoch < 1:
        raise ValueError(
            f"grad_accum_steps={grad_accum_steps} exceeds batches/epoch={len(train_dataloader)}")
    trailing_micro_batches = len(train_dataloader) % grad_accum_steps

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs * optimizer_steps_per_epoch,
        eta_min=lr_min,
    )

    # =========================================================================
    # 5. Load checkpoint: --init_from (new stage) or --resume (interrupted run)
    # =========================================================================
    start_epoch = 0
    global_step = 0

    if args.init_from:
        # New training stage (e.g. stage-2 low-lr phase): load ONLY model weights.
        # Optimizer/scheduler stay fresh so the lr schedule restarts from --config.
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_from}")

        logger.info(f">>> Initializing weights from {args.init_from} "
                    f"(fresh optimizer + lr schedule from config)")
        ckpt = torch.load(args.init_from, map_location='cpu', weights_only=False)

        msg = load_init_weights(action_model, ckpt['model_state_dict'])
        logger.info(f"Model loaded (init_from epoch={ckpt.get('epoch', '?')}). "
                    f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        logger.info(f"Starting new stage: lr={lr:.2e} -> {lr_min:.2e}, epochs={epochs}")

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        logger.info(f">>> Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)

        msg = action_model.load_state_dict(ckpt['model_state_dict'], strict=True)
        logger.info(f"Model loaded. missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            steps_done = ckpt['epoch'] * optimizer_steps_per_epoch
            for _ in range(steps_done):
                scheduler.step()
            logger.warning("Old checkpoint without scheduler_state_dict; "
                           "scheduler advanced manually (lr may drift slightly).")

        start_epoch = ckpt['epoch']
        global_step = ckpt.get('global_step',
                               start_epoch * optimizer_steps_per_epoch)

        # New checkpoints preserve RNG state so a run interrupted at an epoch
        # boundary resumes with the same shuffle and stochastic training path.
        if 'python_rng_state' in ckpt:
            random.setstate(ckpt['python_rng_state'])
        if 'numpy_rng_state' in ckpt:
            np.random.set_state(ckpt['numpy_rng_state'])
        if 'torch_rng_state' in ckpt:
            torch.set_rng_state(ckpt['torch_rng_state'])
        if (torch.cuda.is_available() and 'cuda_rng_state_all' in ckpt
                and ckpt['cuda_rng_state_all'] is not None):
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state_all'])
        if 'dataloader_generator_state' in ckpt:
            loader_generator.set_state(ckpt['dataloader_generator_state'])
        else:
            logger.warning("Checkpoint has no RNG state; resumed execution is not bitwise reproducible.")
        logger.info(f"Resumed at epoch={start_epoch}, global_step={global_step}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}")

    # =========================================================================
    # 6. Training Loop
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_save_dir = os.path.join(args.save_dir, f"sft_{timestamp}")
    os.makedirs(run_save_dir, exist_ok=True)

    OmegaConf.save(config, os.path.join(run_save_dir, "config.yaml"))
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    loss_logger = LossLogger(log_dir="log/loss")

    for epoch in range(start_epoch, epochs):
        model.train()
        if action_model.use_spatial_intent and action_model.inject_spatial_residual:
            action_model.set_spatial_schedule(epoch + 1)
            logger.info(
                f"Spatial residual schedule: epoch={epoch + 1}, "
                f"scale={action_model.spatial_schedule_scale:.3f}, "
                f"effective_gate={action_model.effective_spatial_gate().item():.6f}"
            )
        epoch_loss = 0.0
        optimizer.zero_grad()
        start_time = time.time()

        for step, batch in enumerate(train_dataloader):
            # Use the actual divisor for the shorter final accumulation group,
            # so its gradient magnitude matches a regular optimizer update.
            accumulation_divisor = (
                trailing_micro_batches
                if trailing_micro_batches and step >= len(train_dataloader) - trailing_micro_batches
                else grad_accum_steps
            )
            with torch.amp.autocast('cuda', dtype=dtype):
                loss, info_dic = model(batch)
                loss = loss / accumulation_divisor

            loss.backward()

            current_step_loss = loss.item() * accumulation_divisor
            epoch_loss += current_step_loss

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_logger.log(epoch + 1, step + 1, global_step, current_step_loss)

                if global_step % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    log_msg = (
                        f"Epoch [{epoch+1}/{epochs}] "
                        f"Step [{step+1}/{len(train_dataloader)}] "
                        f"Loss: {info_dic['loss_mse']:.4f} "
                        f"LR: {current_lr:.2e} "
                    )
                    if use_future_feat:
                        log_msg += f"FutureFeat: {info_dic.get('loss_future_feat', 0.0):.4f} "
                    if use_global_intent:
                        log_msg += (
                            f"Intent: {info_dic.get('loss_intent', 0.0):.4f} "
                            f"TargetNorm: {info_dic.get('intent_target_norm', 0.0):.3f} "
                            f"PredNorm: {info_dic.get('intent_pred_norm', 0.0):.3f} "
                            f"Gate: {info_dic.get('intent_gate', 0.0):.6f} "
                        )
                    if use_spatial_intent:
                        log_msg += (
                            f"Spatial: {info_dic.get('loss_spatial_intent', 0.0):.4f} "
                            f"SpatialTargetNorm: {info_dic.get('spatial_target_norm', 0.0):.3f} "
                            f"SpatialPredNorm: {info_dic.get('spatial_pred_norm', 0.0):.3f} "
                            f"SpatialGate: {info_dic.get('spatial_gate', 0.0):.6f} "
                        )
                    logger.info(log_msg)

        avg_loss = epoch_loss / len(train_dataloader)
        elapsed = time.time() - start_time
        logger.info(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s ===")

        if (epoch + 1) % save_interval_epoch == 0 or (epoch + 1) == epochs:
            ckpt_name = f"checkpoint_epoch_{epoch+1}.pt"
            ckpt_path = os.path.join(run_save_dir, ckpt_name)
            save_dict = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': action_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'train_seed': train_seed,
                'python_rng_state': random.getstate(),
                'numpy_rng_state': np.random.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'dataloader_generator_state': loader_generator.get_state(),
            }
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training Complete.")
