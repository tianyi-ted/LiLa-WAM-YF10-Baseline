"""MetaWorld-only teacher/student trainer for the lightweight PlanWAM experiment.

RoboTwin uses the independent root entrypoint train_planwam_robotwin.py.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metaworld_vla.dataset import collate_fn
from metaworld_vla.dataset_native import NativeMetaWorldDataset
from metaworld_vla.image_orientation import IMAGE_ROTATION
from models.model_runner import ModelFactory, VLAWrapper
from models.planwam_g2 import FutureDrivenPosterior, freeze_module, planwam_flow_loss
from models.intent_pool_contract import LATE_POOL, teacher_input_contract
from metaworld_vla.planwam_transfer import (
    validate_transfer_config, initialize_action_from_teacher, build_transfer_optimizer,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
VERSION = 'future_driven_g2_v1'


def image_rotation(cfg):
    return IMAGE_ROTATION


def build_dataset(cfg, stage):
    """MetaWorld-only reader; constructor and sample ordering are unchanged."""
    d, c = cfg.dataset, cfg.common
    dataset = NativeMetaWorldDataset(str(d.dataset_dir), list(d.task_names), int(c.action_chunk_size),
                                    int(c.state_dim), tuple(d.image_size), str(d.task_cond_dir),
                                    stage != 'action_only', int(d.future_frame_offset), False,
                                    camera_name=str(d.camera_names[0]))
    return dataset, collate_fn


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_config(cfg):
    if cfg.dataset.get('type', 'metaworld') != 'metaworld':
        raise ValueError('MetaWorld-only trainer: use train_planwam_robotwin.py for RoboTwin configs')
    stage = cfg.planwam.stage
    if stage not in ('teacher', 'student', 'action_only'):
        raise ValueError('planwam.stage must be teacher/student/action_only')
    m, t, c, d = cfg.model, cfg.training, cfg.common, cfg.dataset
    g = m.global_intent
    if not (m.use_task_cond and g.enabled and g.inject_into_dit):
        raise ValueError('this experiment requires indirect VTT through active G2')
    if m.get('inject_task_cond', True) or g.get('use_gate', True):
        raise ValueError('this experiment requires no direct VTT token and no gate')
    if g.pooling not in ('task_attention', LATE_POOL) or g.get('target_pooling', 'mean') != 'mean':
        raise ValueError('student input must be task_attention or task_attention_late; attention-target is excluded')
    if g.pooling == LATE_POOL:
        if stage == 'teacher':
            raise ValueError('late-pool experiment reuses the ORIGINAL single-query-contract teacher')
        if int(g.get('pool_num_queries', 4)) < 2 or int(g.get('pool_interaction_depth', 1)) < 1:
            raise ValueError('late pool requires >=2 slots and >=1 interaction block')
    if m.future_feat.enabled or float(t.lambda_future_feat) != 0 or float(t.get('lambda_intent', 0)) != 0:
        raise ValueError('disable dense future loss AND fixed mean intent loss for this trainer')
    if m.get('spatial_intent', {}).get('enabled', False):
        raise ValueError('spatial branches are excluded')
    if t.get('action_prior', {}).get('enabled', False) or t.get('use_vel_weight', False):
        raise ValueError('action-prior and velocity weighting are excluded')
    if int(t.get('grad_accum_steps', 1)) != 1:
        raise ValueError('this trainer requires grad_accum_steps=1')
    if d.get('format', 'native') != 'native' or d.get('image_aug', False):
        raise ValueError('first version requires native data and image_aug=false')
    if int(c.proprio_len) != 1 or int(c.state_dim) != 4 or int(c.action_dim) != 4:
        raise ValueError('first version requires single-frame Meta-World 4D state/action')
    if not 1 <= int(c.action_execution_horizon) <= int(c.action_chunk_size):
        raise ValueError('invalid execution horizon')
    if m.future_feat.target_layer not in m.vision_encoder.feat_layers:
        raise ValueError('target_layer must be one of feat_layers')
    if float(cfg.planwam.lambda_kd) < 0 or (stage == 'student' and float(cfg.planwam.lambda_kd) <= 0):
        raise ValueError('student lambda_kd must be positive')
    if int(t.epochs) < 1 or int(t.save_interval_epoch) < 1:
        raise ValueError('epochs and save interval must be positive')
    validate_transfer_config(cfg)
    return stage


def input_contract(cfg, stats):
    """Bind teacher targets to input coordinates, not teacher optimizer settings."""
    vtt = Path(cfg.dataset.task_cond_dir)
    return dict(version=VERSION, image_rotation=image_rotation(cfg),
                common=OmegaConf.to_container(cfg.common, resolve=True),
                dataset=OmegaConf.to_container(cfg.dataset, resolve=True),
                model=OmegaConf.to_container(cfg.model, resolve=True),
                stats_sha256=sha256(stats),
                vtt_meta_sha256=sha256(vtt/'meta.json'),
                vtt_sha256={task: sha256(vtt/task/'task_cond.npy') for task in cfg.dataset.task_names})


def load_frozen_teacher(path, contract, spec, device, dtype, allow_partial=False):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('planwam_stage') != 'teacher' or ck.get('planwam_version') != VERSION:
        raise ValueError('expected a checkpoint produced by train_planwam.py teacher stage')
    if not ck.get('epoch_complete', False) and not allow_partial:
        raise ValueError('partial smoke teacher checkpoint cannot supervise a full experiment')
    if ck.get('input_contract') != teacher_input_contract(contract) or ck.get('posterior_spec') != spec:
        raise ValueError('teacher input protocol/stats/VTT/posterior spec does not match student')
    # Loading a frozen label generator must not consume the student's RNG.
    with torch.random.fork_rng(devices=[]):
        module = FutureDrivenPosterior(**spec)
    module.load_state_dict(ck['posterior_state'], strict=True)
    return freeze_module(module.to(device=device, dtype=dtype))


def compute_batch(wrapper, batch, stage, posterior=None, teacher=None, horizon=8,
                  lambda_kd=0.1, time_sampler='logit_normal', time_mu=0., time_sigma=1.):
    """DINO is run once per present/future frame; teacher never sees actions."""
    features = wrapper.get_vision_features(batch['pixel_values'])
    current = wrapper.get_current_intent_features(features)
    future = None
    if stage != 'action_only':
        future = wrapper.get_future_target_features(batch['future_pixel_values'])
    state = batch['state'].to(wrapper.device, wrapper.dtype)
    if state.ndim == 2:
        state = state.unsqueeze(1)
    state = wrapper.normalize_state(state)
    task = batch['task_cond'].to(wrapper.device, wrapper.dtype)
    actions = wrapper.normalize_action(batch['action_sequence'].to(wrapper.device, wrapper.dtype))
    return planwam_flow_loss(wrapper.action_model, current, future, features, state, task, actions,
                             stage, posterior, teacher, horizon, lambda_kd,
                             time_sampler, time_mu, time_sigma)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--stats', required=True)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--teacher-checkpoint', default=None, help='student only; targets, NOT student initialization')
    p.add_argument('--init-teacher-action', default=None,
                   help='opt-in action_init configs only: teacher action weights, NOT optimizer/resume')
    p.add_argument('--resume', default=None, help='recover this same stage; not stage switching')
    p.add_argument('--max-steps', type=int, default=None, help='optimizer-step smoke limit; not a full epoch')
    a = p.parse_args()
    cfg = OmegaConf.load(a.config)
    stage = validate_config(cfg)
    init_enabled, grouped_lr = validate_transfer_config(cfg)
    if a.init_teacher_action and not init_enabled:
        p.error('--init-teacher-action requires planwam.action_init.enabled=true')
    init_path = a.init_teacher_action or cfg.planwam.get('action_init', {}).get('checkpoint')
    if init_enabled:
        if not init_path or not Path(init_path).is_file():
            p.error('action_init requires an existing teacher checkpoint (config or --init-teacher-action)')
        init_path = str(Path(init_path).resolve())
        cfg.planwam.action_init.checkpoint = init_path
    if a.max_steps is not None and a.max_steps < 1:
        p.error('--max-steps must be positive')
    teacher_path = a.teacher_checkpoint or cfg.planwam.get('teacher_checkpoint')
    if stage == 'student' and not teacher_path:
        p.error('student stage requires --teacher-checkpoint')
    if stage != 'student' and teacher_path:
        p.error('teacher checkpoint is only used in student stage')
    run = Path(a.save_dir)
    if not a.resume and (any(run.glob('checkpoint_epoch_*.pt')) or (run/'config.yaml').exists()):
        p.error('output already has a run: use a new directory or --resume')
    seed = int(cfg.training.seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    d, c, m, t = cfg.dataset, cfg.common, cfg.model, cfg.training
    contract = input_contract(cfg, a.stats)
    ds, batch_collate = build_dataset(cfg, stage)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=int(t.batch_size), shuffle=True, drop_last=True,
                        num_workers=int(cfg.system.num_workers), pin_memory=bool(cfg.system.pin_memory),
                        collate_fn=batch_collate, generator=gen, persistent_workers=int(cfg.system.num_workers)>0)
    if len(loader) == 0:
        raise ValueError('dataset too small for batch_size with drop_last=True')
    vision, hidden, regs, patch = ModelFactory.create_vision_encoder(m.vision_encoder.checkpoint_path, dtype, str(device))
    action = ModelFactory.create_action_model(cfg, hidden, len(m.vision_encoder.feat_layers),
                                             task_cond_dim=hidden, patch_size=patch, num_register_tokens=regs)
    action.to(device=device, dtype=dtype).keep_control_gates_fp32()
    spec = dict(vision_dim=hidden, state_dim=int(c.state_dim), task_dim=hidden,
                grid_size=[int(d.image_size[1])//patch, int(d.image_size[0])//patch],
                hidden_dim=int(cfg.planwam.posterior.hidden_size),
                num_heads=int(cfg.planwam.posterior.num_heads),
                horizon_dim=int(cfg.planwam.posterior.horizon_embed_dim))
    posterior = None
    teacher = None
    if stage == 'teacher':
        posterior = FutureDrivenPosterior(**spec).to(device=device, dtype=dtype)
        # Retain legacy model key layout, but the unused causal predictor must not be optimized.
        freeze_module(action.intent_pool)
        freeze_module(action.intent_predictor)
    if stage == 'student':
        teacher = load_frozen_teacher(teacher_path, contract, spec, device, dtype,
                                      allow_partial=a.max_steps is not None)
    init_report = None
    if init_enabled:
        init_report = initialize_action_from_teacher(
            action, init_path, contract, spec, VERSION, apply_weights=not bool(a.resume),
            allow_partial=a.max_steps is not None)
        if stage == 'student' and sha256(teacher_path) != init_report['source_sha256']:
            raise ValueError('this transfer protocol requires the SAME teacher for initialization and KD')
        log.info('Teacher action initialization: source epoch=%d, copied=%d state tensors (%d elements); '
                 'fresh causal predictor=%d tensors; excluded compatibility=%d tensors; applied=%s',
                 init_report['source_epoch'], len(init_report['copied_state_keys']),
                 init_report['copied_state_numel'], len(init_report['fresh_predictor_state_keys']),
                 len(init_report['unused_compatibility_state_keys']), not bool(a.resume))
    # Feature extraction/normalization only; bypass legacy mean-intent loss.
    wrapper = VLAWrapper(vision, action, t.time_sampler, m.vision_encoder.feat_layers,
                         bool(m.vision_encoder.include_cls_register), regs, device, dtype,
                         a.stats, None, int(m.future_feat.target_layer)).to(device)
    params = [v for v in action.parameters() if v.requires_grad]
    if posterior is not None:
        params += list(posterior.parameters())
    optimizer_policy = None
    if grouped_lr:
        optimizer, scheduler, optimizer_policy = build_transfer_optimizer(
            action, t, cfg.planwam.grouped_lr, init_report, int(t.epochs) * len(loader))
        params = [p for group in optimizer.param_groups for p in group['params']]
        for group in optimizer_policy['groups']:
            log.info('Optimizer group %s: %d tensors, %d parameters, LR %.3g -> %.3g',
                     group['name'], group['tensors'], group['numel'], group['lr'], group['lr_min'])
    else:
        # Do not change this branch: old optimizer ordering/scheduler and resume
        # protocol must remain compatible with previously trained checkpoints.
        optimizer = torch.optim.AdamW(params, lr=float(t.learning_rate), betas=tuple(t.betas), weight_decay=float(t.weight_decay))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(t.epochs)*len(loader), eta_min=float(t.lr_min))
        if init_enabled:
            optimizer_policy = dict(kind='legacy_uniform_cosine', lr=float(t.learning_rate),
                                    lr_min=float(t.lr_min), total_steps=int(t.epochs) * len(loader))
    # Includes teacher identity for resume, but is never used to initialize student weights.
    protocol = dict(stage=stage, contract=contract, posterior_spec=spec,
                    training=OmegaConf.to_container(t, resolve=True),
                    lambda_kd=float(cfg.planwam.lambda_kd),
                    teacher_sha256=sha256(teacher_path) if teacher_path else None)
    if init_enabled:
        protocol['action_initialization'] = init_report
        protocol['optimizer_policy'] = optimizer_policy
    start, step = 0, 0
    if a.resume:
        ck = torch.load(a.resume, map_location='cpu', weights_only=False)
        if ck.get('planwam_protocol') != protocol or ck.get('batches_per_epoch') != len(loader):
            raise ValueError('resume protocol/teacher/training/batch-count mismatch')
        action.load_state_dict(ck['model_state'], strict=True)
        if posterior is not None:
            posterior.load_state_dict(ck['posterior_state'], strict=True)
        optimizer.load_state_dict(ck['optimizer_state']); scheduler.load_state_dict(ck['scheduler_state'])
        start, step = int(ck['epoch']), int(ck['global_step'])
        if not ck.get('epoch_complete', True):
            raise ValueError('smoke checkpoint is partial epoch; start full experiment in a fresh directory')
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'])
        gen.set_state(ck['loader_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
    run.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run/'config.yaml')
    (run/'planwam_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    if init_enabled:
        (run/'action_init_report.json').write_text(json.dumps(init_report, indent=2), encoding='utf-8')
    log.info('PlanWAM stage=%s samples=%d batches=%d teacher=%s; initialization=%s',
             stage, len(ds), len(loader), str(teacher_path),
             'resume (no teacher weight reset)' if a.resume else
             'teacher compatible action modules + fresh causal predictor' if init_enabled else
             'fresh action/predictor weights (legacy)')
    for epoch in range(start, int(t.epochs)):
        action.train()
        if posterior is not None:
            posterior.train(); action.intent_pool.eval(); action.intent_predictor.eval()
        total, count = 0., 0
        began = time.perf_counter()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            amp = torch.amp.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
            with amp:
                loss, info = compute_batch(wrapper, batch, stage, posterior, teacher,
                                            int(d.future_frame_offset), float(cfg.planwam.lambda_kd),
                                            t.time_sampler, float(t.time_mu), float(t.time_sigma))
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite PlanWAM loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, float(t.grad_clip_norm), error_if_nonfinite=True)
            optimizer.step(); scheduler.step()
            step += 1; count += 1; total += loss.detach().item()
            if count == 1 or count % max(1, len(loader)//4) == 0 or count == len(loader):
                log.info('Stage=%s Epoch [%d/%d] Step [%d/%d] Action: %.5f KD: %.5f Total: %.5f LatentNorm: %.3f LatentStd: %.4f TargetNorm: %.3f GradNorm: %.3f LR: %.3g',
                         stage, epoch+1, t.epochs, count, len(loader), info['loss_action'], info['loss_kd'],
                         loss.item(), info['latent_norm'], info['latent_std'], info['target_norm'], float(norm), scheduler.get_last_lr()[0])
                if grouped_lr:
                    log.info('LRGroups %s', ' '.join(f"{g['name']}={g['lr']:.6g}" for g in optimizer.param_groups))
            if a.max_steps and step >= a.max_steps:
                break
        complete = count == len(loader)
        log.info('Epoch %d avg_loss=%.5f time=%.1fs complete=%s', epoch+1, total/max(count,1), time.perf_counter()-began, complete)
        if (epoch+1) % int(t.save_interval_epoch) == 0 or epoch+1 == int(t.epochs) or (a.max_steps and step >= a.max_steps):
            action_state = action.state_dict()
            ck = dict(model_state=action_state, posterior_state=posterior.state_dict() if posterior is not None else None,
                      posterior_spec=spec, input_contract=contract, planwam_version=VERSION, planwam_stage=stage,
                      planwam_protocol=protocol, image_rotation_degrees=image_rotation(cfg),
                      optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                      epoch=epoch+1, global_step=step, epoch_complete=complete, batches_per_epoch=len(loader),
                      python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                      loader_rng=gen.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
            if init_enabled:
                ck['action_initialization'] = init_report
                ck['optimizer_policy'] = optimizer_policy
            path = run/f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(ck, path); log.info('Saved %s', path)
        if a.max_steps and step >= a.max_steps:
            break


if __name__ == '__main__':
    main()
