"""Paired S/S vs S/T offline diagnostic. Never an online/oracle success rate.

S/S: student action network + causal student latent.
S/T: the SAME student action network + privileged teacher preprojection latent.
With --include-teacher-dit, also evaluate T/S and T/T with the teacher action
network, including its OWN adapter, intent projection and action readout.
Without this flag the teacher action network is not loaded (legacy behavior).
No training or rollout code changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'metaworld_vla'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import collate_fn
from dataset_native import NativeMetaWorldDataset
from train_planwam import VERSION, input_contract, load_frozen_teacher, sha256, validate_config
from models.intent_pool_contract import uses_late_pool, teacher_input_contract
from models.model_runner import ModelFactory, VLAWrapper
from models.vla_model_fm import fixed_layer_norm

log = logging.getLogger(__name__)


def checked(x, name):
    if not torch.isfinite(x).all():
        raise FloatingPointError(f'non-finite {name}')
    return x


@torch.no_grad()
def paired_velocities(action, features, state, task, actions, noise, latents,
                      taus=(0.1, 0.5, 0.9)):
    """Per-anchor FM velocity MSE, identical interpolation and target per branch."""
    if actions.shape != noise.shape:
        raise ValueError('noise must match action shape')
    metrics = {}
    for tau in taus:
        t = torch.full((actions.shape[0],), tau, device=actions.device, dtype=torch.float32)
        xt = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
        target = (actions - noise).float()
        for name, z in latents.items():
            pred = action(t=t, noisy_actions=xt.clone(), dino_features_list=features,
                          qpos_history=state, task_cond=task, global_intent=z)['final_pred']
            error = (checked(pred, f'{name} velocity').float() - target).square()
            metrics[f'{name}/velocity_mse_tau_{tau:g}'] = error.mean((1, 2))
    for name in latents:
        metrics[f'{name}/velocity_mse_tau_mean'] = torch.stack(
            [metrics[f'{name}/velocity_mse_tau_{tau:g}'] for tau in taus]).mean(0)
    return metrics


@torch.no_grad()
def paired_generate(action, features, state, task, noise, latents, steps=10):
    """Cold Gaussian -> Euler integration, matching eval.Policy.chunk (no reuse).

    Latents/vision are computed once and held fixed for all Euler steps. Each
    branch starts from its own clone of the exact same Gaussian tensor.
    """
    if steps < 1:
        raise ValueError('steps must be positive')
    grid = torch.linspace(0, 1, steps + 1, device=noise.device, dtype=noise.dtype)
    generated = {}
    for name, z in latents.items():
        x = noise.clone()
        for i in range(steps):
            t = grid[i].expand(noise.shape[0])
            pred = action(t=t, noisy_actions=x, dino_features_list=features,
                          qpos_history=state, task_cond=task, global_intent=z)['final_pred']
            x = checked(x + pred * (grid[i + 1] - grid[i]), f'{name} Euler state')
        generated[name] = x
    return generated


def chunk_metrics(generated, target, valid_steps, execution_horizon, denormalize):
    """Offline imitation errors, NOT rewards/poses/success. Retain padded targets.

    The all-step metric uses the exact padded training target. The valid-only
    metric excludes synthetic tail targets for diagnosis only, never for training.
    Physical XYZ are actuator commands, not measured Cartesian endpoint errors.
    """
    length = target.shape[1]
    if not 1 <= execution_horizon <= length:
        raise ValueError('invalid execution horizon')
    valid_steps = torch.as_tensor(valid_steps, device=target.device)
    if not ((valid_steps >= 1) & (valid_steps <= length)).all():
        raise ValueError('invalid recorded action lengths')
    mask = torch.arange(length, device=target.device)[None] < valid_steps[:, None]
    truth = denormalize(target).float()
    metrics = {}
    for name, pred in generated.items():
        squared = (pred.float() - target.float()).square()
        metrics[f'{name}/chunk_mse_normalized'] = squared.mean((1, 2))
        metrics[f'{name}/prefix_mse_normalized'] = squared[:, :execution_horizon].mean((1, 2))
        metrics[f'{name}/recorded_steps_mse_normalized'] = (
            squared.mean(-1) * mask).sum(1) / valid_steps
        for step in range(length):
            metrics[f'{name}/step_{step + 1}_mse_normalized'] = squared[:, step].mean(-1)
        physical = denormalize(pred).float()
        for label, command in (('raw', physical), ('clipped', physical.clamp(-1, 1))):
            err = (command - truth).square()
            metrics[f'{name}/command_{label}_xyz_mse'] = err[:, :, :3].mean((1, 2))
            metrics[f'{name}/command_{label}_gripper_mse'] = err[:, :, 3].mean(1)
    return metrics


def summarize(rows):
    """Per-anchor descriptive statistics. Negative st-ss means S/T has less MSE."""
    if not rows:
        return dict(samples=0, metrics={}, paired={})
    keys = rows[0]['metrics'].keys()
    values = {key: np.asarray([row['metrics'][key] for row in rows], dtype=float) for key in keys}
    metrics = {key: dict(mean=float(v.mean()), std=float(v.std()), n=len(v)) for key, v in values.items()}
    paired = {}
    for key, ss in values.items():
        if not key.startswith('ss/') or 'st/' + key[3:] not in values:
            continue
        st = values['st/' + key[3:]]
        diff = st - ss
        paired[key[3:]] = dict(ss_mean=float(ss.mean()), st_mean=float(st.mean()),
                              delta_st_minus_ss=float(diff.mean()),
                              relative_mean_reduction=float(1 - st.mean() / ss.mean()) if ss.mean() > 0 else None,
                              st_better_fraction=float((diff < -1e-8).mean()),
                              tied_fraction=float((np.abs(diff) <= 1e-8).mean()))
    return dict(samples=len(rows), metrics=metrics, paired=paired)


def summarize_matrix(rows):
    """Keep the original ss/st summary and add explicitly directed 2x2 pairs."""
    summary = summarize(rows)
    comparisons = {}
    for candidate, reference in (('st', 'ss'), ('ts', 'tt'), ('tt', 'ss'), ('ts', 'ss')):
        metrics = {}
        if rows:
            for key in rows[0]['metrics']:
                if not key.startswith(reference + '/'):
                    continue
                metric = key.split('/', 1)[1]
                other = candidate + '/' + metric
                if other not in rows[0]['metrics']:
                    continue
                ref = np.asarray([r['metrics'][key] for r in rows], dtype=float)
                cand = np.asarray([r['metrics'][other] for r in rows], dtype=float)
                diff = cand - ref
                metrics[metric] = dict(reference_mean=float(ref.mean()), candidate_mean=float(cand.mean()),
                                       delta_candidate_minus_reference=float(diff.mean()),
                                       relative_mean_reduction=float(1 - cand.mean() / ref.mean()) if ref.mean() > 0 else None,
                                       candidate_better_fraction=float((diff < -1e-8).mean()),
                                       tied_fraction=float((np.abs(diff) <= 1e-8).mean()))
        comparisons[f'{candidate}_vs_{reference}'] = dict(candidate=candidate, reference=reference, metrics=metrics)
    summary['comparisons'] = comparisons
    return summary


@torch.no_grad()
def evaluate_teacher_pairs(teacher_action, features, state, task, actions, noise,
                           student_latent, teacher_latent, taus, generation_steps=None):
    """Use the teacher decoder but NEVER its untrained causal intent predictor."""
    latents = dict(ts=student_latent, tt=teacher_latent)
    metrics = paired_velocities(teacher_action, features, state, task, actions, noise, latents, taus)
    generated = None
    if generation_steps is not None:
        generated = paired_generate(teacher_action, features, state, task, noise, latents, generation_steps)
    return metrics, generated


def select_anchors(ds, tasks, samples_per_task, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, (episode, _) in enumerate(ds.indices):
        groups[ds.episodes[episode]['task']].append(i)
    chosen = []
    for task in tasks:
        if len(groups[task]) < samples_per_task:
            raise ValueError(f'{task}: fewer than {samples_per_task} anchors; lower --samples-per-task')
        chosen.extend(rng.sample(groups[task], samples_per_task))
    rng.shuffle(chosen)
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='student run config.yaml; must match checkpoint contract')
    p.add_argument('--stats', required=True)
    p.add_argument('--checkpoint', required=True, help='student checkpoint')
    p.add_argument('--teacher-checkpoint', required=True, help='the exact teacher used to train this student')
    p.add_argument('--output', required=True, help='new JSON path; existing output is not overwritten')
    p.add_argument('--samples-per-task', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--tasks', nargs='+', help='optional subset; does not change model/input contract')
    p.add_argument('--taus', type=float, nargs='+', default=[0.1, 0.5, 0.9])
    p.add_argument('--seed', type=int, default=0, help='offline anchor selection and paired Gaussian noise, not env seed')
    p.add_argument('--generate-chunks', action='store_true', help='also integrate full chunks; substantially more DiT calls')
    p.add_argument('--include-teacher-dit', action='store_true', help='full S/S,S/T,T/S,T/T matrix; loads a second action network')
    p.add_argument('--num-inference-steps', type=int, default=10)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--allow-smoke-checkpoint', action='store_true', help='development only: permit partial-epoch checkpoints')
    a = p.parse_args()
    if min(a.samples_per_task, a.batch_size, a.num_inference_steps, a.cpu_threads) < 1:
        p.error('counts must be positive')
    if a.seed < 0 or a.seed >= 2**32:
        p.error('seed must be in [0, 2**32)')
    if any(not 0 < tau < 1 for tau in a.taus) or len(set(a.taus)) != len(a.taus):
        p.error('taus must be distinct numbers strictly between 0 and 1')
    out = Path(a.output)
    if out.exists():
        p.error('output exists; choose a new JSON path')
    cfg = OmegaConf.load(a.config)
    if validate_config(cfg) != 'student':
        p.error('provide student configuration')
    tasks = list(a.tasks or cfg.dataset.task_names)
    if len(set(tasks)) != len(tasks) or set(tasks) - set(cfg.dataset.task_names):
        p.error('tasks must be unique and belong to the student config')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.cpu_threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
    contract = input_contract(cfg, a.stats)
    teacher_sha = sha256(a.teacher_checkpoint)
    student_sha = sha256(a.checkpoint)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck.get('planwam_stage') != 'student' or ck.get('planwam_version') != VERSION:
        raise ValueError('expected a train_planwam.py student checkpoint')
    if not ck.get('epoch_complete', False) and not a.allow_smoke_checkpoint:
        raise ValueError('partial student checkpoint; use a complete epoch')
    if ck.get('input_contract') != contract:
        raise ValueError('student config/stats/VTT/input contract mismatch')
    if ck.get('planwam_protocol', {}).get('teacher_sha256') != teacher_sha:
        raise ValueError('teacher SHA does not match the teacher used for student training')
    teacher = load_frozen_teacher(a.teacher_checkpoint, contract, ck['posterior_spec'],
                                  device, dtype, a.allow_smoke_checkpoint)
    d, c, m = cfg.dataset, cfg.common, cfg.model
    vision, hidden, regs, patch = ModelFactory.create_vision_encoder(
        m.vision_encoder.checkpoint_path, dtype, str(device))
    action = ModelFactory.create_action_model(cfg, hidden, len(m.vision_encoder.feat_layers),
                                             task_cond_dim=hidden, patch_size=patch, num_register_tokens=regs)
    action.load_state_dict(ck['model_state'], strict=True)
    action.to(device=device, dtype=dtype).keep_control_gates_fp32()
    action.eval().requires_grad_(False)
    student_epoch = int(ck['epoch'])
    del ck
    teacher_action, teacher_epoch = None, None
    if a.include_teacher_dit:
        # The posterior/input protocol has already been checked above. Load the
        # SAME teacher checkpoint's complete action network, not a student copy
        # with only the output head swapped. Keep one shared frozen DINO encoder.
        teacher_ck = torch.load(a.teacher_checkpoint, map_location='cpu', weights_only=False)
        teacher_cfg = cfg
        if uses_late_pool(contract):
            # Instantiate the original teacher pool, not the new student pool.
            teacher_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            teacher_cfg.model = OmegaConf.create(teacher_input_contract(contract)['model'])
        with torch.random.fork_rng(devices=[]):
            teacher_action = ModelFactory.create_action_model(
                teacher_cfg, hidden, len(m.vision_encoder.feat_layers), task_cond_dim=hidden,
                patch_size=patch, num_register_tokens=regs)
        teacher_action.load_state_dict(teacher_ck['model_state'], strict=True)
        teacher_action.to(device=device, dtype=dtype).keep_control_gates_fp32()
        teacher_action.eval().requires_grad_(False)
        teacher_epoch = int(teacher_ck['epoch'])
        del teacher_ck
    wrapper = VLAWrapper(vision, action, cfg.training.time_sampler, m.vision_encoder.feat_layers,
                         bool(m.vision_encoder.include_cls_register), regs, device, dtype,
                         a.stats, None, int(m.future_feat.target_layer)).to(device).eval()
    ds = NativeMetaWorldDataset(str(d.dataset_dir), list(d.task_names), int(c.action_chunk_size),
                               int(c.state_dim), tuple(d.image_size), str(d.task_cond_dir), True,
                               int(d.future_frame_offset), False, camera_name=str(d.camera_names[0]))
    selected = select_anchors(ds, tasks, a.samples_per_task, a.seed)
    generator = torch.Generator(device=device).manual_seed(a.seed)
    rows = []
    log.info('Offline paired diagnostic: %d anchors, %d tasks; device=%s dtype=%s generation=%s full_matrix=%s',
             len(selected), len(tasks), device, dtype, a.generate_chunks, a.include_teacher_dit)
    for start in range(0, len(selected), a.batch_size):
        ids = selected[start:start + a.batch_size]
        batch = collate_fn([ds[i] for i in ids])
        metadata = []
        for i in ids:
            episode, anchor = ds.indices[i]
            ep = ds.episodes[episode]
            future_index = min(anchor + int(d.future_frame_offset), ep['length'] - 1)
            metadata.append(dict(dataset_index=i, task=ep['task'], episode_index=episode,
                                 episode_path=ep['path'], anchor=anchor, episode_length=ep['length'],
                                 future_index=future_index, realized_future_offset=future_index - anchor,
                                 future_clipped=future_index < anchor + int(d.future_frame_offset),
                                 action_valid_steps=min(int(c.action_chunk_size), ep['length'] - anchor)))
        amp = torch.amp.autocast('cuda', dtype=dtype) if device.type == 'cuda' else nullcontext()
        with torch.no_grad(), amp:
            features = wrapper.get_vision_features(batch['pixel_values'])
            current = wrapper.get_current_intent_features(features)
            future = wrapper.get_future_target_features(batch['future_pixel_values'])
            state = wrapper.normalize_state(batch['state'].to(device, dtype).unsqueeze(1))
            task = batch['task_cond'].to(device, dtype)
            actions = wrapper.normalize_action(batch['action_sequence'].to(device, dtype))
            ss = checked(action.predict_global_intent(current, state, task, horizon=int(d.future_frame_offset)), 'student latent')
            st = checked(teacher(current, future, state, task, int(d.future_frame_offset)), 'teacher latent')
            # Replace the RAW preprojection latent. Student intent_to_token applies
            # its own normalization/projection identically in both branches.
            latents = dict(ss=ss, st=st)
            noise = torch.randn(actions.shape, generator=generator, device=device, dtype=dtype)
            metrics = paired_velocities(action, features, state, task, actions, noise, latents, a.taus)
            ss_norm, st_norm = fixed_layer_norm(ss.float()), fixed_layer_norm(st.float())
            metrics['latent/normalized_mse'] = (ss_norm - st_norm).square().mean(-1)
            metrics['latent/normalized_cosine'] = torch.nn.functional.cosine_similarity(ss_norm, st_norm, dim=-1)
            metrics['latent/student_raw_norm'] = ss.float().norm(dim=-1)
            metrics['latent/teacher_raw_norm'] = st.float().norm(dim=-1)
            metrics['latent/student_projected_token_mse_vs_teacher'] = (
                action.intent_to_token(ss).float() - action.intent_to_token(st).float()).square().mean(-1)
            if a.generate_chunks:
                generated = paired_generate(action, features, state, task, noise, latents, a.num_inference_steps)
                metrics.update(chunk_metrics(generated, actions, [item['action_valid_steps'] for item in metadata],
                                             int(c.action_execution_horizon), wrapper.denormalize_action))
            if teacher_action is not None:
                extra, generated = evaluate_teacher_pairs(
                    teacher_action, features, state, task, actions, noise, ss, st, a.taus,
                    a.num_inference_steps if a.generate_chunks else None)
                metrics.update(extra)
                if generated is not None:
                    metrics.update(chunk_metrics(generated, actions, [item['action_valid_steps'] for item in metadata],
                                                 int(c.action_execution_horizon), wrapper.denormalize_action))
        cpu_metrics = {key: checked(value, key).float().cpu().tolist() for key, value in metrics.items()}
        for j, item in enumerate(metadata):
            rows.append(dict(**item, metrics={key: value[j] for key, value in cpu_metrics.items()}))
        log.info('Completed %d/%d anchors', len(rows), len(selected))
    summary_fn = summarize_matrix if a.include_teacher_dit else summarize
    per_task = {task: summary_fn([row for row in rows if row['task'] == task]) for task in tasks}
    branches = dict(ss='student_action_network + student_raw_latent', st='SAME_student_action_network + teacher_raw_latent')
    if a.include_teacher_dit:
        branches.update(ts='teacher_action_network + student_raw_latent', tt='SAME_teacher_action_network + teacher_raw_latent')
    result = dict(kind='offline_2x2_latent_decoder_swap_not_rollout_success' if a.include_teacher_dit else 'offline_student_latent_swap_not_rollout_success', version=VERSION,
                  branches=branches,
                  config=str(Path(a.config).resolve()), config_sha256=sha256(a.config),
                  checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_sha256=student_sha, student_epoch=student_epoch,
                  teacher_checkpoint=str(Path(a.teacher_checkpoint).resolve()), teacher_sha256=teacher_sha,
                  include_teacher_dit=a.include_teacher_dit, teacher_epoch=teacher_epoch,
                  stats_sha256=sha256(a.stats), seed=a.seed, batch_size=a.batch_size, samples_per_task=a.samples_per_task,
                  sampling='without replacement per task; all training anchors eligible; no validation split',
                  taus=a.taus, generate_chunks=a.generate_chunks,
                  num_inference_steps=a.num_inference_steps if a.generate_chunks else None,
                  action_execution_horizon=int(c.action_execution_horizon), action_chunk_size=int(c.action_chunk_size),
                  future_frame_offset=int(d.future_frame_offset), action_reuse=False,
                  device=str(device), dtype=str(dtype), torch_version=str(torch.__version__),
                  allow_smoke_checkpoint=a.allow_smoke_checkpoint,
                  caveats=['Offline expert future, not future caused by generated actions.',
                           'Training-data diagnostic, not generalization or closed-loop success.',
                           'S/T can be worse due to student readout distribution mismatch; not a strict oracle upper bound.',
                           'One Gaussian draw per anchor; varying seed changes anchors and noise.',
                           'Batch size/hardware can change noise draws or numerical results; keep fixed for comparisons.',
                           'Command XYZ MSE is not robot endpoint position error.'],
                  overall=summary_fn(rows), per_task=per_task,
                  strata=dict(full_horizon=summary_fn([r for r in rows if not r['future_clipped']]),
                              clipped_future=summary_fn([r for r in rows if r['future_clipped']])),
                  samples=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 'x' also prevents a concurrent process from overwriting an existing run.
    with out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps(result['overall']['comparisons'] if a.include_teacher_dit else result['overall']['paired'],
                     indent=2, allow_nan=False))
    print('Saved', out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
