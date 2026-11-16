"""RoboTwin-only teacher action initialization and per-group cosine LR.

Intentionally independent of metaworld_vla/planwam_transfer.py.

No changes to model keys, forward, losses, or legacy training defaults.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

TRANSFER_VERSION = 'teacher_action_init_v1'
PREDICTOR_ROOTS = frozenset({'intent_pool', 'intent_predictor'})
UNUSED_ROOTS = frozenset({'task_cond_proj', 'type_emb_task_cond', 'intent_gate'})
# Only components active in the teacher's action forward may be transferred.
ACTION_ROOTS = frozenset({
    'time_mlp', 'action_proj', 'proprio_proj', 'action_pos_emb', 'proprio_pos_emb',
    'type_emb_action', 'type_emb_proprio', 'type_emb_dino_layers', 'register_tokens',
    'intent_to_token', 'type_emb_intent', 'concat_fusion', 'dino_adapters',
    'blocks', 'final_norm', 'output_proj',
})


def _root(name):
    return name.split('.', 1)[0]


def validate_transfer_config(cfg):
    """Old configs have neither switch and stay on the original code path."""
    init = cfg.planwam.get('action_init', {})
    groups = cfg.planwam.get('grouped_lr', {})
    enabled = bool(init.get('enabled', False))
    grouped = bool(groups.get('enabled', False))
    if enabled and cfg.planwam.stage not in ('student', 'action_only'):
        raise ValueError('teacher action initialization is only for student/action_only')
    if grouped and not enabled:
        raise ValueError('grouped_lr requires action_init.enabled=true')
    if grouped:
        for prefix in ('inherited', 'predictor'):
            high = float(groups[f'{prefix}_lr'])
            low = float(groups[f'{prefix}_lr_min'])
            if not (math.isfinite(high) and math.isfinite(low) and 0 <= low <= high and high > 0):
                raise ValueError(f'invalid {prefix} LR: require 0 <= lr_min <= lr and lr > 0')
    return enabled, grouped


def initialize_action_from_teacher(action, path, contract, spec, expected_version,
                                   *, apply_weights=True, allow_partial=False):
    """Copy an explicit whitelist; leave causal/unused branches untouched.

    Resume uses apply_weights=False: validate source/provenance, never reset the
    resumed model to teacher weights. Optimizer/scheduler are never loaded here.
    """
    path = Path(path).resolve()
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('planwam_stage') != 'teacher' or ck.get('planwam_version') != expected_version:
        raise ValueError('action initialization requires a matching PlanWAM teacher checkpoint')
    if not ck.get('epoch_complete', False) and not allow_partial:
        raise ValueError('partial teacher checkpoint cannot initialize a full experiment')
    if ck.get('input_contract') != contract or ck.get('posterior_spec') != spec:
        raise ValueError('action-init teacher input/stats/VTT/model/posterior protocol mismatch')
    source = ck['model_state']
    current = action.state_dict()
    if set(source) != set(current):
        raise ValueError('teacher/student action state keys differ; do not silently transfer partial architectures')
    copied, retained, unused = [], [], []
    for key in current:
        root = _root(key)
        if root in ACTION_ROOTS:
            floating_cast = source[key].is_floating_point() and current[key].is_floating_point()
            if source[key].shape != current[key].shape or (
                    source[key].dtype != current[key].dtype and not floating_cast):
                raise ValueError(f'incompatible teacher action tensor: {key}')
            if source[key].is_floating_point() and not torch.isfinite(source[key]).all():
                raise FloatingPointError(f'non-finite teacher action tensor: {key}')
            copied.append(key)
        elif root in PREDICTOR_ROOTS:
            retained.append(key)
        elif root in UNUSED_ROOTS:
            unused.append(key)
        else:
            raise ValueError(f'unclassified model key {key}; review transfer whitelist first')
    if not copied or not retained:
        raise ValueError('expected both transferable action modules and fresh causal predictor modules')
    if apply_weights:
        result = action.load_state_dict({k: source[k] for k in copied}, strict=False)
        if set(result.missing_keys) != set(retained + unused) or result.unexpected_keys:
            raise RuntimeError('unexpected selective action initialization result')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return dict(version=TRANSFER_VERSION, source_checkpoint=str(path), source_sha256=h.hexdigest(),
                source_epoch=int(ck['epoch']), source_epoch_complete=bool(ck.get('epoch_complete', False)),
                copied_state_keys=sorted(copied), fresh_predictor_state_keys=sorted(retained),
                unused_compatibility_state_keys=sorted(unused),
                copied_state_numel=sum(current[k].numel() for k in copied))


def build_transfer_optimizer(action, training, groups, report, total_steps):
    """Two disjoint active groups, each with its own floor and same cosine phase.

    Unused direct-VTT compatibility parameters are retained in the state dict but
    not placed into this opt-in optimizer. No active parameter may be omitted.
    """
    if total_steps < 1:
        raise ValueError('total_steps must be positive')
    copied = set(report['copied_state_keys'])
    buckets = {'inherited_action': [], 'new_predictor': []}
    names = {key: [] for key in buckets}
    unused = []
    for name, param in action.named_parameters():
        if not param.requires_grad:
            continue
        if _root(name) in UNUSED_ROOTS:
            unused.append(name)
            continue
        if _root(name) in PREDICTOR_ROOTS:
            group = 'new_predictor'
        elif name in copied and _root(name) in ACTION_ROOTS:
            group = 'inherited_action'
        else:
            raise ValueError(f'active parameter not assigned to an LR group: {name}')
        names[group].append(name)
        buckets[group].append(param)
    if not all(buckets.values()):
        raise ValueError('both inherited and predictor LR groups must be nonempty')
    identities = [id(p) for params in buckets.values() for p in params]
    if len(identities) != len(set(identities)):
        raise ValueError('parameter duplicated across optimizer groups')
    configs, summaries = [], []
    for name, prefix in (('inherited_action', 'inherited'), ('new_predictor', 'predictor')):
        high, low = float(groups[f'{prefix}_lr']), float(groups[f'{prefix}_lr_min'])
        if not (math.isfinite(high) and math.isfinite(low) and high > 0 and 0 <= low <= high):
            raise ValueError(f'invalid group LR: {name}')
        configs.append(dict(params=buckets[name], lr=high, name=name))
        summaries.append(dict(name=name, lr=high, lr_min=low, tensors=len(names[name]),
                              numel=sum(p.numel() for p in buckets[name]), parameter_names=names[name]))
    optimizer = torch.optim.AdamW(configs, lr=float(training.learning_rate),
                                  betas=tuple(training.betas), weight_decay=float(training.weight_decay))
    lambdas = []
    for group in summaries:
        floor = group['lr_min'] / group['lr']
        lambdas.append(lambda step, floor=floor: floor + (1 - floor) * 0.5 * (
            1 + math.cos(math.pi * min(max(step, 0), total_steps) / total_steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
    protocol = dict(kind='per_group_cosine_v1', total_steps=int(total_steps), groups=summaries,
                    unused_trainable_compatibility_names=unused)
    return optimizer, scheduler, protocol
