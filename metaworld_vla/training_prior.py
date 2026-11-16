"""Opt-in C/N/R source construction and versioned offline-prior validation."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cache_spec(cfg, stats):
    orientation = 'robotwin_legacy_no_rotation' if cfg.dataset.get('type') == 'robotwin' else 180
    return dict(version=1, image_rotation=orientation, stats_sha256=sha256(stats),
                common=OmegaConf.to_container(cfg.common, resolve=True),
                model=OmegaConf.to_container(cfg.model, resolve=True),
                dataset=OmegaConf.to_container(cfg.dataset, resolve=True))


def episode_record(ds, ep):
    p = Path(ep['path']).resolve()
    stat = p.stat()
    return dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                length=ep['length'], task=ep['task'])


def cache_key(path):
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest() + '.npy'


def prior_options(cfg):
    p = cfg.training.get('action_prior', {})
    if not p.get('enabled', False):
        return None
    mode = str(p.get('mode', 'cold'))
    if mode not in ('cold', 'scale', 'tail'):
        raise ValueError('action_prior.mode must be cold, scale or tail')
    probability = float(p.get('cold_probability', .5))
    sigma, strength = float(p.get('sigma', .5)), float(p.get('strength', 1))
    if not 0 <= probability <= 1 or not all(math.isfinite(x) and x >= 0 for x in (sigma, strength)):
        raise ValueError('invalid action_prior mixing probability/scale')
    h, e = int(cfg.common.action_chunk_size), int(cfg.common.action_execution_horizon)
    if not 0 < e < h:
        raise ValueError('prior training needs 0 < execution_horizon < chunk_size')
    if cfg.model.future_feat.enabled:
        raise ValueError('this source protocol keeps the original future-feature decoder disabled')
    options = dict(mode=mode, cold_probability=probability, sigma=sigma, strength=strength,
                execution=e, overlap=h-e, cache_dir=p.get('cache_dir'),
                selection_seed=int(p.get('selection_seed', 100003)))
    # Do not add a new key to old protocols (keeps their resume signatures stable).
    if p.get('cache_reference_config'):
        options['cache_reference_config'] = str(p.cache_reference_config)
    return options


def expected_cache_spec(cfg, stats, options):
    reference = options.get('cache_reference_config')
    if not reference:
        return cache_spec(cfg, stats)
    ref = OmegaConf.load(reference)
    student, teacher = cache_spec(cfg, stats), cache_spec(ref, stats)
    # Explicit transfer of the SAME teacher priors to a student with G2 added.
    # Still reject changes to images, tasks, normalization, H/E or base backbone.
    student_model = dict(student['model'])
    teacher_model = dict(teacher['model'])
    student_model.pop('global_intent', None)
    old_intent = teacher_model.pop('global_intent', {})
    if old_intent.get('enabled', False):
        raise ValueError('cache reference must be the original non-G2 teacher')
    comparable_student, comparable_teacher = dict(student), dict(teacher)
    comparable_student['model'], comparable_teacher['model'] = student_model, teacher_model
    if comparable_student != comparable_teacher:
        raise ValueError('G2 cache transfer permits only global_intent model differences')
    return teacher


class PriorDataset(Dataset):
    def __init__(self, base, cfg, stats, options):
        self.base, self.options = base, options
        self.records = {}
        self.root = None
        if options['mode'] == 'tail':
            if not options['cache_dir']:
                raise ValueError('tail training requires action_prior.cache_dir')
            self.root = Path(options['cache_dir'])
            manifest = json.loads((self.root / 'manifest.json').read_text())
            if manifest['spec'] != expected_cache_spec(cfg, stats, options) or not manifest.get('complete'):
                raise ValueError('prior cache incomplete or config/stats mismatch')
            self.records = manifest['episodes']
            if len(self.records) != len(base.episodes):
                raise ValueError('prior cache episode count mismatch')
            for ep in base.episodes:
                key = cache_key(ep['path'])
                rec = self.records.get(key, {})
                if rec.get('source') != episode_record(base, ep):
                    raise ValueError(f'cache episode identity mismatch: {ep["path"]}')
                array = np.load(self.root / key, mmap_mode='r')
                if sha256(self.root / key) != rec.get('sha256'):
                    raise ValueError(f'prior cache checksum mismatch: {key}')
                if array.shape != (ep['length'], options['overlap'], base.action_dim) or array.dtype != np.float32:
                    raise ValueError(f'invalid prior cache shape/dtype: {key}')
                if not np.isfinite(array).all():
                    raise ValueError(f'nonfinite prior cache: {key}')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        ep_id, anchor = self.base.indices[index]
        ep = self.base.episodes[ep_id]
        prior = torch.zeros_like(item['action_sequence'])
        valid = torch.zeros(prior.shape[0], 1, dtype=torch.bool)
        # Near termination only recorded future positions have a usable prior.
        if anchor >= self.options['execution']:
            count = min(self.options['overlap'], ep['length'] - anchor)
            valid[:count] = True
            if self.root is not None:
                values = np.load(self.root / cache_key(ep['path']), mmap_mode='r')
                prior[:count] = torch.from_numpy(np.array(values[anchor, :count], copy=True))
        item['cached_prior_norm'] = prior
        item['cached_prior_valid'] = valid
        return item


def prepare_source_batch(batch, options, generator):
    """Preserve global RNG: the model still draws its original noise/time stream."""
    prior = batch.pop('cached_prior_norm')
    valid = batch.pop('cached_prior_valid')
    selected = torch.rand(prior.shape[0], generator=generator) >= options['cold_probability']
    mask = valid & selected[:, None, None]
    mean = torch.zeros_like(prior)
    if options['mode'] == 'tail':
        mean = torch.where(mask, options['strength'] * prior, mean)
    scale = torch.where(mask, torch.full_like(prior, options['sigma']), torch.ones_like(prior))
    batch['fm_source_mean'], batch['fm_source_scale'] = mean, scale
    return float(mask.any(dim=1).float().mean())
