"""Batch-generate causal normalized tails from a frozen baseline (no expert copying)."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from eval import Policy
from dataset_native import NativeMetaWorldDataset
from dataset import collate_fn
from training_prior import sha256, cache_spec, cache_key, episode_record


def save_manifest(root, manifest):
    path = root / 'manifest.json.tmp'
    path.write_text(json.dumps(manifest, indent=2))
    path.replace(root / 'manifest.json')


@torch.no_grad()
def generate(policy, batch, steps, generator):
    state = policy.wrapper.normalize_state(batch['state'].to(policy.device, policy.dtype))
    if state.ndim == 2:
        state = state.unsqueeze(1)
    pixels = batch['pixel_values'].to(policy.device, policy.dtype)
    features = policy.wrapper.get_vision_features(pixels)
    task = batch.get('task_cond')
    if task is not None:
        task = task.to(policy.device, policy.dtype)
    cc = policy.cfg.common
    x = torch.randn(len(state), int(cc.action_chunk_size), int(cc.action_dim),
                    device=policy.device, dtype=policy.dtype, generator=generator)
    times = torch.linspace(0, 1, steps+1, device=policy.device, dtype=policy.dtype)
    for i in range(steps):
        pred = policy.action(t=times[i].expand(len(state)), noisy_actions=x,
                             qpos_history=state, dino_features_list=features,
                             task_cond=task)['final_pred']
        x = x + pred * (times[i+1]-times[i])
    return x.float().cpu().numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--stats', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=321)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.batch_size <= 0 or a.workers < 0:
        p.error('invalid batch size/workers')
    cfg = OmegaConf.load(a.config)
    is_robotwin = cfg.dataset.get('type') == 'robotwin'
    if (not is_robotwin and cfg.dataset.get('format') != 'native') or cfg.dataset.get('image_aug', False):
        p.error('initial cache generator requires native data and image_aug=false')
    if cfg.model.get('global_intent', {}).get('enabled', False) or cfg.model.future_feat.enabled:
        p.error('use the easy no-auxiliary baseline as frozen teacher')
    h, e = int(cfg.common.action_chunk_size), int(cfg.common.action_execution_horizon)
    if not 0 < e < h:
        p.error('need 0 < execution horizon < chunk size')
    dc = cfg.dataset
    if is_robotwin:
        from dataloader.dataset import create_dataset
        from robotwin_action_prior import RobotWinPriorAdapter
        ds = RobotWinPriorAdapter(create_dataset(cfg, val=False))
    else:
        ds = NativeMetaWorldDataset(str(dc.dataset_dir), list(dc.task_names), h, int(cfg.common.state_dim),
               tuple(dc.image_size), dc.get('task_cond_dir') if cfg.model.use_task_cond else None,
               False, int(dc.future_frame_offset), False, str(dc.camera_names[0]))
    spec = cache_spec(cfg, a.stats)
    teacher = dict(checkpoint=str(Path(a.checkpoint).resolve()), sha256=sha256(a.checkpoint),
                   batch_size=a.batch_size, seed=a.seed, nfe=int(cfg.common.num_inference_steps),
                   rng='per_episode_v1')
    root = Path(a.output_dir)
    dataset_records = [episode_record(ds, ep) for ep in ds.episodes]
    if (root / 'manifest.json').exists():
        if not a.resume:
            p.error('cache exists; use --resume or a new directory')
        manifest = json.loads((root / 'manifest.json').read_text())
        if manifest['spec'] != spec or manifest['teacher'] != teacher:
            p.error('cache config/stats/teacher/batch/seed mismatch')
        if manifest.get('dataset_records') != dataset_records:
            p.error('dataset episode ordering/identity changed; use a new cache')
    else:
        if root.exists() and any(root.iterdir()):
            p.error('refusing to write an unrecognized nonempty cache directory')
        root.mkdir(parents=True, exist_ok=True)
        manifest = dict(spec=spec, teacher=teacher, dataset_records=dataset_records,
                        episodes={}, complete=False)
        # Persist before GPU work so even an interruption in episode 1 can resume.
        save_manifest(root, manifest)
    if is_robotwin:
        from types import SimpleNamespace
        from robotwin_infer import RobotWinInference
        engine = RobotWinInference(a.config, a.checkpoint, a.stats, device=a.device,
                                   dtype=torch.bfloat16 if a.device.startswith('cuda') else torch.float32)
        policy = SimpleNamespace(wrapper=engine.model, action=engine.model.action_model,
                                 device=torch.device(a.device), dtype=engine.dtype, cfg=cfg)
    else:
        policy = Policy(cfg, a.stats, a.checkpoint, a.device)
    generator = torch.Generator(device=policy.device)
    started = time.monotonic()
    offset = 0
    for ep_id, ep in enumerate(ds.episodes):
        key, record = cache_key(ep['path']), episode_record(ds, ep)
        output = root / key
        cached = manifest['episodes'].get(key)
        if cached:
            if cached['source'] != record or not output.is_file() or cached['sha256'] != sha256(output):
                raise ValueError(f'changed/incomplete cached episode: {ep["path"]}')
            offset += ep['length']
            continue
        generator.manual_seed(a.seed + ep_id)
        partial = root / (key + '.partial')
        array = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                         shape=(ep['length'], h-e, ds.action_dim))
        array[:] = 0
        # Earlier observation j forecasts the current anchor j+E; never read a future frame.
        subset = Subset(ds, range(offset, offset + max(0, ep['length']-e)))
        loader = DataLoader(subset, batch_size=a.batch_size, num_workers=a.workers,
                            shuffle=False, collate_fn=collate_fn)
        anchor = e
        for batch in loader:
            values = generate(policy, batch, teacher['nfe'], generator)[:, e:]
            if not np.isfinite(values).all():
                raise ValueError('teacher produced nonfinite predictions')
            array[anchor:anchor+len(values)] = values
            anchor += len(values)
        array.flush()
        del array
        os.replace(partial, output)
        manifest['episodes'][key] = dict(source=record, sha256=sha256(output))
        manifest['complete'] = len(manifest['episodes']) == len(ds.episodes)
        manifest['wall_seconds'] = time.monotonic()-started
        save_manifest(root, manifest)
        offset += ep['length']
        print(f'[{ep_id+1}/{len(ds.episodes)}] cached {ep["task"]}/{Path(ep["path"]).name}', flush=True)
    print(f'Cache complete={manifest["complete"]}: {root}', flush=True)


if __name__ == '__main__':
    main()
