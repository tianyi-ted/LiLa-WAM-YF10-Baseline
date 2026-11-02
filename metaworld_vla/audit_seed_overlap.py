"""Reconstruct native collection reset vectors and compare legacy evaluation resets.

Read-only diagnostic: no policy rollout, checkpoint, or evaluation protocol is
changed. Training reset vectors are reconstructed from HDF5 seeds using the
currently installed simulator, not read directly from the old HDF5 files.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

import h5py
import numpy as np


def fingerprint(vector):
    return tuple(np.round(np.asarray(vector, dtype=np.float64), 10).tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, default=Path('data/metaworld_mt50_corner2'))
    p.add_argument('--tasks', nargs='+', default=['reach-v3', 'push-v3', 'pick-place-v3'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--test-num', type=int, default=50)
    p.add_argument('--output', type=Path, default=Path('eval_result/seed_overlap_audit/report.json'))
    args = p.parse_args()
    import gymnasium as gym
    import metaworld  # noqa: F401
    report = {'eval_constructor_seed': args.seed, 'test_num': args.test_num,
              'method': 'reconstructed training reset rand_vec; reset-only legacy evaluation sequence; no policy rollouts',
              'precision': 'reset vectors rounded to 10 decimal places', 'tasks': {}}
    for task in args.tasks:
        seeds = []
        for path in sorted((args.dataset_dir / task).glob('*.hdf5')):
            with h5py.File(path, 'r') as f:
                seeds.append(int(f.attrs['seed']))
        if not seeds:
            raise ValueError(f'No training episodes for {task}')
        train_vectors = {}
        for i, seed in enumerate(seeds):
            env = gym.make('Meta-World/MT1', env_name=task, seed=seed)
            try:
                env.reset(seed=seed)
                train_vectors.setdefault(fingerprint(env.unwrapped._last_rand_vec.copy()), []).append(seed)
            finally:
                env.close()
            if (i + 1) % 10 == 0:
                print(f'{task}: reconstructed {i+1}/{len(seeds)} collection scenes', flush=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        env = gym.make('Meta-World/MT1', env_name=task, seed=args.seed)
        episodes = []
        seen = []
        try:
            for i in range(args.test_num):
                env.reset(seed=args.seed + i)
                key = fingerprint(env.unwrapped._last_rand_vec.copy())
                seen.append(key)
                episodes.append({'episode': i, 'requested_reset_seed': args.seed+i,
                                 'rand_vec': list(key), 'matching_train_seeds': train_vectors.get(key, [])})
        finally:
            env.close()
        counts = Counter(seen)
        report['tasks'][task] = dict(training_episodes=len(seeds), training_seeds=seeds,
            unique_training_scenes=len(train_vectors), unique_eval_scenes=len(counts),
            eval_episodes_matching_training=sum(bool(x['matching_train_seeds']) for x in episodes),
            max_eval_scene_repeats=max(counts.values()), episodes=episodes)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(task, {k:v for k,v in report['tasks'][task].items() if k not in ('training_seeds','episodes')}, flush=True)
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
