"""Compute 4D Meta-World state/action min-max statistics."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from episode_io import episode_paths, episode_keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    states, actions = [], []
    paths = [p for task in sorted(Path(args.dataset_dir).iterdir()) if task.is_dir()
             for p in episode_paths(args.dataset_dir, task.name)]
    for path in paths:
        with h5py.File(path, "r") as f:
            if 'proprio/state' in f:
                state_key, action_key = 'proprio/state', 'joint_action/vector'
            else:
                state_key, action_key = 'observations/state', 'actions'
            states.append(f[state_key][:])
            actions.append(f[action_key][:])
    if not states:
        raise RuntimeError("no Meta-World HDF5 files found")
    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    out = {
        "robotwin2": {
            "action": {"min": action.min(0).tolist(), "max": action.max(0).tolist(), "dim": 4},
            "state": {"min": state.min(0).tolist(), "max": state.max(0).tolist(), "dim": 4},
        },
        "dataset": Path(args.dataset_dir).name,
        "state_dim": 4,
        "action_dim": 4,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {args.output}: frames={len(state)}")


if __name__ == "__main__":
    main()
