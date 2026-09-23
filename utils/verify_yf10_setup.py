#!/usr/bin/env python3
"""Validate all files required by the local YF10 LiLa-WAM baseline."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoConfig


def require(path: Path, kind: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/yf10_base.yaml")
    parser.add_argument("--stats", default="artifacts/yf10/norm_stats.json")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    dataset = require(Path(cfg.dataset.dataset_dir), "dataset")
    vtt_root = require(Path(cfg.dataset.task_cond_dir), "VTT directory")
    dino_root = require(Path(cfg.model.vision_encoder.checkpoint_path), "DINO checkpoint")
    stats_path = require(Path(args.stats), "normalization statistics")
    require(dino_root / "config.json", "DINO config")
    require(dino_root / "model.safetensors", "DINO weights")
    dino_cfg = AutoConfig.from_pretrained(dino_root, local_files_only=True)
    hidden_size = getattr(dino_cfg, "hidden_size", None)
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError(f"Invalid DINO hidden size: {hidden_size}")

    task_dirs = sorted(p for p in dataset.iterdir() if p.is_dir())
    if len(task_dirs) != 10:
        raise ValueError(f"Expected 10 task directories, found {len(task_dirs)}")

    total_episodes = 0
    total_frames = 0
    for task_dir in task_dirs:
        task_episodes = 0
        for split in ("demo_clean", "demo_randomized"):
            for path in (task_dir / split / "data").glob("*.hdf5"):
                with h5py.File(path, "r") as f:
                    total_frames += int(f["joint_action/vector"].shape[0])
                task_episodes += 1
        total_episodes += task_episodes

        vtt_path = require(vtt_root / task_dir.name / "task_cond.npy",
                           f"VTT for {task_dir.name}")
        vector = np.load(vtt_path)
        if vector.shape != (hidden_size,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid VTT {vtt_path}: expected ({hidden_size},), shape={vector.shape}")

    stats = json.loads(stats_path.read_text())["robotwin2"]
    if stats["action"]["dim"] != cfg.common.action_dim:
        raise ValueError("Action-stat dimension does not match config")
    if stats["state"]["dim"] != cfg.common.state_dim:
        raise ValueError("State-stat dimension does not match config")

    print("YF10 setup: OK")
    print(f"  dataset : {dataset}")
    print(f"  episodes: {total_episodes}")
    print(f"  frames  : {total_frames}")
    print(f"  VTT     : {vtt_root} (10 x {hidden_size})")
    print(f"  DINO    : {dino_root} ({dino_cfg.model_type})")
    print(f"  stats   : {stats_path} (action=14, state=16)")


if __name__ == "__main__":
    main()
