"""AdaPolicy-style unfrozen scene sampling with reproducible episode seeds.

Reuses this project's Policy and RGB pipeline; does not change legacy eval.py.
Unlike AdaPolicy's released harness, explicitly seeds the actual scene RNG.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import random

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from omegaconf import OmegaConf

from eval import Policy
from camera_config import apply_profile, load_profile, render_rgb
from reuse_options import add_reuse_arguments, validate_reuse_arguments

log = logging.getLogger(__name__)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_environment(task, seed, camera, render_size):
    import metaworld

    # Same direct MT1/train_class route as AdaPolicy (no random-task wrapper).
    mt1 = metaworld.MT1(task, seed=seed)
    env = mt1.train_classes[task](
        render_mode="rgb_array", camera_name=camera,
        width=int(render_size[0]), height=int(render_size[1]))
    pool = [instance for instance in mt1.train_tasks if instance.env_name == task]
    env.set_task(pool[np.random.RandomState(seed).randint(len(pool))])
    env.unwrapped._freeze_rand_vec = False
    return env


def reset_scene(env, seed):
    # Meta-World 3.1.1 ignores reset(seed). Its default unfrozen sampling uses
    # global NumPy; seeded_rand_vec=True instead uses env.np_random. Seed both.
    random.seed(seed)
    np.random.seed(seed)
    env.unwrapped.seed(seed)
    env.unwrapped._freeze_rand_vec = False
    obs, info = env.reset(seed=seed)
    vector = np.asarray(env.unwrapped._last_rand_vec, dtype=np.float64).copy()
    return obs, info, vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "stats", "checkpoint", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--test-num", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=500,
                        help="500 keeps our historical budget; use 200 for AdaPolicy's default")
    parser.add_argument("--seed", type=int, default=10000,
                        help="effective scene seed for episode ep is seed + ep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-file", type=Path,
                        help="optional additional logging file; console output follows eval.py")
    add_reuse_arguments(parser)
    args = parser.parse_args()
    validate_reuse_arguments(parser, args)
    if args.test_num <= 0 or args.horizon <= 0:
        parser.error("test-num and horizon must be positive")
    if args.seed < 0 or args.seed + args.test_num - 1 >= 2**32:
        parser.error("episode seeds must be in NumPy's [0, 2**32) range")
    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    log_path = args.log_file
    if log_path is not None and log_path.resolve() == output.resolve():
        parser.error("log-file must differ from output")
    handler = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(handler)
    try:
        run(args)
    finally:
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()


def run(args):
    output = Path(args.output)
    cfg = OmegaConf.load(args.config)
    if cfg.dataset.get("online_eval_verified", True) is False:
        raise ValueError("External dataset rendering has not been calibrated for online evaluation")
    if args.num_inference_steps is not None:
        cfg.common.num_inference_steps = args.num_inference_steps
    tasks = args.tasks or list(cfg.dataset.task_names)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be nonempty and unique")
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    policy_seed = args.policy_seed if args.policy_seed is not None else 0
    seed_all(policy_seed)
    policy = Policy(cfg, args.stats, args.checkpoint, device,
                    action_reuse=args.action_reuse, reuse_sigma=args.reuse_sigma,
                    reuse_strength=args.reuse_strength, policy_seed=policy_seed,
                    profile_generation=args.profile_generation)
    profile = load_profile(cfg.dataset.get("camera_profile"))
    camera = profile["slot"] if profile else str(cfg.dataset.camera_names[0])
    render_size = tuple(cfg.dataset.get("render_size", cfg.dataset.image_size))
    result = dict(
        checkpoint=str(Path(args.checkpoint).resolve()), config=str(Path(args.config).resolve()),
        stats=str(Path(args.stats).resolve()), seed=args.seed, policy_seed=policy_seed,
        horizon=args.horizon, test_num=args.test_num,
        rng_protocol="adapolicy_unfrozen_explicit_episode_seed_v1",
        scene_sampling="continuous_unfrozen", stop_on_success=True,
        metaworld_version=importlib.metadata.version("metaworld"),
        camera=camera, camera_profile=profile, render_size=list(render_size),
        image_size=list(cfg.dataset.image_size),
        action_generation=dict(action_reuse=args.action_reuse, reuse_sigma=args.reuse_sigma,
                               reuse_strength=args.reuse_strength,
                               num_inference_steps=int(cfg.common.num_inference_steps),
                               chunk_size=int(cfg.common.action_chunk_size),
                               execution_horizon=int(cfg.common.action_execution_horizon)),
        tasks={})
    for task in tasks:
        seed_all(policy_seed)
        policy.set_task(task)
        env = make_environment(task, args.seed, camera, render_size)
        episodes = []
        try:
            for ep in range(args.test_num):
                scene_seed = args.seed + ep
                obs, _, vector = reset_scene(env, scene_seed)
                apply_profile(env, profile)
                policy.reset_episode()
                # Independent episode noise: prior episode length cannot change
                # this episode's noise stream. Scene sampling never consumes it.
                noise_seed = policy_seed + ep
                policy.generator.manual_seed(noise_seed)
                ok = False
                for step in range(args.horizon):
                    rgb = render_rgb(env, profile)
                    obs, _, term, trunc, info = env.step(policy.step(obs, rgb))
                    ok = bool(info.get("success", 0))
                    if ok or term or trunc:
                        break
                episodes.append(dict(environment_seed=scene_seed, policy_seed=noise_seed,
                                     initial_rand_vec=vector.tolist(), success=int(ok), steps=step + 1))
                log.info("%s episode %d success=%s", task, ep, ok)
        finally:
            env.close()
        unique = len({tuple(np.round(ep["initial_rand_vec"], 10)) for ep in episodes})
        success = sum(ep["success"] for ep in episodes)
        result["tasks"][task] = dict(success=success, episodes=len(episodes),
                                     success_rate=success / len(episodes), unique_initializations=unique,
                                     episode_results=episodes, generation_metrics=policy.metrics_summary())
        log.info("%s success rate %.3f", task, result["tasks"][task]["success_rate"])
    result["overall_success_rate"] = float(np.mean([v["success_rate"] for v in result["tasks"].values()]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Overall success rate %.4f; saved %s", result["overall_success_rate"], output)


if __name__ == "__main__":
    main()
