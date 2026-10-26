"""Collect Meta-World scripted expert demonstrations into compact HDF5 files."""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import re
from pathlib import Path

import cv2
import h5py
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from image_orientation import IMAGE_ROTATION
from camera_config import load_profile, apply_profile, render_rgb

TASKS = [
    "reach-v3", "push-v3", "pick-place-v3", "door-open-v3", "drawer-open-v3",
    "drawer-close-v3", "button-press-topdown-v3", "peg-insert-side-v3",
    "window-open-v3", "box-close-v3",
]


def policy_class(task: str):
    aliases = {
        "peg-insert-side-v3": ("sawyer_peg_insertion_side_v3_policy", "SawyerPegInsertionSideV3Policy"),
    }
    if task in aliases:
        module, cls_name = aliases[task]
        try:
            return getattr(importlib.import_module(f"metaworld.policies.{module}"), cls_name)
        except (ImportError, AttributeError):
            pass
    stem = task.removesuffix("-v3")
    words = re.split(r"[-_]", stem)
    cls_name = "Sawyer" + "".join(w.capitalize() for w in words) + "V3Policy"
    module = "sawyer_" + "_".join(words) + "_v3_policy"
    try:
        mod = importlib.import_module(f"metaworld.policies.{module}")
        return getattr(mod, cls_name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"No scripted policy mapping for {task}: {module}.{cls_name}") from exc


def collect_one(task, episode_id, args):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym

    profile = load_profile(getattr(args, "camera_profile", None))
    policy = policy_class(task)()
    # RandomTaskSelectWrapper samples its task before reset(seed=...) reaches
    # the base environment. Seed construction per attempt too, otherwise each
    # newly created environment selects the same task and duplicates demos.
    episode_seed = args.seed + episode_id
    env = gym.make(
        "Meta-World/MT1", env_name=task, render_mode="rgb_array",
        camera_name=profile["slot"] if profile else args.camera, width=args.width, height=args.height,
        seed=episode_seed,
    )
    obs, _ = env.reset(seed=episode_seed)
    apply_profile(env, profile)
    states, actions, rewards, successes, frames = [], [], [], [], []
    success = False
    for _ in range(args.horizon):
        rgb = render_rgb(env, profile)
        action = np.asarray(policy.get_action(obs), dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        ok = bool(info.get("success", 0))
        states.append(np.asarray(obs[:4], dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        successes.append(ok)
        ok_img, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok_img:
            raise RuntimeError("JPEG encoding failed")
        frames.append(encoded.tobytes())
        obs = next_obs
        success |= ok
        if ok and args.stop_on_success:
            break
        if terminated or truncated:
            break
    env.close()
    if args.success_only and not success:
        return False, len(states)

    out_dir = Path(args.output_dir) / task
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"episode_{episode_id:06d}.hdf5"
    with h5py.File(path, "w") as f:
        f.create_dataset("observations/state", data=np.asarray(states, dtype=np.float32))
        f.create_dataset("actions", data=np.asarray(actions, dtype=np.float32))
        f.create_dataset("rewards", data=np.asarray(rewards, dtype=np.float32))
        f.create_dataset("success", data=np.asarray(successes, dtype=np.bool_))
        dtype = h5py.vlen_dtype(np.dtype("uint8"))
        ds = f.create_dataset("observations/rgb_jpeg", (len(frames),), dtype=dtype)
        for i, buf in enumerate(frames):
            ds[i] = np.frombuffer(buf, dtype=np.uint8)
        f.attrs["task_name"] = task
        f.attrs["seed"] = args.seed + episode_id
        f.attrs["camera_name"] = profile["slot"] if profile else args.camera
        if profile:
            import json
            f.attrs["camera_profile_json"] = json.dumps(profile)
        f.attrs["image_rotation_degrees"] = IMAGE_ROTATION
        f.attrs["render_width"] = args.width
        f.attrs["render_height"] = args.height
        f.attrs["expert_success"] = success
    return True, len(states)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', help='read task list and camera/render settings from a native training YAML')
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--episodes-per-task", type=int, default=100)
    p.add_argument("--num-tasks", type=int, default=50)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--camera-profile", help="project fixed camera JSON; overrides --camera slot")
    p.add_argument("--camera", default="corner2")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--success-only", action="store_true")
    p.add_argument("--stop-on-success", action="store_true")
    p.add_argument(
        "--target-successes", type=int, default=None,
        help="save this many successful episodes per task; retries failed attempts",
    )
    p.add_argument(
        "--max-attempts", type=int, default=0,
        help="maximum attempts per task when --target-successes is set (0 = target*2)",
    )
    args = p.parse_args()
    if args.config:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        if cfg.dataset.get('format', 'native') != 'native':
            p.error('--config must describe a native collection dataset, not an external dataset')
        args.tasks = args.tasks or list(cfg.dataset.task_names)
        args.camera = str(cfg.dataset.camera_names[0])
        args.camera_profile = cfg.dataset.get('camera_profile', None)
        args.width, args.height = map(int, cfg.dataset.render_size)
    args.tasks = args.tasks or TASKS
    if args.target_successes is not None:
        args.success_only = True
    if args.horizon <= 0 or args.max_attempts < 0 or (args.target_successes is not None and args.target_successes <= 0):
        p.error('horizon and target-successes must be positive; max-attempts must be nonnegative')
    # Fail before writing any episodes if a task or scripted expert is invalid.
    # In particular Meta-World v3 provides box-close-v3, not box-open-v3.
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS
    for task in args.tasks:
        if task not in ALL_V3_ENVIRONMENTS:
            hint = " Did you mean box-close-v3?" if task == "box-open-v3" else ""
            p.error(f"Unknown Meta-World task: {task}.{hint}")
        policy_class(task)
    for task in args.tasks:
        saved = 0
        if args.target_successes is None:
            attempts = args.episodes_per_task
        else:
            if args.target_successes <= 0:
                raise ValueError("--target-successes must be positive")
            attempts = args.max_attempts or args.target_successes * 2
        for episode_id in range(attempts):
            ok, frames = collect_one(task, episode_id, args)
            saved += int(ok)
            logger.info("%s episode=%d saved=%s frames=%d", task, episode_id, ok, frames)
            if args.target_successes is not None and saved >= args.target_successes:
                break
        target = args.target_successes if args.target_successes is not None else args.episodes_per_task
        logger.info("%s: saved %d/%d successful episodes", task, saved, target)
        if args.target_successes is not None and saved < args.target_successes:
            logger.warning(
                "%s: reached attempt limit with only %d/%d successful episodes",
                task, saved, args.target_successes,
            )


if __name__ == "__main__":
    main()
