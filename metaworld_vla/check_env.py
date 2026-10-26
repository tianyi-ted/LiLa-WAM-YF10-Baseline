"""Check Meta-World installation, spaces, and RGB rendering."""
import argparse
import os
from camera_config import load_profile, apply_profile, render_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="reach-v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-profile")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    profile = load_profile(args.camera_profile)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym
    import metaworld
    env = gym.make(
        "Meta-World/MT1",
        env_name=args.task,
        render_mode="rgb_array",
        camera_name=profile["slot"] if profile else "corner2",
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    obs, info = env.reset(seed=args.seed)
    apply_profile(env, profile)
    rgb = render_rgb(env, profile)
    try:
        from importlib.metadata import version
        mw_version = version("metaworld")
    except Exception:
        mw_version = "unknown"
    print("metaworld:", mw_version)
    print("observation shape:", obs.shape, obs.dtype)
    print("action shape:", env.action_space.shape, env.action_space.dtype)
    print("action bounds:", env.action_space.low, env.action_space.high)
    print("rgb shape:", rgb.shape, rgb.dtype)
    print("reset info keys:", sorted(info.keys()))
    env.close()


if __name__ == "__main__":
    main()
