"""Open a Meta-World scene in MuJoCo's interactive free-camera viewer."""
import argparse
import time


def launch_free_viewer(env):
    import mujoco
    import mujoco.viewer
    base = env.unwrapped
    viewer = mujoco.viewer.launch_passive(base.model, base.data)
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = [0, 0.6, 0.15]
        viewer.cam.distance = 1.8
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25
    viewer.sync()
    return viewer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='reach-v3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expert', action='store_true', help='run scripted expert; otherwise inspect frozen reset scene')
    args = parser.parse_args()
    import gymnasium as gym
    import metaworld  # Registers Gymnasium environments.
    import numpy as np
    env = gym.make('Meta-World/MT1', env_name=args.task, seed=args.seed)
    try:
        obs, _ = env.reset(seed=args.seed)
        policy = None
        if args.expert:
            from collect_expert import policy_class
            policy = policy_class(args.task)()
        with launch_free_viewer(env) as viewer:
            while viewer.is_running():
                start = time.monotonic()
                if policy is not None:
                    obs, _, terminated, truncated, _ = env.step(np.clip(policy.get_action(obs), -1, 1))
                    if terminated or truncated:
                        obs, _ = env.reset()
                viewer.sync()
                time.sleep(max(0, 0.05 - (time.monotonic() - start)))
    finally:
        env.close()


if __name__ == '__main__':
    main()
