"""Choose a project workspace camera interactively; S saves, P previews, Space pauses."""
import argparse
import json
import threading
import time
from pathlib import Path
from camera_config import load_profile, apply_profile, profile_from_free_camera, render_rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', default='pick-place-v3')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('metaworld_vla/configs/workspace_camera.json'))
    p.add_argument('--profile', help='load a saved profile for fixed-camera preview')
    p.add_argument('--slot', default='corner2')
    p.add_argument('--fixed', action='store_true', help='start with loaded fixed camera instead of editable free camera')
    p.add_argument('--fovy', type=float, help='vertical field of view in degrees; e.g. 45 or 60')
    p.add_argument('--expert', action='store_true')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    args = p.parse_args()
    import gymnasium as gym
    import metaworld
    import mujoco
    import mujoco.viewer
    import numpy as np
    from collect_expert import policy_class
    profile = load_profile(args.profile)
    slot = profile['slot'] if profile else args.slot
    env = gym.make('Meta-World/MT1', env_name=args.task, seed=args.seed,
                   render_mode='rgb_array', camera_name=slot, width=args.width, height=args.height)
    save_event, preview_event, pause_event = (threading.Event() for _ in range(3))
    def key_callback(key):
        if key == ord('S'): save_event.set()
        if key == ord('P'): preview_event.set()
        if key == 32: pause_event.set()
    try:
        obs, _ = env.reset(seed=args.seed)
        apply_profile(env, profile)
        fovy = args.fovy if args.fovy is not None else (profile['fovy'] if profile else 45)
        if not 1 < fovy < 179:
            raise ValueError('fovy must be between 1 and 179')
        env.unwrapped.model.vis.global_.fovy = fovy
        expert = policy_class(args.task)() if args.expert else None
        paused = False
        with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data,
                                         key_callback=key_callback) as viewer:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = [0, 0.65, 0.10]
                viewer.cam.distance = 1.4
                viewer.cam.azimuth = 230
                viewer.cam.elevation = -40
                if profile and 'free_camera' in profile:
                    for name, value in profile['free_camera'].items():
                        if name == 'lookat': viewer.cam.lookat[:] = value
                        else: setattr(viewer.cam, name, value)
                if args.fixed and profile:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = env.unwrapped.model.camera(slot).id
            print('Drag to rotate/pan, wheel to zoom. S: save free view; P: fixed RGB PNG; Space: pause expert.', flush=True)
            while viewer.is_running():
                if pause_event.is_set():
                    paused = not paused; pause_event.clear()
                if save_event.is_set():
                    save_event.clear()
                    with viewer.lock():
                        if viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                            print('Choose Free camera in viewer before saving.', flush=True)
                        else:
                            profile = profile_from_free_camera(env, viewer.cam, slot)
                            apply_profile(env, profile)
                            args.output.parent.mkdir(parents=True, exist_ok=True)
                            args.output.write_text(json.dumps(profile, indent=2) + '\n')
                            print(f'Saved {args.output}', flush=True)
                            preview_event.set()
                if preview_event.is_set():
                    preview_event.clear()
                    from PIL import Image
                    dest = args.output.with_suffix('.png')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(render_rgb(env, profile)).save(dest)
                    print(f'Fixed camera RGB preview: {dest}', flush=True)
                if expert and not paused:
                    obs, _, term, trunc, _ = env.step(np.clip(expert.get_action(obs), -1, 1))
                    if term or trunc:
                        obs, _ = env.reset(); apply_profile(env, profile)
                viewer.sync()
                time.sleep(0.05)
    finally:
        env.close()


if __name__ == '__main__': main()
