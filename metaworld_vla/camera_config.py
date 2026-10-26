"""Project fixed camera profiles, without changing Meta-World assets."""
import json
from pathlib import Path
import numpy as np
from image_orientation import orient_rgb, IMAGE_ROTATION


def load_profile(path):
    if not path:
        return None
    profile = json.loads(Path(path).read_text())
    for key in ('slot', 'position', 'quaternion_wxyz', 'fovy'):
        if key not in profile:
            raise ValueError(f'{path}: missing {key}')
    if not 1 < float(profile['fovy']) < 179:
        raise ValueError('fovy must be between 1 and 179 degrees')
    return profile


def apply_profile(env, profile):
    if profile is None:
        return
    import mujoco
    model = env.unwrapped.model
    camera = model.camera(profile['slot'])
    if int(model.cam_bodyid[camera.id]) != 0:
        raise ValueError('Custom workspace camera requires a world-attached slot')
    quat = np.asarray(profile['quaternion_wxyz'], dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError('Invalid camera quaternion')
    camera.pos[:] = profile['position']
    camera.quat[:] = quat / np.linalg.norm(quat)
    camera.fovy[0] = float(profile['fovy'])
    mujoco.mj_forward(model, env.unwrapped.data)


def render_rgb(env, profile=None):
    # A custom profile already defines an upright pose. Tag it as canonical
    # before the shared orientation function so it is never turned again.
    return orient_rgb(env.render(), IMAGE_ROTATION if profile is not None else 0)


def profile_from_free_camera(env, camera, slot='corner2'):
    """Get the exact mono pose from MuJoCo's camera pipeline, not Euler guesses."""
    import mujoco
    model, data = env.unwrapped.model, env.unwrapped.data
    scene = mujoco.MjvScene(model, maxgeom=10000)
    mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, camera,
                         mujoco.mjtCatBit.mjCAT_ALL, scene)
    position = np.mean([c.pos for c in scene.camera], axis=0)
    forward = np.mean([c.forward for c in scene.camera], axis=0)
    forward /= np.linalg.norm(forward)
    up = np.mean([c.up for c in scene.camera], axis=0)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    matrix = np.column_stack((right, up, -forward))
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, matrix.reshape(-1))
    return dict(name='workspace', slot=slot, position=position.tolist(),
                quaternion_wxyz=quaternion.tolist(), fovy=float(model.vis.global_.fovy),
                free_camera=dict(lookat=camera.lookat.tolist(), distance=float(camera.distance),
                                 azimuth=float(camera.azimuth), elevation=float(camera.elevation)))
