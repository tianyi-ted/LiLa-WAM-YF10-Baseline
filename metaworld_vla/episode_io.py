"""Read native collector episodes and external MetaWorld_YF without conversion."""
from pathlib import Path


def episode_paths(root, task):
    directory = Path(root) / task
    return sorted(directory.glob('*.hdf5')) + sorted(directory.glob('demo_clean/data/*.hdf5'))


def episode_keys(f, camera='corner'):
    if 'observations/state' in f:
        return 'observations/state', 'actions', 'observations/rgb_jpeg'
    keys = ('proprio/state', 'joint_action/vector', f'observation/{camera}/rgb')
    for key in keys:
        if key not in f:
            raise ValueError(f'{f.filename}: missing {key}')
    return keys


def stored_rotation(f):
    # YF PNGs are already upright. 180 here means canonical/already processed
    # in the legacy orientation API; it does not rotate the YF pixels.
    default = 180 if 'proprio/state' in f else 0
    return int(f.attrs.get('image_rotation_degrees', default))
