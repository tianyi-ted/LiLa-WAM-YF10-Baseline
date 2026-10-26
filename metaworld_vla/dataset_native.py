"""DataLoader dataset for this project's self-collected Meta-World episodes."""
try:
    from .dataset import MetaWorldTaskDataset
except ImportError:
    from dataset import MetaWorldTaskDataset


class NativeMetaWorldDataset(MetaWorldTaskDataset):
    def _episode_paths(self, task):
        return sorted((self.root / task).glob('*.hdf5'))

    def _episode_keys(self, f):
        keys = ('observations/state', 'actions', 'observations/rgb_jpeg')
        for key in keys:
            if key not in f:
                raise ValueError(f'{f.filename}: native dataset missing {key}')
        camera = f.attrs.get('camera_name')
        if isinstance(camera, bytes):
            camera = camera.decode()
        if camera is not None and camera != self.camera_name:
            raise ValueError(f'{f.filename}: camera={camera}, configured={self.camera_name}')
        return keys
