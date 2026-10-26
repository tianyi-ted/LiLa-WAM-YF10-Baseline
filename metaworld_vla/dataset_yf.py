"""DataLoader dataset for external MetaWorld_YF demo_clean PNG episodes."""
try:
    from .dataset import MetaWorldTaskDataset
except ImportError:
    from dataset import MetaWorldTaskDataset


class YFMetaWorldDataset(MetaWorldTaskDataset):
    def _episode_paths(self, task):
        return sorted((self.root / task / 'demo_clean' / 'data').glob('*.hdf5'))

    def _episode_keys(self, f):
        keys = ('proprio/state', 'joint_action/vector', f'observation/{self.camera_name}/rgb')
        for key in keys:
            if key not in f:
                raise ValueError(f'{f.filename}: MetaWorld_YF dataset missing {key}')
        return keys
