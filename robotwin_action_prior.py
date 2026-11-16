"""Small adapter for the opt-in prior tools; no RGB/state/action conversion."""
import numpy as np
from torch.utils.data import Dataset


class FrameIndices:
    def __init__(self, episodes):
        self.starts = np.array([e['global_start'] for e in episodes])
        self.ends = np.array([e['global_end'] for e in episodes])

    def __getitem__(self, index):
        ep = int(np.searchsorted(self.ends, index, side='right'))
        return ep, int(index-self.starts[ep])


class RobotWinPriorAdapter(Dataset):
    def __init__(self, base):
        if not base.strict_loading:
            raise ValueError('prior tools require dataset.strict_loading=true: no random sample substitution')
        self.base = base
        self.action_dim = base.action_dim
        self.episodes = [dict(path=e['hdf5_path'], task=e['task_name'], length=e['length'])
                         for e in base.episode_metadata]
        self.indices = FrameIndices(base.episode_metadata)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self.base[index]
