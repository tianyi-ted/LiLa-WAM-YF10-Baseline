"""Meta-World expert-trajectory dataset for the existing Flow VLA model.

The collector stores JPEG RGB frames in HDF5 to keep disk usage manageable.
At the end of an episode, Cartesian delta commands are padded with zero,
while the gripper repeats the final recorded control command. Padded steps
remain training targets and are not masked out of the loss.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

try:
    from .episode_io import episode_paths, episode_keys, stored_rotation
except ImportError:
    from episode_io import episode_paths, episode_keys, stored_rotation

try:
    from .image_orientation import orient_rgb, check_task_conditions
except ImportError:
    from image_orientation import orient_rgb, check_task_conditions

logger = logging.getLogger(__name__)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _decode_rgb(buf: np.ndarray | bytes, image_size: tuple[int, int], stored_rotation: int = 0) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("invalid JPEG frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != image_size:
        rgb = cv2.resize(rgb, image_size, interpolation=cv2.INTER_LINEAR)
    return orient_rgb(rgb, stored_rotation)


def _normalise_rgb(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1))).float()


class MetaWorldTaskDataset(Dataset):
    """Flat frame-indexed dataset with 4D state and 4D actions."""

    def _episode_paths(self, task):
        return episode_paths(self.root, task)

    def _episode_keys(self, f):
        return episode_keys(f, self.camera_name)

    def __init__(
        self,
        dataset_dir: str,
        task_names: List[str],
        action_chunk_size: int = 32,
        state_dim: int = 4,
        image_size: tuple[int, int] = (224, 224),
        task_cond_dir: Optional[str] = None,
        use_future_feat: bool = True,
        future_frame_offset: Optional[int] = None,
        image_aug: bool = False,
        camera_name: str = "corner",
    ) -> None:
        if state_dim != 4:
            raise ValueError("MetaWorldTaskDataset currently expects state_dim=4")
        self.root = Path(dataset_dir)
        self.task_names = list(task_names)
        self.action_dim = 4
        self.state_dim = state_dim
        self.chunk_size = int(action_chunk_size)
        self.image_size = tuple(image_size)
        self.task_cond_dir = Path(task_cond_dir) if task_cond_dir else None
        self.use_future_feat = bool(use_future_feat)
        self.future_offset = int(future_frame_offset or self.chunk_size)
        self.image_aug = bool(image_aug)
        self.camera_name = camera_name
        self.aug_pool = [
            T.ColorJitter(brightness=0.05),
            T.ColorJitter(contrast=0.05),
            T.ColorJitter(saturation=0.05),
            T.ColorJitter(hue=0.05),
        ] if image_aug else []
        self.task_cond_cache: Dict[str, torch.Tensor] = {}
        self.episodes: List[Dict[str, Any]] = []
        self.indices: List[tuple[int, int]] = []
        self._scan()
        self._load_task_conditions()
        logger.info(
            "MetaWorld dataset: %d episodes, %d valid anchors, tasks=%s",
            len(self.episodes), len(self.indices), self.task_names,
        )

    def _scan(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        for task in self.task_names:
            task_dir = self.root / task
            if not task_dir.exists():
                raise FileNotFoundError(f"missing task directory: {task_dir}")
            paths = self._episode_paths(task)
            if not paths:
                raise ValueError(f'no episodes for task {task} in {type(self).__name__}')
            for path in paths:
                with h5py.File(path, "r") as f:
                    state_key, action_key, rgb_key = self._episode_keys(f)
                    n = int(f[action_key].shape[0])
                    if f[action_key].shape != (n, 4) or f[state_key].shape != (n, 4):
                        raise ValueError(f"{path}: expected actions [T,4] and state [T,4]")
                    if f[rgb_key].shape[0] != n:
                        raise ValueError(f"{path}: RGB/action length mismatch")
                episode_id = len(self.episodes)
                self.episodes.append({"path": str(path), "task": task, "length": n})
                # Keep every frame as an anchor. Action steps beyond the
                # episode use zero xyz and the last gripper command; a future image is
                # clipped to the final valid frame below.
                for anchor in range(n):
                    self.indices.append((episode_id, anchor))
        if not self.episodes or not self.indices:
            raise ValueError("no complete Meta-World action chunks found")

    def _load_task_conditions(self) -> None:
        if self.task_cond_dir is None:
            return
        check_task_conditions(self.task_cond_dir, self.image_size, self.camera_name)
        for task in self.task_names:
            path = self.task_cond_dir / task / "task_cond.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing task condition: {path}")
            vec = np.load(path).astype(np.float32)
            self.task_cond_cache[task] = torch.from_numpy(vec)
        dims = {tuple(v.shape) for v in self.task_cond_cache.values()}
        if len(dims) != 1:
            raise ValueError(f"inconsistent task condition dimensions: {dims}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ep_id, anchor = self.indices[index]
        ep = self.episodes[ep_id]
        with h5py.File(ep["path"], "r") as f:
            state_key, action_key, rgb_key = self._episode_keys(f)
            state = f[state_key][anchor].astype(np.float32)
            available = min(self.chunk_size, ep["length"] - anchor)
            actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            if available > 0:
                actions[:available] = f[action_key][anchor:anchor + available].astype(np.float32)
                # Gripper control is not a Cartesian displacement: zero does
                # not mean retaining the previous open/close command.
                actions[available:, 3] = actions[available - 1, 3]
            rotation = stored_rotation(f)
            rgb = _decode_rgb(f[rgb_key][anchor], self.image_size, rotation)
            future = None
            if self.use_future_feat:
                future_index = min(anchor + self.future_offset, ep["length"] - 1)
                future = _decode_rgb(
                    f[rgb_key][future_index],
                    self.image_size, rotation,
                )
        if self.aug_pool:
            pil = Image.fromarray(rgb)
            for op in random.sample(self.aug_pool, random.choice([1, 2])):
                pil = op(pil)
            rgb = np.asarray(pil)
        result: Dict[str, Any] = {
            "state": torch.from_numpy(state),
            "action_sequence": torch.from_numpy(actions),
            "pixel_values": _normalise_rgb(rgb),
            "action_mask": torch.ones(self.chunk_size, dtype=torch.bool),
            "state_mask": torch.ones(1, dtype=torch.bool),
        }
        if future is not None:
            result["future_pixel_values"] = _normalise_rgb(future)
        if self.task_cond_cache:
            result["task_cond"] = self.task_cond_cache[ep["task"]]
        return result


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in batch[0]:
        value = batch[0][key]
        if isinstance(value, torch.Tensor):
            out[key] = torch.stack([sample[key] for sample in batch])
        else:
            out[key] = [sample[key] for sample in batch]
    return out
