"""Canonical image orientation shared by collection, training and evaluation."""
import json
from pathlib import Path
import numpy as np

IMAGE_ROTATION = 180

def orient_rgb(rgb, stored_rotation=0):
    """Convert raw/previously rotated pixels to canonical 180-degree orientation."""
    stored_rotation = int(stored_rotation)
    if stored_rotation not in (0, 180):
        raise ValueError(f"unsupported stored image_rotation_degrees={stored_rotation}")
    return np.ascontiguousarray(np.rot90(rgb, k=((IMAGE_ROTATION - stored_rotation) % 360) // 90))

def check_task_conditions(directory, image_size=None, camera_name=None):
    path = Path(directory) / "meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    if meta.get("image_rotation_degrees") != IMAGE_ROTATION:
        raise ValueError(f"{path}: VTT orientation is missing or outdated. Recompute with precompute_vtt.py into a new directory for rotation=180.")
    if image_size is not None and list(meta.get("image_size", [])) != list(image_size):
        raise ValueError(f"{path}: VTT image_size={meta.get('image_size')} differs from model input {list(image_size)}. Recompute VTT at the model input size.")
    if camera_name is not None and meta.get('camera_name', camera_name) != camera_name:
        raise ValueError(f'{path}: VTT camera differs from dataset camera {camera_name}')
