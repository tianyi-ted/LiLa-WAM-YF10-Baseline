"""Compute one frozen DINO task condition per Meta-World task."""
from __future__ import annotations

import argparse, json, logging
from pathlib import Path
import cv2, h5py, numpy as np, torch
from transformers import AutoConfig, AutoModel

from image_orientation import IMAGE_ROTATION
from dataset import _decode_rgb, _normalise_rgb
from episode_io import episode_paths, episode_keys, stored_rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="use dataset, tasks, camera, size and DINO path from a training YAML")
    p.add_argument("--dataset-dir")
    p.add_argument("--output-dir")
    p.add_argument("--dino-path")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--camera", default="corner2", help="YF camera key; --config reads camera_names[0]")
    p.add_argument("--image-size", nargs=2, type=int, default=[224, 224], help="model input width height; stored images are resized on read")
    p.add_argument("--target-layer", type=int, default=-2)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.config:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        args.dataset_dir = args.dataset_dir or str(cfg.dataset.dataset_dir)
        args.output_dir = args.output_dir or str(cfg.dataset.task_cond_dir)
        args.dino_path = args.dino_path or str(cfg.model.vision_encoder.checkpoint_path)
        args.tasks = args.tasks or list(cfg.dataset.task_names)
        args.camera = str(cfg.dataset.camera_names[0])
        args.image_size = list(cfg.dataset.image_size)
    if not all([args.dataset_dir, args.output_dir, args.dino_path, args.tasks]):
        p.error('provide --config or dataset-dir, output-dir, dino-path and tasks')
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.dino_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    for param in model.parameters(): param.requires_grad_(False)
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)
    all_meta = {"dataset_dir": str(Path(args.dataset_dir).resolve()), "dino_path": str(Path(args.dino_path).resolve()), "target_layer": args.target_layer, "image_size": args.image_size, "camera_name": args.camera, "image_rotation_degrees": IMAGE_ROTATION, "tasks": {}}
    for task in args.tasks:
        paths = episode_paths(args.dataset_dir, task)
        if not paths: raise FileNotFoundError(f"no episodes for {task}")
        vectors = []
        for path in paths:
            with h5py.File(path, "r") as f:
                _, _, rgb_key = episode_keys(f, args.camera)
                n = len(f[rgb_key])
                rotation = stored_rotation(f)
                first = _normalise_rgb(_decode_rgb(f[rgb_key][0], tuple(args.image_size), rotation))
                last = _normalise_rgb(_decode_rgb(f[rgb_key][n - 1], tuple(args.image_size), rotation))
            batch = torch.stack([first, last]).to(device=device, dtype=dtype)
            with torch.no_grad():
                h = model(pixel_values=batch, output_hidden_states=True, return_dict=True).hidden_states[args.target_layer]
            # CLS plus registers are removed; mean spatial change is the task vector.
            r = int(getattr(model.config, "num_register_tokens", 0))
            vec = (h[1, 1 + r:, :].float().mean(0) - h[0, 1 + r:, :].float().mean(0))
            # Use the final frame embedding itself when a difference is numerically zero.
            if torch.linalg.vector_norm(vec) < 1e-8:
                vec = h[1, 1 + r:, :].float().mean(0)
            vectors.append(vec.cpu().numpy())
        cond = np.mean(np.stack(vectors), axis=0).astype(np.float32)
        task_dir = out_root / task; task_dir.mkdir(parents=True, exist_ok=True)
        np.save(task_dir / "task_cond.npy", cond)
        all_meta["tasks"][task] = {"episodes": len(paths), "dim": int(cond.shape[0])}
        log.info("%s: %d episodes, condition dim=%d", task, len(paths), cond.shape[0])
    (out_root / "meta.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
