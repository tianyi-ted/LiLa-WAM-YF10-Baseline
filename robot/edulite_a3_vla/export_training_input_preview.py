#!/usr/bin/env python3
"""Export the exact pre-normalization image used by the training dataloader."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("episode", type=Path)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--camera", default="chest_camera")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--output-dir", type=Path, default=Path("training_input_preview"))
    args = p.parse_args()
    with h5py.File(args.episode, "r") as f:
        key = f"observation/{args.camera}/rgb"
        if key not in f:
            raise SystemExit(f"missing {key}")
        frames = f[key]
        if not 0 <= args.frame < len(frames):
            raise SystemExit(f"frame must be in [0, {len(frames)-1}]")
        encoded = np.frombuffer(frames[args.frame], dtype=np.uint8)
    # This is deliberately identical to dataloader/dataset.py::_decode.
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise SystemExit("JPEG decode failed")
    model_input = cv2.resize(decoded, (args.width, args.height), interpolation=cv2.INTER_LINEAR)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "model_input_uint8.npy", model_input)
    # Display the exact numeric model input as RGB values; no extra channel swap.
    Image.fromarray(model_input).save(args.output_dir / "model_input_before_normalize.png")
    # Also retain OpenCV's native display interpretation for side-by-side diagnosis.
    cv2.imwrite(str(args.output_dir / "opencv_display.png"), model_input)
    print(f"episode={args.episode}")
    print(f"camera={args.camera}, frame={args.frame}, source_shape={decoded.shape}")
    print(f"model_input_shape={model_input.shape}, dtype={model_input.dtype}")
    print(f"saved={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
