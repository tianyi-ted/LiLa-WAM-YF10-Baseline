#!/usr/bin/env python3
"""Offline Tkinter viewer for a Meta-World expert episode.

The viewer is read-only: it opens one HDF5 file, decodes the stored JPEG
frames, and never creates a simulator or sends actions to an environment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import h5py
import numpy as np
from PIL import Image, ImageTk


from image_orientation import orient_rgb
from episode_io import episode_keys, stored_rotation

def decode_rgb(payload: bytes, stored_rotation: int = 0) -> Image.Image:
    """Decode the RGB JPEG format written by collect_expert.py."""
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("cannot decode JPEG frame")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(orient_rgb(image, stored_rotation))


class Viewer:
    def __init__(self, root: tk.Tk, path: Path, camera: str = 'corner'):
        self.root = root
        self.path = path
        self.file = h5py.File(path, "r")
        self.camera = camera
        state_key, action_key, self.rgb_key = episode_keys(self.file, camera)
        self.states = np.asarray(self.file[state_key][:], dtype=np.float32)
        self.actions = np.asarray(self.file[action_key][:], dtype=np.float32)
        self.rewards = np.asarray(self.file['rewards' if 'rewards' in self.file else 'reward'][:], dtype=np.float32)
        # YF provides an episode-level success attribute, not per-step success.
        self.step_success = 'success' in self.file
        self.success = (np.asarray(self.file['success'][:], dtype=bool) if self.step_success
                        else np.full(len(self.actions), bool(self.file.attrs.get('success', False))))
        self.length = len(self.actions)
        if not (
            len(self.states) == len(self.rewards) == len(self.success)
            == len(self.file[self.rgb_key])
        ):
            raise ValueError(f"inconsistent lengths in {path}")
        self.frame = 0
        self.playing = False
        self.photo = None
        self.fps = 20.0

        task = self.file.attrs.get("task_name", path.parent.name)
        self.root.title(f"Meta-World dataset viewer — {task} — {path.name}")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        top = ttk.Frame(root)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text=str(path)).pack(side="left", expand=True, anchor="w")
        ttk.Button(top, text="Open episode", command=self.open_file).pack(side="right")

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=8)
        self.play_button = ttk.Button(controls, text="Play", command=self.toggle)
        self.play_button.pack(side="left")
        ttk.Button(controls, text="|<", command=lambda: self.set_frame(0)).pack(side="left")
        ttk.Button(controls, text="<<", command=lambda: self.set_frame(self.frame - 30)).pack(side="left")
        ttk.Button(controls, text="<", command=lambda: self.set_frame(self.frame - 1)).pack(side="left")
        ttk.Button(controls, text=">", command=lambda: self.set_frame(self.frame + 1)).pack(side="left")
        ttk.Button(controls, text=">>", command=lambda: self.set_frame(self.frame + 30)).pack(side="left")
        ttk.Button(controls, text="|>", command=lambda: self.set_frame(self.length - 1)).pack(side="left")
        self.scale = tk.Scale(
            controls, from_=0, to=max(0, self.length - 1), orient="horizontal",
            showvalue=False, command=lambda value: self.set_frame(int(float(value))),
        )
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.frame_info = ttk.Label(controls, width=27)
        self.frame_info.pack(side="right")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, padx=8, pady=6)
        self.image_label = ttk.Label(body, text="No image")
        self.image_label.pack(pady=4)
        self.canvas = tk.Canvas(body, width=900, height=300, background="white", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.status = ttk.Label(root, justify="left")
        self.status.pack(fill="x", padx=8, pady=4)
        self.render()

    def open_file(self):
        selected = filedialog.askopenfilename(
            filetypes=[("HDF5 episode", "*.hdf5"), ("All files", "*")]
        )
        if selected:
            self.close_file()
            self.root.destroy()
            new_root = tk.Tk()
            type(self)(new_root, Path(selected).resolve(), self.camera)
            new_root.mainloop()

    def close_file(self):
        try:
            self.file.close()
        except Exception:
            pass

    def close(self):
        self.close_file()
        self.root.destroy()

    def toggle(self):
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self.tick()

    def tick(self):
        if not self.playing:
            return
        if self.frame >= self.length - 1:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.set_frame(self.frame + 1)
        self.root.after(max(1, int(1000 / self.fps)), self.tick)

    def set_frame(self, index: int):
        self.frame = max(0, min(self.length - 1, int(index)))
        self.scale.set(self.frame)
        self.render()

    def render(self):
        i = self.frame
        action = self.actions[i]
        state = self.states[i]
        self.frame_info.configure(text=f"frame {i + 1}/{self.length}  time={i / self.fps:.2f}s")
        self.status.configure(
            text=(
                f"task={self.file.attrs.get('task_name', self.path.parent.name)}  "
                f"seed={self.file.attrs.get('seed', '?')}  "
                f"camera={self.camera} reward={self.rewards[i]:.4f}  "
                f"{'step_success' if self.step_success else 'episode_success'}={bool(self.success[i])}\n"
                f"state [xyz, gripper] = {np.array2string(state, precision=4)}\n"
                f"action [dx, dy, dz, gripper] = {np.array2string(action, precision=4)}"
            )
        )
        payload = self.file[self.rgb_key][i]
        image = decode_rgb(bytes(payload), stored_rotation(self.file))
        image.thumbnail((900, 430))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo, text="")
        self.draw_plot()

    def draw_plot(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(500, canvas.winfo_width()), max(220, canvas.winfo_height())
        values = self.actions
        lo, hi = values.min(axis=0), values.max(axis=0)
        margin = np.maximum(0.05, (hi - lo) * 0.08)
        lo, hi = lo - margin, hi + margin
        x0, y0, x1, y1 = 48, 15, width - 20, height - 28
        canvas.create_rectangle(x0, y0, x1, y1, outline="#888")
        colors = ("#d62728", "#2ca02c", "#1f77b4", "#111111")
        labels = ("dx", "dy", "dz", "grip")
        for dim in range(4):
            points = []
            for i, value in enumerate(values[:, dim]):
                x = x0 + (x1 - x0) * i / max(1, self.length - 1)
                y = y1 - (y1 - y0) * (value - lo[dim]) / max(1e-8, hi[dim] - lo[dim])
                points.extend((x, y))
            if len(points) >= 4:
                canvas.create_line(*points, fill=colors[dim], width=2)
            canvas.create_text(8, 28 + dim * 18, text=labels[dim], fill=colors[dim], anchor="w")
        x = x0 + (x1 - x0) * self.frame / max(1, self.length - 1)
        canvas.create_line(x, y0, x, y1, fill="#000", dash=(4, 3))
        canvas.create_text(x0, height - 8, text=f"min {lo.round(3).tolist()}", anchor="w", fill="#555")
        canvas.create_text(x1, height - 8, text=f"max {hi.round(3).tolist()}", anchor="e", fill="#555")


def main():
    parser = argparse.ArgumentParser(description="Offline viewer for Meta-World HDF5 episodes")
    parser.add_argument("episode", nargs="?", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/metaworld_mt10"))
    parser.add_argument("--camera", default='corner', help='YF camera: corner or behind')
    args = parser.parse_args()
    path = args.episode
    if path is None:
        episodes = sorted(args.dataset_root.glob("**/*.hdf5"))
        if not episodes:
            raise SystemExit(f"no HDF5 episodes under {args.dataset_root}")
        path = episodes[0]
        print(f"Opening first episode: {path}")
    root = tk.Tk()
    Viewer(root, path.resolve(), args.camera)
    root.mainloop()


if __name__ == "__main__":
    main()
