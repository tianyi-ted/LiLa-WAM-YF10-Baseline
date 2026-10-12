#!/usr/bin/env python3
"""Offline Tkinter viewer for EDULITE-A3 RoboTwin-compatible episodes.

This tool never opens CAN and never enables a motor.  It is intended to find
bad demonstrations (jumps, frozen joints, gripper inversions and camera/action
timestamp mismatches) before training or hardware replay.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import h5py
import cv2
import numpy as np
from PIL import Image, ImageTk


def decode(payload: bytes) -> Image.Image:
    # Match dataloader/dataset.py exactly: cv2.imdecode with no BGR2RGB swap.
    # The resulting numeric array is what the training model receives before
    # ImageNet normalization; PIL is used only to display those same numbers.
    arr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("cannot decode JPEG frame")
    return Image.fromarray(arr)


class Viewer:
    def __init__(
        self, root: tk.Tk, path: Path, image_name: str = "chest_camera",
        trajectory_source: str = "action",
    ):
        self.root, self.path = root, path
        self.f = h5py.File(path, "r")
        self.action = np.asarray(self.f["joint_action/vector"][:], dtype=np.float64)
        self.feedback = (
            np.asarray(self.f["teleop/slave_joint_feedback"][:], dtype=np.float64)
            if "teleop/slave_joint_feedback" in self.f else None
        )
        if trajectory_source not in {"action", "feedback"}:
            raise ValueError("trajectory_source must be action or feedback")
        if trajectory_source == "feedback" and self.feedback is None:
            raise ValueError("episode does not contain teleop/slave_joint_feedback")
        self.trajectory_source = trajectory_source
        self.t = len(self.action)
        self.image_name = image_name if f"observation/{image_name}/rgb" in self.f else None
        self.frame = 0
        self.playing = False
        self.photo = None
        self.fps = float(self.f.attrs.get("sample_rate_hz", 30.0))
        root.title(f"Dataset viewer — {path.name}")
        root.protocol("WM_DELETE_WINDOW", self.close)

        top = ttk.Frame(root); top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text=str(path)).pack(side="left", expand=True, anchor="w")
        ttk.Button(top, text="Open episode", command=self.open_file).pack(side="right")
        controls = ttk.Frame(root); controls.pack(fill="x", padx=8)
        self.play_btn = ttk.Button(controls, text="Play", command=self.toggle); self.play_btn.pack(side="left")
        ttk.Button(controls, text="<<", command=lambda: self.set_frame(self.frame - 30)).pack(side="left")
        ttk.Button(controls, text=">>", command=lambda: self.set_frame(self.frame + 30)).pack(side="left")
        self.scale = tk.Scale(controls, from_=0, to=max(0, self.t - 1), orient="horizontal", showvalue=False,
                              command=lambda x: self.set_frame(int(float(x))))
        self.scale.pack(side="left", fill="x", expand=True, padx=8)
        self.info = ttk.Label(controls, width=48); self.info.pack(side="right")

        body = ttk.Frame(root); body.pack(fill="both", expand=True, padx=8, pady=6)
        self.image_label = ttk.Label(body, text="No camera image")
        self.image_label.pack(side="top", pady=4)
        self.canvas = tk.Canvas(body, width=900, height=360, background="white", highlightthickness=1)
        self.canvas.pack(fill="both", expand=True)
        self.status = ttk.Label(root, text="")
        self.status.pack(fill="x", padx=8, pady=4)
        self.render()

    def open_file(self):
        p = filedialog.askopenfilename(filetypes=[("HDF5 episode", "*.hdf5"), ("All files", "*")])
        if p:
            self.close_file(); self.__init__(
                self.root, Path(p), self.image_name or "chest_camera",
                self.trajectory_source,
            )

    def close_file(self):
        try: self.f.close()
        except Exception: pass

    def close(self):
        self.close_file(); self.root.destroy()

    def toggle(self):
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing: self.tick()

    def tick(self):
        if not self.playing: return
        if self.frame >= self.t - 1:
            self.playing = False; self.play_btn.configure(text="Play"); return
        self.set_frame(self.frame + 1)
        self.root.after(max(1, int(1000 / self.fps)), self.tick)

    def set_frame(self, i: int):
        self.frame = max(0, min(self.t - 1, int(i)))
        self.scale.set(self.frame); self.render()

    def render(self):
        a = self.action[self.frame]
        self.info.configure(text=f"frame {self.frame+1}/{self.t}  time={self.frame/self.fps:.2f}s")
        selected = a if self.trajectory_source == "action" else self.feedback[self.frame]
        label = "action target" if self.trajectory_source == "action" else "slave feedback"
        self.status.configure(text=f"{label} [rad, ..., gripper_norm] = " + np.array2string(selected, precision=4))
        if self.image_name:
            payload = self.f[f"observation/{self.image_name}/rgb"][self.frame]
            image = decode(bytes(payload)); image.thumbnail((900, 430))
            self.photo = ImageTk.PhotoImage(image); self.image_label.configure(image=self.photo, text="")
        self.draw_plot()

    def draw_plot(self):
        c = self.canvas; c.delete("all")
        w, h = max(500, c.winfo_width()), max(250, c.winfo_height())
        vals = (self.action if self.trajectory_source == "action" else self.feedback)[:, :7]
        lo, hi = vals.min(0), vals.max(0)
        lo -= np.maximum(0.05, (hi - lo) * .08); hi += np.maximum(0.05, (hi - lo) * .08)
        colors = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#111111")
        pad = (45, 20, 20, 25); x0, y0, x1, y1 = pad[0], 15, w-pad[2], h-pad[3]
        c.create_rectangle(x0, y0, x1, y1, outline="#888")
        for j in range(7):
            pts=[]
            for i, v in enumerate(vals[:, j]):
                x=x0+(x1-x0)*i/max(1,self.t-1); y=y1-(y1-y0)*(v-lo[j])/(hi[j]-lo[j]); pts += [x,y]
            if len(pts)>=4: c.create_line(*pts, fill=colors[j], width=2)
            c.create_text(8, 28+j*18, text=f"L{j+1}", fill=colors[j], anchor="w")
        x=x0+(x1-x0)*self.frame/max(1,self.t-1); c.create_line(x,y0,x,y1,fill="#000",dash=(4,3))
        c.create_text(x0, h-8, text=f"min {lo.round(3).tolist()}", anchor="w", fill="#555")
        c.create_text(x1, h-8, text=f"max {hi.round(3).tolist()}", anchor="e", fill="#555")
        c.create_text(x1, 8, text=f"source: {self.trajectory_source}", anchor="e", fill="#333")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("episode", nargs="?", type=Path)
    ap.add_argument("--dataset-root", type=Path, default=Path("robot/edulite_a3_collect/collected_data_single_static_clean"))
    ap.add_argument("--camera", default="chest_camera")
    ap.add_argument(
        "--trajectory-source", choices=["action", "feedback"], default="action",
        help="plot joint_action/vector or teleop/slave_joint_feedback",
    )
    args=ap.parse_args()
    path=args.episode
    if path is None:
        files=sorted(args.dataset_root.glob("**/data/*.hdf5"))
        if not files: raise SystemExit(f"no HDF5 episodes under {args.dataset_root}")
        path=files[0]; print(f"Opening first episode: {path}")
    root=tk.Tk(); Viewer(root, path.resolve(), args.camera, args.trajectory_source); root.mainloop()

if __name__ == "__main__": main()
