#!/usr/bin/env python3
"""Read-only MetaWorld_YF PNG viewer with a corner/behind camera selector."""
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import h5py

from inspect_dataset_tk import Viewer
from episode_io import episode_keys


class YFViewer(Viewer):
    def __init__(self, root, path, camera='corner'):
        with h5py.File(path, 'r') as f:
            if 'proprio/state' not in f:
                raise ValueError(f'{path}: expected external MetaWorld_YF format')
            episode_keys(f, camera)
            cameras = [name for name in f['observation'] if f'observation/{name}/rgb' in f]
            episode_seed = f.attrs.get('episode_seed', '?')
        super().__init__(root, path, camera)
        root.title(f'MetaWorld_YF — {path.name} — seed {episode_seed}')
        controls = ttk.Frame(root)
        controls.pack(fill='x', padx=8, pady=4)
        ttk.Label(controls, text='Stored camera:').pack(side='left')
        self.camera_choice = tk.StringVar(value=camera)
        selector = ttk.Combobox(controls, textvariable=self.camera_choice,
                                values=cameras, state='readonly', width=16)
        selector.pack(side='left', padx=6)
        selector.bind('<<ComboboxSelected>>', self.select_camera)
        ttk.Label(controls, text='Original PNG orientation; success is episode-level.').pack(side='left')

    def select_camera(self, _event=None):
        self.camera = self.camera_choice.get()
        _, _, self.rgb_key = episode_keys(self.file, self.camera)
        self.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', nargs='?', type=Path)
    parser.add_argument('--dataset-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/metaworld_yf_mt50')
    parser.add_argument('--task', help='optional task filter, e.g. pick-place-v3')
    parser.add_argument('--camera', choices=['corner', 'behind'], default='corner')
    args = parser.parse_args()
    path = args.episode
    if path is None:
        task_pattern = args.task if args.task else '*'
        episodes = sorted(args.dataset_root.glob(f'{task_pattern}/demo_clean/data/*.hdf5'))
        if not episodes:
            parser.error(f'no YF episodes under {args.dataset_root}, task={task_pattern}')
        path = episodes[0]
    print(f'Opening {path.resolve()}, camera={args.camera}')
    root = tk.Tk()
    YFViewer(root, path.resolve(), args.camera)
    root.mainloop()


if __name__ == '__main__':
    main()
