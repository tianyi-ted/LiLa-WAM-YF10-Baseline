#!/usr/bin/env python3
"""Create a non-destructive 10-task RoboTwin dataset view with symlinks."""

import argparse
import json
import re
from pathlib import Path


TASKS = (
    "adjust_bottle",
    "grab_roller",
    "hanging_mug",
    "move_stapler_pad",
    "open_microwave",
    "press_stapler",
    "scan_object",
    "stack_blocks_two",
    "stamp_seal",
    "turn_switch",
)


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def link_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Refusing to replace a different link: {destination}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    destination.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Build the LiLa-WAM 10-task subset as a symlink view")
    parser.add_argument("--source", type=Path, required=True,
                        help="full RoboTwin dataset root")
    parser.add_argument("--output", type=Path, required=True,
                        help="new subset root (must differ from source)")
    parser.add_argument("--clean-limit", type=int, default=50)
    parser.add_argument("--randomized-limit", type=int, default=200)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if source == output:
        raise ValueError("--output must not be the source dataset")

    manifest = {
        "source": str(source),
        "output": str(output),
        "selection": "natural episode-number order",
        "requested_limits": {
            "demo_clean": args.clean_limit,
            "demo_randomized": args.randomized_limit,
        },
        "tasks": {},
    }
    total = 0

    for task in TASKS:
        task_counts = {}
        for split, limit in (("demo_clean", args.clean_limit),
                             ("demo_randomized", args.randomized_limit)):
            data_dir = source / task / split / "data"
            if not data_dir.is_dir():
                raise FileNotFoundError(f"Missing split: {data_dir}")
            episodes = sorted(data_dir.glob("*.hdf5"), key=natural_key)
            selected = episodes[:limit]
            if not selected:
                raise RuntimeError(f"No HDF5 episodes found in {data_dir}")

            for hdf5_path in selected:
                link_file(hdf5_path, output / task / split / "data" / hdf5_path.name)
                json_path = hdf5_path.with_suffix(".json")
                if json_path.exists():
                    link_file(json_path, output / task / split / "data" / json_path.name)

            task_counts[split] = {
                "available": len(episodes),
                "selected": len(selected),
                "first": selected[0].name,
                "last": selected[-1].name,
            }
            total += len(selected)
            print(f"{task:24s} {split:18s} {len(selected):3d}/{len(episodes):3d}")
        manifest["tasks"][task] = task_counts

    manifest["total_selected_hdf5"] = total
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "subset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nCreated/reused {total} HDF5 links under {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
