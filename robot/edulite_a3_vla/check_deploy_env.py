#!/usr/bin/env python3
"""Offline environment, kinematics and optional recorded-frame inference check.

Never creates a robot interface, opens CAN, or opens a camera.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pinocchio
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.real_on_blue_block.yaml")))
    parser.add_argument("--episode", type=Path, help="Also load the checkpoint and infer from this HDF5's first frame")
    args = parser.parse_args()

    prefix = Path(sys.prefix).resolve()
    pin_path = Path(pinocchio.__file__).resolve()
    if not pin_path.is_relative_to(prefix):
        raise RuntimeError(f"Pinocchio was loaded from outside this environment: {pin_path}")
    print(f"Python: {sys.executable}")
    print(f"NumPy: {np.__version__}; Torch: {torch.__version__}; Pinocchio: {pinocchio.__version__}")
    print(f"Pinocchio path: {pin_path}")
    values = np.arange(6, dtype=np.float32)
    np.testing.assert_array_equal(torch.from_numpy(values).numpy(), values)

    import real_vla
    from el_a3_sdk.kinematics import ELA3Kinematics

    deployment, _ = real_vla.load_deployment_config(args.config)
    checked = real_vla.static_check(deployment)
    real_vla.print_static_report(deployment, checked)
    collector = checked["collector_config"]
    q = np.deg2rad(collector["episode_reset"]["target_deg"]).tolist()
    kin = ELA3Kinematics()
    pose = kin.forward_kinematics(q)
    gravity = np.asarray(kin.compute_gravity(q))
    pose_values = [pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz]
    if not np.isfinite(pose_values).all() or not np.isfinite(gravity).all():
        raise RuntimeError("Nonfinite kinematics/gravity output")
    print(f"Home FK: {pose_values}; gravity: {gravity.tolist()}")

    if args.episode:
        camera = deployment["hardware"]["camera_name"]
        with h5py.File(args.episode, "r") as f:
            image = cv2.imdecode(np.frombuffer(f[f"observation/{camera}/rgb"][0], dtype=np.uint8), cv2.IMREAD_COLOR)
            observation = {
                "observation": {camera: {"rgb": image}},
                "endpose": {
                    "left_endpose": f["endpose/left_endpose"][0],
                    "left_gripper": float(f["endpose/left_gripper"][0]),
                },
            }
        if image is None:
            raise RuntimeError("Failed to decode recorded frame")
        agent = real_vla.build_agent(deployment)
        action = np.asarray(agent.step(observation))
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid inference output: {action}")
        print(f"Recorded-frame inference action: {action.tolist()}")
    print("PASS: offline checks completed; no CAN/camera/motor access")


if __name__ == "__main__":
    main()
