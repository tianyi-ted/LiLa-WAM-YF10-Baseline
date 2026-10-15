#!/usr/bin/env python3
"""Safety-gated replay of a native EDULITE-A3 7D HDF5 demonstration.

This program replays only ``joint_action/vector`` to the can0 task arm.  It
does not load a VLA model or open cameras.  A dry preview is the default;
physical motion requires ``--execute``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty

import h5py
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
COLLECTOR_ROOT = PROJECT_ROOT / "robot" / "edulite_a3_collect"
for path in (COLLECTOR_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hardware import DEFAULT_LIMITS, GripperCalibration, _feedback, _make_arm  # noqa: E402
from home import move_arm_home  # noqa: E402


def load_action(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "joint_action/vector" not in f:
            raise ValueError(f"{path}: missing joint_action/vector")
        action = np.asarray(f["joint_action/vector"][:], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != 7:
        raise ValueError(f"{path}: expected action [T,7], got {action.shape}")
    if action.shape[0] < 2 or not np.isfinite(action).all():
        raise ValueError(f"{path}: action is too short or contains NaN/Inf")
    # Replay is deliberately stricter than the collector: do not silently
    # send an out-of-range demonstration to the hardware.
    if np.any(action[:, :6] < DEFAULT_LIMITS[:, 0]) or np.any(action[:, :6] > DEFAULT_LIMITS[:, 1]):
        raise ValueError(f"{path}: L1-L6 action exceeds EDULITE physical limits")
    if np.any(action[:, 6] < -1e-4) or np.any(action[:, 6] > 1.0001):
        raise ValueError(f"{path}: L7 normalized action is outside [0,1]")
    return action


def resolve_episode(root: Path, episode: str, task: str | None) -> Path:
    candidate = Path(episode).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    name = candidate.name if candidate.suffix else f"{candidate.name}.hdf5"
    roots = [root / task] if task else [p for p in root.iterdir() if p.is_dir()]
    matches = sorted(p for r in roots if r.exists() for p in r.glob(f"demo_*/data/{name}"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"episode {episode!r} matched {len(matches)} files; use an explicit .hdf5 path"
        )
    return matches[0].resolve()


def load_collector_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("hardware_calibrated", False):
        raise ValueError("collector config is invalid or hardware_calibrated is false")
    return config


@contextmanager
def keyboard_chars(enabled: bool):
    if not enabled or not sys.stdin.isatty():
        yield
        return
    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        yield
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def read_key() -> str | None:
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    return sys.stdin.read(1) if readable else None


def wait_for_single_key(expected: str, prompt: str) -> bool:
    """Read one raw terminal key, matching the collector's ``a`` control."""
    print(prompt, end="", flush=True)
    if not sys.stdin.isatty():
        return input().strip().lower() == expected
    with keyboard_chars(True):
        while True:
            key = read_key()
            if key is None:
                time.sleep(0.01)
                continue
            print(key, flush=True)
            return key.lower() == expected


def configure_l7_limit(arm, config: dict) -> None:
    from el_a3_sdk.protocol import ParamIndex

    grip = config.get("gripper_control", {})
    limits = (
        (ParamIndex.LIMIT_TORQUE, float(grip.get("motor_torque_limit_nm", 0.10)), "torque"),
        (ParamIndex.LIMIT_CUR, float(grip.get("motor_current_limit_a", 0.30)), "current"),
    )
    for param, value, label in limits:
        if not arm.WriteMotorParameter(7, param, value):
            raise RuntimeError(f"failed to write L7 {label} limit")
        time.sleep(0.03)
        result = arm.ReadMotorParameter(7, param)
        if result is None or not result.success:
            raise RuntimeError(f"failed to verify L7 {label} limit")
        print(f"L7 {label} limit verified: {value:.3f}", flush=True)


def home_arm(arm, config: dict, timeout: float = 20.0) -> None:
    reset = config.get("episode_reset", {})
    target = np.deg2rad(np.asarray(reset.get("target_deg", [0, 5, -5, 0, 0, 0]), dtype=float))
    startup_timeout = float(config.get("safety", {}).get("startup_feedback_timeout_s", 2.0))
    max_velocity = float(reset.get("max_velocity_rad_s", 0.25))
    max_acceleration = float(reset.get("max_acceleration_rad_s2", 0.50))
    tolerance = math.radians(float(reset.get("tolerance_deg", 2.0)))
    timeout = float(reset.get("timeout_s", timeout))
    print(
        "REPLAY HOMING (shared collector reset): "
        f"target_deg={np.rad2deg(target).round(2).tolist()}, "
        f"v_max={max_velocity:.3f} rad/s, a_max={max_acceleration:.3f} rad/s^2, "
        f"tolerance={math.degrees(tolerance):.2f} deg, timeout={timeout:.1f}s",
        flush=True,
    )
    result = move_arm_home(
        arm, target, name="replay arm", v_max=max_velocity,
        a_max=max_acceleration, tolerance=tolerance, timeout=timeout,
        feedback_timeout=startup_timeout,
    )
    print(
        "REPLAY HOME READY: "
        f"actual_deg={np.rad2deg(result[:6]).round(2).tolist()}, "
        f"error_deg={np.rad2deg(result[:6] - target).round(2).tolist()}",
        flush=True,
    )


def replay(action: np.ndarray, arm, config: dict, speed: float, max_seconds: float) -> None:
    if speed <= 0:
        raise ValueError("--speed must be positive")
    rate = float(config.get("collection", {}).get("rate_hz", 30.0))
    period = 1.0 / (rate * speed)
    sdk = config.get("sdk", {})
    max_step = float(sdk.get("max_velocity", 1.0)) * period
    calibration = GripperCalibration(config["arms"]["right"]["gripper"])
    actual = _feedback(arm, "replay arm", 0.25)
    current_grip = calibration.normalize(float(actual[6]))
    last_target = actual[:6].copy()
    force_cfg = config.get("gripper_control", {}).get("force_stop", {})
    reopen_margin = float(
        config.get("master_gripper_control", {}).get(
            "reopen_latch_margin_norm", 0.05
        )
    )
    if reopen_margin <= 0:
        raise ValueError("master_gripper_control.reopen_latch_margin_norm must be positive")
    hard_torque = float(force_cfg.get("hard_torque_limit_nm", 0.18))
    force_stop = bool(force_cfg.get("enabled", True))
    grip_latched = False
    torque_ema = 0.0
    ema_alpha = float(force_cfg.get("ema_alpha", 0.5))
    stall_since: float | None = None
    stall_time = float(force_cfg.get("stall_time_s", 0.30))
    stall_velocity = float(force_cfg.get("max_stall_velocity_rad_s", 0.01))
    stall_torque = float(force_cfg.get("stall_torque_threshold_nm", 0.07))
    started = time.monotonic()
    print("REPLAY ACTIVE: e=E-STOP, q=abort, space=pause/resume", flush=True)
    paused = False
    with keyboard_chars(True):
        for index, sample in enumerate(action):
            if time.monotonic() - started > max_seconds:
                raise RuntimeError(f"replay exceeded --max-seconds={max_seconds}")
            while True:
                key = read_key()
                if key == "e":
                    raise KeyboardInterrupt("software emergency stop")
                if key == "q":
                    raise KeyboardInterrupt("replay aborted")
                if key == " ":
                    paused = not paused
                    print("REPLAY PAUSED" if paused else "REPLAY RESUMED", flush=True)
                if not paused:
                    break
                time.sleep(0.05)

            desired = np.clip(sample[:6], DEFAULT_LIMITS[:, 0] + 0.03, DEFAULT_LIMITS[:, 1] - 0.03)
            delta = np.clip(desired - last_target, -max_step, max_step)
            target = last_target + delta
            if not arm.JointCtrl(*target.tolist(), velocities=(delta / period).tolist()):
                raise RuntimeError(f"joint replay command failed at frame {index}")
            last_target = target

            desired_grip = float(np.clip(sample[6], 0.0, 1.0))
            feedback = _feedback(arm, "replay arm", 0.25)
            current_grip = calibration.normalize(float(feedback[6]))
            if grip_latched and desired_grip >= current_grip + reopen_margin:
                grip_latched = False
                torque_ema = 0.0
                stall_since = None
                print(
                    f"REPLAY GRIPPER FORCE LATCH CLEARED at frame {index}: "
                    f"target={desired_grip:.3f}, current={current_grip:.3f}",
                    flush=True,
                )
            if force_stop and desired_grip < current_grip - 0.015 and not grip_latched:
                effort = abs(float(arm.GetArmJointEfforts().joint_7))
                torque_ema = ema_alpha * effort + (1.0 - ema_alpha) * torque_ema
                velocity = abs(float(arm.GetArmJointVelocities().joint_7))
                if effort >= hard_torque:
                    grip_latched = True
                    print(f"REPLAY GRIPPER FORCE STOP at frame {index}: raw={effort:.3f} Nm", flush=True)
                elif velocity <= stall_velocity and torque_ema >= stall_torque:
                    stall_since = time.monotonic() if stall_since is None else stall_since
                    if time.monotonic() - stall_since >= stall_time:
                        grip_latched = True
                        print(
                            f"REPLAY GRIPPER STALL STOP at frame {index}: "
                            f"raw={effort:.3f} Nm, ema={torque_ema:.3f} Nm",
                            flush=True,
                        )
                else:
                    stall_since = None
            else:
                stall_since = None
            if not grip_latched:
                if not arm.GripperCtrl(calibration.closed + desired_grip * (calibration.open - calibration.closed)):
                    raise RuntimeError(f"gripper replay command failed at frame {index}")
            else:
                # Hold the measured contact angle; never keep tightening.
                if not arm.GripperCtrl(float(feedback[6])):
                    raise RuntimeError("failed to hold gripper after force stop")
            time.sleep(period)
    print(f"REPLAY FINISHED: {len(action)} frames", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=COLLECTOR_ROOT / "config.master_slave.yaml")
    parser.add_argument("--dataset-root", type=Path, default=COLLECTOR_ROOT / "collected_data_single")
    parser.add_argument("--episode", required=True, help="episode0, episode0.hdf5, or an explicit HDF5 path")
    parser.add_argument("--task", default="place_red_block_into_box")
    parser.add_argument("--speed", type=float, default=0.5, help="time scaling; 0.5 is half speed")
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--execute", action="store_true", help="enable motors and move the real arm")
    args = parser.parse_args()

    config = load_collector_config(args.config.resolve())
    reset_cfg = config.get("episode_reset", {})
    print(
        f"COLLECTOR CONFIG: {args.config.resolve()} | "
        f"home_target_deg={reset_cfg.get('target_deg', [0, 5, -5, 0, 0, 0])}",
        flush=True,
    )
    path = resolve_episode(args.dataset_root.resolve(), args.episode, args.task)
    action = load_action(path)
    print(f"EPISODE: {path}")
    print(f"FRAMES: {len(action)}, recorded_rate={config.get('collection', {}).get('rate_hz', 30.0)} Hz")
    print(f"ACTION RANGE: min={action.min(axis=0).round(4).tolist()} max={action.max(axis=0).round(4).tolist()}")
    if not args.execute:
        print("DRY PREVIEW ONLY: add --execute to enable can0 and move the arm.")
        return 0

    arm = None
    try:
        arm_cfg = config["arms"]["right"]
        arm = _make_arm(arm_cfg, config.get("sdk", {}))
        if not arm.ConnectPort():
            raise RuntimeError("cannot connect follower/task arm")
        if not wait_for_single_key(
            "a",
            "Press 'a' to enable the task arm and move it to home "
            "(anything else aborts): ",
        ):
            print("REPLAY ABORTED: enable/home key was not 'a'", flush=True)
            return 2
        if not arm.EnableArm():
            raise RuntimeError("cannot enable follower/task arm")
        configure_l7_limit(arm, config)
        arm.start_control_loop(float(config.get("sdk", {}).get("control_rate_hz", 200.0)))
        initial_feedback = _feedback(arm, "replay arm", 2.0)
        # Establish the same active-position hold used by the collector before
        # entering the shared home trajectory.
        if not arm.JointCtrl(
            *initial_feedback[:6].tolist(), velocities=[0.0] * 6
        ):
            raise RuntimeError("replay arm could not establish startup position hold")
        if not args.no_home:
            home_arm(arm, config)
        input("Press Enter to start replay; keep the physical E-STOP reachable: ")
        replay(action, arm, config, args.speed, args.max_seconds)
        return 0
    except KeyboardInterrupt as exc:
        print(f"\nREPLAY STOPPED: {exc}", flush=True)
        return 2
    finally:
        if arm is not None:
            try:
                arm.EmergencyStop()
            finally:
                arm.DisconnectPort()


if __name__ == "__main__":
    raise SystemExit(main())
