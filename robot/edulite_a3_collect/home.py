"""Shared EDULITE-A3 active-position homing routines.

The collector is the reference implementation for episode reset.  Both the
collector and offline hardware replay should use these helpers so their
MoveJ/wait/feedback-verification semantics cannot drift apart.
"""
from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np

from hardware import DEFAULT_LIMITS, _feedback


def _validate_target(target: np.ndarray) -> None:
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("home target must contain six finite joint values")
    if np.any(target < DEFAULT_LIMITS[:, 0] + 0.03) or np.any(
        target > DEFAULT_LIMITS[:, 1] - 0.03
    ):
        raise ValueError(f"home target is outside replay safety limits: {target.tolist()}")


def move_arm_home(
    arm,
    target: Iterable[float],
    *,
    name: str,
    v_max: float,
    a_max: float,
    tolerance: float,
    timeout: float,
    feedback_timeout: float,
    hold_before: bool = True,
) -> np.ndarray:
    """Move one active-position arm home using the collector's exact protocol."""
    target = np.asarray(target, dtype=np.float64)
    _validate_target(target)
    if v_max <= 0 or a_max <= 0 or tolerance <= 0 or timeout <= 0:
        raise ValueError("home velocity, acceleration, tolerance and timeout must be positive")
    start = _feedback(arm, name, feedback_timeout)
    if hold_before and not arm.JointCtrl(*start[:6].tolist(), velocities=[0.0] * 6):
        raise RuntimeError(f"{name} could not enter active position hold before homing")

    deadline = time.monotonic() + timeout
    queued = arm.MoveJ(target.tolist(), v_max=v_max, a_max=a_max, block=False)
    if not queued:
        arm.cancel_motion()
        raise RuntimeError(f"could not queue {name} home trajectory")
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not arm.wait_for_motion(timeout=remaining):
        arm.cancel_motion()
        raise RuntimeError(f"{name} home trajectory timed out")

    last = None
    while time.monotonic() < deadline:
        last = _feedback(arm, name, feedback_timeout)
        error = float(np.max(np.abs(last[:6] - target)))
        if error <= tolerance:
            if not arm.JointCtrl(*target.tolist(), velocities=[0.0] * 6):
                raise RuntimeError(f"{name} could not hold home pose")
            return last
        time.sleep(0.02)
    if last is None:
        raise RuntimeError(f"{name} home verification timed out (no feedback)")
    error_vector = np.rad2deg(last[:6] - target)
    error_deg = float(np.max(np.abs(error_vector)))
    raise RuntimeError(
        f"{name} home verification timed out: "
        f"actual_deg={np.rad2deg(last[:6]).round(2).tolist()}, "
        f"target_deg={np.rad2deg(target).round(2).tolist()}, "
        f"error_deg={error_vector.round(2).tolist()}, "
        f"max_error={error_deg:.2f} deg"
    )


def move_both_home(
    master,
    slave,
    target: Iterable[float],
    *,
    v_max: float,
    a_max: float,
    tolerance: float,
    timeout: float,
    feedback_timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collector-style simultaneous two-arm homing and verification."""
    target = np.asarray(target, dtype=np.float64)
    _validate_target(target)
    deadline = time.monotonic() + timeout
    master_start = _feedback(master, "master", feedback_timeout)
    slave_start = _feedback(slave, "slave", feedback_timeout)
    if not master.JointCtrl(*master_start[:6].tolist(), velocities=[0.0] * 6):
        raise RuntimeError("master could not enter position hold before homing")
    if not slave.JointCtrl(*slave_start[:6].tolist(), velocities=[0.0] * 6):
        raise RuntimeError("slave could not enter position hold before homing")
    if not master.MoveJ(target.tolist(), v_max=v_max, a_max=a_max, block=False):
        master.cancel_motion(); slave.cancel_motion()
        raise RuntimeError("could not queue master home trajectory")
    if not slave.MoveJ(target.tolist(), v_max=v_max, a_max=a_max, block=False):
        master.cancel_motion(); slave.cancel_motion()
        raise RuntimeError("could not queue slave home trajectory")
    for name, arm in (("master", master), ("slave", slave)):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not arm.wait_for_motion(timeout=remaining):
            master.cancel_motion(); slave.cancel_motion()
            raise RuntimeError(f"{name} home trajectory timed out")
    last_master = last_slave = None
    while time.monotonic() < deadline:
        last_master = _feedback(master, "master", feedback_timeout)
        last_slave = _feedback(slave, "slave", feedback_timeout)
        if (
            np.max(np.abs(last_master[:6] - target)) <= tolerance
            and np.max(np.abs(last_slave[:6] - target)) <= tolerance
        ):
            master.JointCtrl(*target.tolist(), velocities=[0.0] * 6)
            slave.JointCtrl(*target.tolist(), velocities=[0.0] * 6)
            return last_master, last_slave
        time.sleep(0.02)
    raise RuntimeError("home verification timed out for master or slave")
