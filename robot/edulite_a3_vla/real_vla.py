#!/usr/bin/env python3
"""Safety-gated 7-action / 8-state VLA inference for one EDULITE-A3 arm."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import math
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import BSpline, make_lsq_spline


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
COLLECTOR_ROOT = PROJECT_ROOT / "robot" / "edulite_a3_collect"
for path in (HERE, PROJECT_ROOT, COLLECTOR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _make_pinocchio_visible() -> None:
    """Expose the existing edulite_collect Pinocchio wheel to RoboTwin.

    The EDULITE SDK uses raw SocketCAN and needs no python-can package.  Only
    Pinocchio's Python module is absent from the RoboTwin environment.  The
    existing wheel is ABI-compatible with the current Python 3.10 runtime.
    """

    try:
        import pinocchio  # noqa: F401
        return
    except ImportError:
        pass

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates: list[Path] = []
    explicit = os.environ.get("EDULITE_PINOCCHIO_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    # .../envs/RoboTwin/bin/python -> .../envs/edulite_collect
    envs_root = Path(sys.executable).resolve().parents[2]
    candidates.append(
        envs_root
        / "edulite_collect"
        / "lib"
        / pyver
        / "site-packages"
        / "cmeel.prefix"
        / "lib"
        / pyver
        / "site-packages"
    )
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))
            try:
                import pinocchio  # noqa: F401
                return
            except ImportError:
                continue
    raise RuntimeError(
        "Pinocchio is unavailable. Activate RoboTwin and set "
        "EDULITE_PINOCCHIO_PATH to edulite_collect's "
        "cmeel.prefix/lib/python3.10/site-packages directory."
    )


_make_pinocchio_visible()

from cameras import build_cameras, load_camera_profile  # noqa: E402
from filters import CriticallyDampedJointFilter  # noqa: E402
from hardware import (  # noqa: E402
    DEFAULT_LIMITS,
    GripperCalibration,
    _feedback,
    _make_arm,
    _pose,
    _wait_feedback_ready,
)
from visualization import InferenceVisualizer  # noqa: E402
from runtime_diagnostics import RuntimeDiagnostics  # noqa: E402
from trajectory_refit import fit_local_prefix  # noqa: E402


def _model_camera_image(camera_rgb: np.ndarray) -> np.ndarray:
    """Return the camera frame in the training input channel convention.

    ``cameras.py`` exposes frames as RGB (it converts the OpenCV BGR capture
    immediately).  The dataset loader decodes the stored JPEG without a
    BGR2RGB swap, preserving the same numeric channel order.  Keep this
    explicit identity boundary so preview and execution use the same pixels.
    """
    rgb = np.asarray(camera_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 camera RGB, got {rgb.shape}/{rgb.dtype}")
    # --- REAL-ROBOT MODEL INPUT: keep training-compatible RGB numeric order ---
    return np.ascontiguousarray(rgb)


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_deployment_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("deployment configuration root must be a mapping")
    for section in ("model", "hardware", "runtime", "safety"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing configuration section: {section}")
    base = config_path.parent
    for key in ("config_path", "checkpoint_path", "norm_stats_path"):
        config["model"][key] = str(_resolve(base, config["model"][key]))
    config["hardware"]["collector_config_path"] = str(
        _resolve(base, config["hardware"]["collector_config_path"])
    )
    visualization = config.setdefault("visualization", {})
    if visualization.get("output_root"):
        visualization["output_root"] = str(
            _resolve(base, visualization["output_root"])
        )
    return config, config_path


def load_collector_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("collector configuration root must be a mapping")
    load_camera_profile(config, path)
    if not bool(config.get("hardware_calibrated", False)):
        raise RuntimeError("collector configuration has hardware_calibrated=false")
    return config


def load_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stats = data.get("robotwin2")
    if not isinstance(stats, dict):
        raise ValueError("normalization file is missing robotwin2")
    action = stats.get("action", {})
    state = stats.get("state", {})
    action_min = np.asarray(action.get("min"), dtype=np.float64)
    action_max = np.asarray(action.get("max"), dtype=np.float64)
    state_min = np.asarray(state.get("min"), dtype=np.float64)
    state_max = np.asarray(state.get("max"), dtype=np.float64)
    if action_min.shape != (7,) or action_max.shape != (7,):
        raise ValueError("normalization action stats must be 7-D")
    if state_min.shape != (8,) or state_max.shape != (8,):
        raise ValueError("normalization state stats must be 8-D")
    for name, value in {
        "action_min": action_min,
        "action_max": action_max,
        "state_min": state_min,
        "state_max": state_max,
    }.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN/Inf")
    if np.any(action_min > action_max) or np.any(state_min > state_max):
        raise ValueError("normalization min exceeds max")
    return action_min, action_max, state_min, state_max, stats.get("meta", {})


def _chunk_update_mode(config: dict) -> str:
    mode = str(config["runtime"].get("chunk_update_mode", "drain_queue"))
    if mode not in {"drain_queue", "latest_ready"}:
        raise ValueError("runtime.chunk_update_mode must be drain_queue or latest_ready")
    return mode


def _time_alignment_enabled(config: dict, prediction_steps: int, horizon: int) -> bool:
    cfg = config["runtime"].get("time_alignment", {})
    if not isinstance(cfg, dict) or not isinstance(cfg.get("enabled", False), bool):
        raise ValueError("runtime.time_alignment must be a mapping with boolean enabled")
    enabled = cfg.get("enabled", False)
    if enabled and prediction_steps <= horizon:
        raise ValueError("time alignment requires action_chunk_size > action_execution_horizon")
    return enabled


def _joint_bspline_settings(config: dict, horizon: int) -> dict[str, Any]:
    cfg = config["runtime"].get("joint_bspline", {})
    if not isinstance(cfg, dict):
        raise ValueError("runtime.joint_bspline must be a mapping")
    settings = {
        "enabled": bool(cfg.get("enabled", False)),
        "method": str(cfg.get("method", "weighted_history")),
        "degree": int(cfg.get("degree", 3)),
        "history_actions": int(cfg.get("history_actions", 8)),
        "control_points": int(cfg.get("control_points", 8)),
        "history_weight": float(cfg.get("history_weight", 4.0)),
        "fallback_if_worse": bool(cfg.get("fallback_if_worse", True)),
        "refit_free_control_points": int(cfg.get("refit_free_control_points", 3)),
        "refit_regularization": float(cfg.get("refit_regularization", 0.05)),
    }
    # B-spline is an optional deployment post-process.  When disabled, its
    # structural parameters are inert placeholders; do not reject a config
    # such as history_actions=0 merely because the smoother is not used.
    if not settings["enabled"]:
        return settings
    if settings["method"] not in {"weighted_history", "prefix_refit"}:
        raise ValueError("runtime.joint_bspline.method must be weighted_history or prefix_refit")
    if settings["degree"] != 3:
        raise ValueError("runtime.joint_bspline.degree must be 3")
    if settings["history_actions"] < 2:
        raise ValueError("runtime.joint_bspline.history_actions must be >= 2")
    if settings["control_points"] < settings["degree"] + 1:
        raise ValueError(
            "runtime.joint_bspline.control_points must be >= degree + 1"
        )
    if settings["control_points"] > settings["history_actions"] + horizon:
        raise ValueError(
            "runtime.joint_bspline.control_points cannot exceed history + horizon"
        )
    if not math.isfinite(settings["history_weight"]) or settings["history_weight"] <= 0.0:
        raise ValueError("runtime.joint_bspline.history_weight must be positive")
    if settings["method"] == "prefix_refit":
        if settings["control_points"] > horizon:
            raise ValueError("prefix_refit.control_points cannot exceed the future horizon")
        if not 1 <= settings["refit_free_control_points"] < settings["control_points"]:
            raise ValueError("refit_free_control_points must be in [1, control_points)")
        if not math.isfinite(settings["refit_regularization"]) or settings["refit_regularization"] < 0:
            raise ValueError("refit_regularization must be finite and nonnegative")
    return settings


def static_check(config: dict) -> dict:
    model_cfg = Path(config["model"]["config_path"])
    checkpoint = Path(config["model"]["checkpoint_path"])
    stats_path = Path(config["model"]["norm_stats_path"])
    collector_path = Path(config["hardware"]["collector_config_path"])
    for label, path in {
        "model config": model_cfg,
        "checkpoint": checkpoint,
        "normalization stats": stats_path,
        "collector config": collector_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    with model_cfg.open("r", encoding="utf-8") as handle:
        train_cfg = yaml.safe_load(handle)
    common = train_cfg.get("common", {})
    if int(common.get("action_dim", -1)) != 7:
        raise ValueError("trained model action_dim is not 7")
    if int(common.get("state_dim", -1)) != 8:
        raise ValueError("trained model state_dim is not 8")
    cameras = train_cfg.get("dataset", {}).get("camera_names", [])
    requested_camera = config["hardware"]["camera_name"]
    if cameras != [requested_camera]:
        raise ValueError(
            f"model cameras {cameras!r} do not match runtime camera {requested_camera!r}"
        )
    state_indices = train_cfg.get("dataset", {}).get("indices_config", {}).get(
        "state_indices", []
    )
    if state_indices != [0]:
        raise ValueError(
            "the real prefetch runtime currently requires state_indices=[0]"
        )
    if bool(train_cfg.get("model", {}).get("use_task_cond", False)):
        cond_root = train_cfg.get("dataset", {}).get("task_cond_dir")
        task_name = config["model"].get("task_name")
        if not cond_root or not task_name:
            raise ValueError("task-conditioned model needs dataset.task_cond_dir and model.task_name")
        condition = Path(cond_root) / str(task_name) / "task_cond.npy"
        if not condition.is_file():
            raise FileNotFoundError(f"task condition not found: {condition}")
        cond_vec = np.load(condition)
        if cond_vec.ndim != 1 or not np.isfinite(cond_vec).all():
            raise ValueError(f"invalid task condition: {condition}")
    horizon = int(common.get("action_execution_horizon", -1))
    if horizon <= 0:
        raise ValueError("trained model action_execution_horizon must be positive")
    rate = float(config["runtime"].get("action_rate_hz", 30.0))
    threshold = int(config["runtime"].get("prefetch_threshold_actions", horizon // 2))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("runtime.action_rate_hz must be finite and positive")
    if not 0 <= threshold < horizon:
        raise ValueError("prefetch_threshold_actions must be in [0, horizon)")
    bspline = _joint_bspline_settings(config, horizon)
    _chunk_update_mode(config)
    _time_alignment_enabled(config, int(common.get("action_chunk_size", -1)), horizon)

    collector = load_collector_config(collector_path)
    side = str(config["hardware"].get("arm_side", "right"))
    if side not in {"left", "right"}:
        raise ValueError("hardware.arm_side must be left or right")
    camera_configs = [c for c in collector.get("cameras", []) if c.get("enabled", True)]
    if requested_camera not in {str(c.get("name")) for c in camera_configs}:
        raise ValueError(f"camera {requested_camera!r} is absent from collector config")

    action_min, action_max, state_min, state_max, meta = load_stats(stats_path)
    return {
        "train_config": train_cfg,
        "collector_config": collector,
        "action_min": action_min,
        "action_max": action_max,
        "state_min": state_min,
        "state_max": state_max,
        "stats_meta": meta,
        "joint_bspline": bspline,
    }


class EDULITEVLAController:
    """One-arm controller with an independent 100 Hz safety loop."""

    def __init__(self, deployment: dict, checked: dict):
        self.deployment = deployment
        self.collector = checked["collector_config"]
        self.safety = deployment["safety"]
        self.runtime = deployment["runtime"]
        side = str(deployment["hardware"].get("arm_side", "right"))
        self.arm_cfg = self.collector["arms"][side]
        self.arm = _make_arm(self.arm_cfg, self.collector.get("sdk", {}))
        self.gripper = GripperCalibration(self.arm_cfg["gripper"])

        action_min = checked["action_min"][:7]
        action_max = checked["action_max"][:7]
        margin = float(self.safety.get("joint_limit_margin_rad", 0.15))
        mechanical_lower = DEFAULT_LIMITS[:, 0] + margin
        mechanical_upper = DEFAULT_LIMITS[:, 1] - margin
        if bool(self.safety.get("clamp_to_training_range", True)):
            self.lower = np.maximum(mechanical_lower, action_min[:6])
            self.upper = np.minimum(mechanical_upper, action_max[:6])
        else:
            self.lower, self.upper = mechanical_lower, mechanical_upper
        if np.any(self.lower >= self.upper):
            raise ValueError("runtime joint limits are empty after intersecting training ranges")
        home_deg = self.collector.get("episode_reset", {}).get(
            "target_deg", [0, 5, -5, 0, 0, 0]
        )
        home_target = np.deg2rad(np.asarray(home_deg, dtype=np.float64))
        if home_target.shape != (6,) or np.any(home_target < self.lower - 0.05) or np.any(
            home_target > self.upper + 0.05
        ):
            raise ValueError(
                "configured home pose is outside runtime joint limits "
                "(training range intersected with mechanical limits and safety.joint_limit_margin_rad); "
                f"home_deg={np.rad2deg(home_target).round(3).tolist()}, "
                f"runtime_min_deg={np.rad2deg(self.lower).round(3).tolist()}, "
                f"runtime_max_deg={np.rad2deg(self.upper).round(3).tolist()}; "
                "check the mechanical margin as well as the demonstrations and normalization stats"
            )
        self.raw_lower = DEFAULT_LIMITS[:, 0]
        self.raw_upper = DEFAULT_LIMITS[:, 1]
        self.train_lower = action_min[:6]
        self.train_upper = action_max[:6]

        self.feedback_timeout = float(self.safety.get("feedback_timeout_s", 0.2))
        self.startup_timeout = float(self.safety.get("startup_feedback_timeout_s", 2.0))
        self.control_rate = float(self.runtime.get("control_rate_hz", 100.0))
        self.max_target_step = np.broadcast_to(
            np.asarray(self.safety.get("max_target_step_rad", 0.1), dtype=float), (6,)
        ).copy()
        self.follow_error_limit = float(self.safety.get("max_follow_error_rad", 0.3))
        self.follow_error_cycles = int(self.safety.get("follow_error_cycles", 20))
        self.raw_tolerance = float(self.safety.get("raw_joint_limit_tolerance_rad", 0.25))
        self.raw_gripper_tolerance = float(self.safety.get("raw_gripper_tolerance", 0.2))
        self.gripper_filter_alpha = float(
            self.safety.get("gripper_filter_alpha", 0.45)
        )
        self.gripper_max_target_rate = float(
            self.safety.get("gripper_max_target_rate_norm_s", 1.0)
        )
        self.gripper_command_deadband = float(
            self.safety.get("gripper_command_deadband_norm", 0.005)
        )
        if not 0.0 < self.gripper_filter_alpha <= 1.0:
            raise ValueError("safety.gripper_filter_alpha must be in (0, 1]")
        if self.gripper_max_target_rate <= 0.0:
            raise ValueError(
                "safety.gripper_max_target_rate_norm_s must be positive"
            )
        if not 0.0 <= self.gripper_command_deadband <= 0.2:
            raise ValueError(
                "safety.gripper_command_deadband_norm must be in [0, 0.2]"
            )
        action_rate = float(self.runtime.get("action_rate_hz", 30.0))
        if action_rate <= 0.0:
            raise ValueError("runtime.action_rate_hz must be positive")
        # Model actions arrive at a fixed cadence.  Using that cadence instead
        # of wall-clock time prevents a preview/start delay from admitting one
        # unbounded first L7 step.
        self._gripper_max_step = self.gripper_max_target_rate / action_rate

        grip_cfg = self.collector.get("gripper_control", {})
        force = grip_cfg.get("force_stop", {})
        self.gripper_torque_limit = float(grip_cfg.get("motor_torque_limit_nm", 0.10))
        self.gripper_current_limit = float(grip_cfg.get("motor_current_limit_a", 0.30))
        self.force_enabled = bool(force.get("enabled", True))
        self.force_soft = float(force.get("torque_threshold_nm", 0.12))
        self.force_hard = float(force.get("hard_torque_limit_nm", 0.18))
        self.force_alpha = float(force.get("ema_alpha", 0.5))
        self.force_samples_required = int(force.get("consecutive_samples", 3))
        self.force_contact_velocity = float(force.get("max_contact_velocity_rad_s", 0.05))
        self.force_min_error = float(force.get("min_target_error_norm", 0.015))
        self.force_stall_time = float(force.get("stall_time_s", 0.30))
        self.force_stall_velocity = float(force.get("max_stall_velocity_rad_s", 0.01))
        self.force_stall_torque = float(force.get("stall_torque_threshold_nm", 0.07))
        self.force_backoff = float(force.get("backoff_norm", 0.0))
        self.reopen_margin = float(
            self.collector.get("master_gripper_control", {}).get(
                "reopen_latch_margin_norm", 0.05
            )
        )

        self.connected = False
        self.armed = False
        self.fault: str | None = None
        self._filter: CriticallyDampedJointFilter | None = None
        self._target = np.zeros(6, dtype=np.float64)
        self._sent = np.zeros(6, dtype=np.float64)
        self._last_model_command = 0.0
        # Raw, validated model intent is kept separately from the conditioned
        # L7 target so diagnostics can expose action-chunk discontinuities.
        self._gripper_target = self.gripper.value
        self._gripper_filtered_target = self.gripper.value
        self._gripper_sent = self.gripper.value
        self._gripper_latched = False
        self._gripper_hold = self.gripper.value
        self._gripper_ema = 0.0
        self._gripper_samples = 0
        self._stall_since: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._emergency = threading.Event()
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        if self.connected:
            return
        if not self.arm.ConnectPort():
            raise RuntimeError(f"cannot connect task arm on {self.arm_cfg['can_name']}")
        self.connected = True

    def _configure_gripper_limits(self) -> None:
        from el_a3_sdk.protocol import ParamIndex

        for parameter, limit, label, unit in (
            (ParamIndex.LIMIT_TORQUE, self.gripper_torque_limit, "torque", "Nm"),
            (ParamIndex.LIMIT_CUR, self.gripper_current_limit, "current", "A"),
        ):
            if not self.arm.WriteMotorParameter(7, parameter, limit):
                raise RuntimeError(f"failed to write L7 {label} limit")
            time.sleep(0.03)
            result = self.arm.ReadMotorParameter(7, parameter)
            tolerance = max(0.01, 0.1 * limit)
            if (
                result is None
                or not result.success
                or abs(float(result.value) - limit) > tolerance
            ):
                raise RuntimeError(f"failed to verify L7 {label} limit")
            print(f"L7 {label} limit verified: {float(result.value):.3f} {unit}")

    def arm_and_home(self) -> None:
        if self.armed:
            return
        if self._emergency.is_set():
            raise RuntimeError(
                "software emergency stop is latched; restart the program before re-arming"
            )
        if not self.connected:
            self.connect()
        if not self.arm.EnableArm():
            raise RuntimeError("task arm failed to enable")
        try:
            self._configure_gripper_limits()
            sdk_rate = float(self.collector.get("sdk", {}).get("control_rate_hz", 200.0))
            initial = _wait_feedback_ready(
                self.arm, "task arm", self.feedback_timeout, self.startup_timeout
            )
            self.gripper.value = self.gripper.normalize(initial[6])
            self._gripper_target = self.gripper.value
            self._gripper_filtered_target = self.gripper.value
            self._gripper_sent = self.gripper.value
            self.arm.start_control_loop(sdk_rate)
            if not self.arm.JointCtrl(*initial[:6].tolist(), velocities=[0.0] * 6):
                raise RuntimeError("could not establish initial joint hold")
            self.armed = True
            self.home()
            feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
            self._target = feedback[:6].copy()
            self._sent = feedback[:6].copy()
            self._filter = CriticallyDampedJointFilter(
                feedback[:6],
                omega=float(self.safety.get("filter_omega", 10.0)),
                max_velocity=self.safety.get("max_velocity_rad_s", 0.35),
                max_acceleration=self.safety.get("max_acceleration_rad_s2", 1.2),
                lower_limits=self.lower,
                upper_limits=self.upper,
            )
            self._last_model_command = time.monotonic()
            self.fault = None
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._control_loop, name="edulite-vla-safety", daemon=True
            )
            self._thread.start()
        except Exception:
            self.emergency_stop()
            raise

    def home(self) -> None:
        if not self.armed:
            raise RuntimeError("arm is not enabled")
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("stop the runtime controller before homing")
        cfg = self.collector.get("episode_reset", {})
        target = np.deg2rad(np.asarray(cfg.get("target_deg", [0, 5, -5, 0, 0, 0]), dtype=float))
        target = np.clip(target, DEFAULT_LIMITS[:, 0] + 0.03, DEFAULT_LIMITS[:, 1] - 0.03)
        initial = _feedback(self.arm, "task arm", self.feedback_timeout)
        filt = CriticallyDampedJointFilter(
            initial[:6],
            omega=6.0,
            max_velocity=float(cfg.get("max_velocity_rad_s", 0.25)),
            max_acceleration=float(cfg.get("max_acceleration_rad_s2", 0.5)),
            lower_limits=DEFAULT_LIMITS[:, 0] + 0.03,
            upper_limits=DEFAULT_LIMITS[:, 1] - 0.03,
        )
        tolerance = math.radians(float(cfg.get("tolerance_deg", 2.0)))
        timeout = float(cfg.get("timeout_s", 20.0))
        deadline = time.monotonic() + timeout
        previous = time.monotonic()
        print(f"HOMING task arm to {np.rad2deg(target).round(2).tolist()} deg")
        while time.monotonic() < deadline:
            if self._emergency.is_set():
                raise RuntimeError("homing interrupted by emergency stop")
            now = time.monotonic()
            state = filt.step(target, max(now - previous, 1e-4))
            previous = now
            if not self.arm.JointCtrl(*state.position.tolist(), velocities=state.velocity.tolist()):
                raise RuntimeError("home joint command failed")
            feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
            if np.max(np.abs(feedback[:6] - target)) <= tolerance:
                self.arm.JointCtrl(*target.tolist(), velocities=[0.0] * 6)
                print("HOME READY: task arm is holding the trained start pose")
                return
            time.sleep(0.01)
        raise RuntimeError("homing timed out")

    def validate_model_action(self, action: np.ndarray) -> tuple[np.ndarray, list[str]]:
        raw = np.asarray(action, dtype=np.float64)
        if raw.shape != (7,) or not np.isfinite(raw).all():
            raise RuntimeError("model action must be finite with shape (7,)")
        q_raw = raw[:6]
        if np.any(q_raw < self.raw_lower - self.raw_tolerance) or np.any(
            q_raw > self.raw_upper + self.raw_tolerance
        ):
            raise RuntimeError(f"model proposed joints beyond hard limits: {q_raw.tolist()}")
        grip_raw = float(raw[6])
        if grip_raw < -self.raw_gripper_tolerance or grip_raw > 1.0 + self.raw_gripper_tolerance:
            raise RuntimeError(f"model proposed invalid gripper value: {grip_raw:.3f}")

        warnings: list[str] = []
        q = np.clip(q_raw, self.lower, self.upper)
        if not np.allclose(q, q_raw, atol=1e-6):
            warnings.append("joint target clipped to trained/mechanical range")
        grip = float(np.clip(grip_raw, 0.0, 1.0))
        if abs(grip - grip_raw) > 1e-6:
            warnings.append("gripper target clipped to [0,1]")
        return np.r_[q, grip], warnings

    def set_model_action(self, action: np.ndarray) -> list[str]:
        safe, warnings = self.validate_model_action(action)
        with self._lock:
            stepped = np.clip(
                safe[:6],
                self._target - self.max_target_step,
                self._target + self.max_target_step,
            )
            if not np.allclose(stepped, safe[:6], atol=1e-6):
                warnings.append("joint target rate-clipped")
            self._target = stepped
            self._gripper_target = float(safe[6])
            if self._gripper_latched and safe[6] < self._gripper_hold + self.reopen_margin:
                # Do not accumulate a hidden closing command behind a force
                # latch.  Otherwise releasing the latch could briefly command
                # another close before the filter catches up.
                self._gripper_filtered_target = self._gripper_hold
            else:
                ema_target = (
                    self.gripper_filter_alpha * float(safe[6])
                    + (1.0 - self.gripper_filter_alpha)
                    * self._gripper_filtered_target
                )
                delta = float(
                    np.clip(
                        ema_target - self._gripper_filtered_target,
                        -self._gripper_max_step,
                        self._gripper_max_step,
                    )
                )
                self._gripper_filtered_target = float(
                    np.clip(self._gripper_filtered_target + delta, 0.0, 1.0)
                )
            self._last_model_command = time.monotonic()
        return warnings

    def joint_target_snapshot(self) -> np.ndarray:
        """Return the current safe L1-L6 target without accessing CAN."""

        with self._lock:
            return self._target.copy()

    def joint_command_snapshot(self) -> np.ndarray:
        """Last filtered L1-L6 command, without reading CAN or changing control."""
        with self._lock:
            return self._sent.copy()

    def hold_current(self) -> None:
        if not self.armed:
            return
        feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
        with self._lock:
            self._target = np.clip(feedback[:6], self.lower, self.upper)
            self._sent = self._target.copy()
            self._gripper_target = self.gripper.normalize(feedback[6])
            self._gripper_filtered_target = self._gripper_target
            self._gripper_sent = self._gripper_target
            self._last_model_command = time.monotonic()
            if self._filter is not None:
                self._filter.reset(self._target)
        self.arm.JointCtrl(*self._target.tolist(), velocities=[0.0] * 6)

    def return_home(self) -> None:
        """Pause the runtime loop, home L1-L6, then resume safe holding.

        L7 deliberately keeps its current opening during this transition. A
        new model preview is required by the UI before the next execution.
        """

        if not self.armed:
            raise RuntimeError("arm is not enabled")
        if self._emergency.is_set():
            raise RuntimeError(
                "software emergency stop is latched; restart before homing"
            )
        if self.fault:
            raise RuntimeError(f"cannot home after controller fault: {self.fault}")

        # Give the homing routine exclusive ownership of JointCtrl.  This is a
        # normal pause, not an emergency stop, so the motors remain enabled.
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                raise RuntimeError("runtime control thread did not stop before homing")
        self._thread = None

        try:
            self.home()
            feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
            with self._lock:
                self._target = np.clip(feedback[:6], self.lower, self.upper)
                self._sent = self._target.copy()
                self._gripper_target = self.gripper.normalize(feedback[6])
                self._gripper_filtered_target = self._gripper_target
                self._gripper_sent = self._gripper_target
                self._last_model_command = time.monotonic()
                self._filter = CriticallyDampedJointFilter(
                    feedback[:6],
                    omega=float(self.safety.get("filter_omega", 10.0)),
                    max_velocity=self.safety.get("max_velocity_rad_s", 0.35),
                    max_acceleration=self.safety.get("max_acceleration_rad_s2", 1.2),
                    lower_limits=self.lower,
                    upper_limits=self.upper,
                )
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._control_loop,
                name="edulite-vla-safety",
                daemon=True,
            )
            self._thread.start()
            print(
                "NEXT EPISODE HOME READY: L1-L6 are holding the trained start pose; "
                "L7 kept its current opening; press p for a new preview",
                flush=True,
            )
        except Exception:
            self.emergency_stop()
            raise

    def telemetry(self) -> dict[str, Any]:
        """Snapshot model target, filtered command and physical feedback."""

        feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
        efforts = self.arm.GetArmJointEfforts()
        with self._lock:
            return {
                "model_target": np.r_[self._target, self._gripper_target].astype(np.float32),
                "sent": np.r_[self._sent, self._gripper_sent].astype(np.float32),
                "feedback": np.r_[
                    feedback[:6], self.gripper.normalize(feedback[6])
                ].astype(np.float32),
                "gripper_torque_raw": float(efforts.joint_7),
                "gripper_torque_ema": float(self._gripper_ema),
                "gripper_force_latched": bool(self._gripper_latched),
            }

    def predicted_xyz(self, actions7: np.ndarray) -> np.ndarray:
        """Apply SDK FK to a safe action chunk for visualization only."""

        kin = self.arm._get_kinematics()
        if kin is None:
            raise RuntimeError("Pinocchio FK is unavailable for trajectory visualization")
        xyz = []
        for action in np.asarray(actions7):
            pose = kin.forward_kinematics(action[:6].tolist())
            xyz.append([pose.x, pose.y, pose.z])
        return np.asarray(xyz, dtype=np.float32)

    def observation(self, rgb: np.ndarray, camera_mono_s: float | None = None) -> dict[str, Any]:
        observed = time.monotonic()
        feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
        left_pose = _pose(self.arm, self.arm_cfg)
        left_gripper = self.gripper.normalize(feedback[6])
        return {
            "_runtime_timing": {"observation_mono_s": observed,
                                "camera_mono_s": observed if camera_mono_s is None else camera_mono_s},
            "endpose": {
                "left_endpose": left_pose,
                "left_gripper": left_gripper,
            },
            "observation": {
                self.deployment["hardware"]["camera_name"]: {"rgb": rgb}
            },
        }

    def _update_gripper(self, feedback: np.ndarray, now: float) -> None:
        actual_norm = self.gripper.normalize(feedback[6])
        efforts = self.arm.GetArmJointEfforts()
        velocities = self.arm.GetArmJointVelocities()
        raw_torque = abs(float(efforts.joint_7))
        velocity = abs(float(velocities.joint_7))
        with self._lock:
            raw_requested = self._gripper_target
            requested = self._gripper_filtered_target
            self._gripper_ema = (
                self.force_alpha * raw_torque
                + (1.0 - self.force_alpha) * self._gripper_ema
            )
            if self._gripper_latched:
                if raw_requested >= self._gripper_hold + self.reopen_margin:
                    self._gripper_latched = False
                    self._gripper_samples = 0
                    self._stall_since = None
                    print("GRIPPER FORCE LATCH RELEASED by an opening model command")
                else:
                    requested = self._gripper_hold

            closing = requested < actual_norm - self.force_min_error
            stalled_contact = velocity <= self.force_contact_velocity
            if self.force_enabled and closing and stalled_contact and self._gripper_ema >= self.force_soft:
                self._gripper_samples += 1
            else:
                self._gripper_samples = 0
            hard = self.force_enabled and closing and raw_torque >= self.force_hard
            soft = self._gripper_samples >= self.force_samples_required
            stalled = (
                self.force_enabled
                and closing
                and velocity <= self.force_stall_velocity
                and self._gripper_ema >= self.force_stall_torque
            )
            if stalled:
                self._stall_since = self._stall_since or now
            else:
                self._stall_since = None
            stall = self._stall_since is not None and now - self._stall_since >= self.force_stall_time
            trigger = hard or soft or stall
            if trigger and not self._gripper_latched:
                self._gripper_hold = min(1.0, actual_norm + self.force_backoff)
                self._gripper_target = self._gripper_hold
                self._gripper_filtered_target = self._gripper_hold
                requested = self._gripper_hold
                self._gripper_latched = True
                reason = "HARD" if hard else ("SOFT" if soft else "STALL")
                print(
                    f"GRIPPER FORCE STOP ({reason}): raw={raw_torque:.3f} Nm, "
                    f"ema={self._gripper_ema:.3f} Nm, velocity={velocity:.3f} rad/s, "
                    f"hold_norm={self._gripper_hold:.3f}; closing LATCHED",
                    flush=True,
                )
            should_send = (
                abs(requested - self._gripper_sent)
                >= self.gripper_command_deadband
            )
            if should_send:
                self.gripper.value = float(np.clip(requested, 0.0, 1.0))
                angle = self.gripper.radians()
                self._gripper_sent = self.gripper.value
        if should_send and not self.arm.GripperCtrl(angle):
            raise RuntimeError("gripper command failed")

    def _control_loop(self) -> None:
        period = 1.0 / self.control_rate
        deadline = time.monotonic()
        previous = deadline
        bad_cycles = 0
        try:
            while not self._stop.is_set() and not self._emergency.is_set():
                now = time.monotonic()
                feedback = _feedback(self.arm, "task arm", self.feedback_timeout)
                with self._lock:
                    target = self._target.copy()
                    last_command = self._last_model_command
                    filt = self._filter
                if filt is None:
                    raise RuntimeError("runtime joint filter is not initialized")
                # Stale inference never extrapolates; it simply holds target.
                if now - last_command > float(self.safety.get("model_command_timeout_s", 1.0)):
                    target = filt.state.position.copy()
                control_interval = now - previous
                state = filt.step(target, max(control_interval, 1e-4))
                previous = now
                error = float(np.max(np.abs(state.position - feedback[:6])))
                bad_cycles = bad_cycles + 1 if error > self.follow_error_limit else 0
                if bad_cycles >= self.follow_error_cycles:
                    raise RuntimeError(
                        f"following error {error:.3f} rad exceeds {self.follow_error_limit:.3f} rad"
                    )
                if not self.arm.JointCtrl(
                    *state.position.tolist(), velocities=state.velocity.tolist()
                ):
                    raise RuntimeError("joint command failed")
                with self._lock:
                    self._sent = state.position.copy()
                self._update_gripper(feedback, now)
                diagnostics = getattr(self, "diagnostics", None)
                if diagnostics is not None:
                    diagnostics.record("controls", {
                        "control_mono_s": now, "interval_s": control_interval,
                        "work_s": time.monotonic() - now,
                        "following_error_rad": error,
                        "command_age_s": now - last_command,
                        "model_command_timed_out": int(now - last_command > float(self.safety.get("model_command_timeout_s", 1.0))),
                        "max_command_velocity_rad_s": float(np.max(np.abs(state.velocity))),
                    })
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
                if time.monotonic() - deadline > period:
                    deadline = time.monotonic()
        except Exception as exc:
            self.fault = str(exc)
            print(f"\nVLA SAFETY STOP: {exc}", file=sys.stderr, flush=True)
            self.emergency_stop()

    def check_health(self) -> None:
        if self.fault:
            raise RuntimeError(f"VLA safety stop: {self.fault}")
        if self._emergency.is_set():
            raise RuntimeError("emergency stop is active")

    def emergency_stop(self) -> None:
        self._emergency.set()
        self._stop.set()
        self.armed = False
        try:
            self.arm.EmergencyStop()
        except Exception:
            pass

    def close(self) -> None:
        self.emergency_stop()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if self.connected:
            try:
                self.arm.stop_control_loop()
            except Exception:
                pass
            try:
                self.arm.DisconnectPort()
            except Exception:
                pass
            self.connected = False


@contextmanager
def raw_terminal(enabled: bool):
    if not enabled:
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class KeyReader:
    """Read keys independently so `e` remains responsive during GPU inference."""

    def __init__(self, emergency_callback):
        self.keys: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.stop = threading.Event()
        self.emergency_callback = emergency_callback
        self.thread = threading.Thread(target=self._run, name="vla-keys", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not readable:
                continue
            key = sys.stdin.read(1)
            if key == "e":
                self.emergency_callback()
            self.keys.put(key)

    def get(self) -> str | None:
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=0.5)


class CubicBSplineJointSmoother:
    """Optional weighted-history or local-prefix fitting of L1-L6; L7 passes through."""

    def __init__(
        self,
        config: dict,
        controller: EDULITEVLAController,
        horizon: int,
    ):
        cfg = _joint_bspline_settings(config, horizon)
        self.enabled = cfg["enabled"]
        self.method = cfg["method"]
        self.degree = cfg["degree"]
        self.history_actions = cfg["history_actions"]
        self.control_points = cfg["control_points"]
        self.history_weight = cfg["history_weight"]
        self.fallback_if_worse = cfg["fallback_if_worse"]
        self.refit_free_control_points = cfg["refit_free_control_points"]
        self.refit_regularization = cfg["refit_regularization"]
        self.controller = controller
        self.history: deque[np.ndarray] = deque(maxlen=self.history_actions)
        self.last_metrics: dict[str, Any] = {}

    def reset(self, anchor: np.ndarray) -> None:
        value = np.asarray(anchor, dtype=np.float64)
        if value.shape != (6,) or not np.isfinite(value).all():
            raise ValueError("B-spline anchor must be finite with shape (6,)")
        self.history.clear()
        # A repeated prefix represents the stationary hold before execution and
        # prevents the first chunk from inventing a non-zero incoming velocity.
        for _ in range(self.history_actions):
            self.history.append(value.copy())
        self.last_metrics = {}

    def observe(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (7,) or not np.isfinite(value).all():
            raise ValueError("executed action must be finite with shape (7,)")
        self.history.append(value[:6].copy())

    def smooth(self, chunk: np.ndarray) -> np.ndarray:
        value = np.asarray(chunk, dtype=np.float32)
        if not self.enabled:
            self.last_metrics = {}
            return value.copy()
        if value.ndim != 2 or value.shape[1] != 7 or not np.isfinite(value).all():
            raise ValueError("B-spline input chunk must have shape (N,7) and be finite")
        if len(self.history) != self.history_actions:
            raise RuntimeError("B-spline history is not initialized")

        history = np.asarray(self.history, dtype=np.float64)
        raw_future = value[:, :6].astype(np.float64)
        fit_started = time.monotonic()
        fit_metrics = {}
        if self.method == "prefix_refit":
            fit = fit_local_prefix(
                raw_future, history, degree=self.degree, control_points=self.control_points,
                free_control_points=self.refit_free_control_points,
                history_weight=self.history_weight, regularization=self.refit_regularization,
            )
            smooth_future = fit.future
            fit_metrics = {
                "refit_history_rmse_before_rad": fit.history_rmse_before,
                "refit_history_rmse_after_rad": fit.history_rmse_after,
                "refit_control_delta_max_rad": float(np.max(np.abs(fit.control_points - fit.original_control_points))),
            }
        else:
            smooth_future = self._fit_weighted_history(raw_future, history)
        clipped = np.clip(smooth_future, self.controller.lower, self.controller.upper)

        anchor = history[-1]
        raw_boundary = float(np.max(np.abs(raw_future[0] - anchor)))
        smooth_boundary = float(np.max(np.abs(clipped[0] - anchor)))
        raw_delta = np.diff(np.vstack([anchor, raw_future]), axis=0)
        smooth_delta = np.diff(np.vstack([anchor, clipped]), axis=0)
        metrics: dict[str, Any] = {
            "bspline_method": self.method,
            "raw_boundary_rad": raw_boundary,
            "smooth_boundary_rad": smooth_boundary,
            "raw_max_step_rad": float(np.max(np.abs(raw_delta))),
            "smooth_max_step_rad": float(np.max(np.abs(smooth_delta))),
            "fit_mae_rad": float(np.mean(np.abs(clipped - raw_future))),
            "clipped_values": int(np.count_nonzero(np.abs(clipped - smooth_future) > 1e-9)),
            "candidate_boundary_rad": smooth_boundary,
            "candidate_max_step_rad": float(np.max(np.abs(smooth_delta))),
            "fit_compute_s": time.monotonic() - fit_started,
            "fallback": 0,
            **fit_metrics,
        }
        if self.fallback_if_worse and (
            metrics["smooth_boundary_rad"] > metrics["raw_boundary_rad"] + 1e-9
            or metrics["smooth_max_step_rad"] > metrics["raw_max_step_rad"] + 1e-9
        ):
            clipped = raw_future
            metrics["smooth_boundary_rad"] = metrics["raw_boundary_rad"]
            metrics["smooth_max_step_rad"] = metrics["raw_max_step_rad"]
            metrics["fit_mae_rad"] = 0.0
            metrics["clipped_values"] = 0
            metrics["fallback"] = 1
        self.last_metrics = metrics
        result = value.copy()
        result[:, :6] = clipped.astype(np.float32)
        return result

    def _fit_weighted_history(self, raw_future: np.ndarray, history: np.ndarray) -> np.ndarray:
        """The original fitter, preserving its knot vector and weighting exactly."""
        samples = np.concatenate([history, raw_future], axis=0)
        sample_count = samples.shape[0]
        if self.control_points > sample_count:
            raise ValueError(
                "runtime.joint_bspline.control_points cannot exceed history + horizon"
            )

        x = np.arange(sample_count, dtype=np.float64)
        # Open-uniform knot vector, matching the ABPolicy fitter included in
        # robot/ABPolicy-code while allowing a weighted real-action prefix.
        internal = np.linspace(
            x[0], x[-1], self.control_points - self.degree + 1
        )
        knots = np.concatenate(
            ([x[0]] * self.degree, internal, [x[-1]] * self.degree)
        ).astype(np.float64)
        weights = np.ones(sample_count, dtype=np.float64)
        weights[: self.history_actions] = self.history_weight
        spline = make_lsq_spline(
            x,
            samples,
            knots,
            k=self.degree,
            w=weights,
            axis=0,
        )
        if not isinstance(spline, BSpline):
            raise RuntimeError("SciPy did not return a B-spline")
        rebuilt = np.asarray(spline(x), dtype=np.float64)
        return rebuilt[self.history_actions :]

    def metrics_line(self, label: str) -> str | None:
        if not self.last_metrics:
            return None
        m = self.last_metrics
        return (
            f"B-SPLINE {label} ({self.method}): boundary {m['raw_boundary_rad']:.4f}->"
            f"{m['smooth_boundary_rad']:.4f} rad, max_step "
            f"{m['raw_max_step_rad']:.4f}->{m['smooth_max_step_rad']:.4f} rad, "
            f"fit_mae={m['fit_mae_rad']:.4f} rad, clipped={m['clipped_values']}, "
            f"fallback={bool(m['fallback'])}"
        )


class AsyncActionChunks:
    """Prefetch model chunks without ever bursting delayed actions."""

    def __init__(self, agent, controller: EDULITEVLAController, instruction: str, config: dict):
        self.agent = agent
        self.controller = controller
        self.instruction = instruction
        self.horizon = int(agent.action_execution_horizon)
        self.chunk_update_mode = _chunk_update_mode(config)
        self.prediction_steps = int(agent.config.common.action_chunk_size)
        self.time_alignment = _time_alignment_enabled(config, self.prediction_steps, self.horizon)
        self.action_rate = float(config["runtime"].get("action_rate_hz", 30.0))
        if not math.isfinite(self.action_rate) or self.action_rate <= 0:
            raise ValueError("runtime.action_rate_hz must be finite and positive")
        self._alignment_refresh_needed = False
        self.prefetch_threshold = int(
            config["runtime"].get("prefetch_threshold_actions", self.horizon // 2)
        )
        if not 0 <= self.prefetch_threshold < self.horizon:
            raise ValueError("prefetch_threshold_actions must be in [0, horizon)")
        self.actions: deque[np.ndarray] = deque()
        self.pending: np.ndarray | None = None
        self.future: Future | None = None
        self.future_generation: int | None = None
        self.generation = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-prefetch")
        self.smoother = CubicBSplineJointSmoother(
            config, controller, self.horizon
        )
        self.diagnostics = getattr(controller, "diagnostics", None)
        self._chunk_id = 0
        self._future_meta = None
        self._pending_meta = None
        self.current_meta = None
        self.preview_meta = None
        self.preview_metrics = {}
        self._last_emitted_joint = None
        self._current_selected = 0
        self._first_selected_mono = None
        self._last_selected_mono = None
        if self.diagnostics is not None:
            self.diagnostics.interface = {
                "action_chunk_size": int(agent.config.common.action_chunk_size),
                "action_execution_horizon": self.horizon,
                "num_inference_steps": getattr(agent, "num_inference_steps", None),
                "chunk_update_mode": self.chunk_update_mode,
                "time_alignment_enabled": self.time_alignment,
            }

    def _timed_infer(self, observation):
        started = time.monotonic()
        result = self._infer(observation)
        # RobotWinInference returns a CPU NumPy array after .cpu(), so this
        # includes GPU completion/transfer, not only CUDA launch overhead.
        return result, started, time.monotonic()

    def _metadata(self, observation, source):
        self._chunk_id += 1
        now = time.monotonic()
        timing = observation.get("_runtime_timing", {})
        if self.time_alignment:
            stamp = timing.get("observation_mono_s")
            if stamp is None or not math.isfinite(float(stamp)) or float(stamp) > now + 1e-6:
                raise ValueError("time alignment requires a finite, nonfuture observation_mono_s")
        return {"chunk_id": self._chunk_id, "source": source, "submitted_mono_s": now,
                "observation_mono_s": timing.get("observation_mono_s", now),
                "camera_mono_s": timing.get("camera_mono_s", now)}

    def _record_admission(self, meta, chunk, metrics):
        if meta is None:
            return
        now = time.monotonic()
        meta = dict(meta)
        meta.update({"admitted_mono_s": now,
                     "observation_age_s": now - meta["observation_mono_s"],
                     "camera_age_s": now - meta["camera_mono_s"],
                     "ready_to_admit_s": now - meta["completed_mono_s"],
                     "worker_wait_s": meta["infer_started_mono_s"] - meta["submitted_mono_s"],
                     "infer_s": meta["completed_mono_s"] - meta["infer_started_mono_s"]})
        anchor = self._last_emitted_joint
        meta["emitted_boundary_rad"] = float(np.max(np.abs(chunk[0, :6] - anchor))) if anchor is not None else None
        meta["chunk_update_mode"] = self.chunk_update_mode
        meta["planned_actions"] = len(chunk)
        meta.update(metrics)
        self.current_meta = meta
        self._current_selected = 0
        self._first_selected_mono = None
        self._last_selected_mono = None
        if self.diagnostics is not None:
            self.diagnostics.record("chunks", meta)

    def _finish_current_chunk(self, reason: str, discarded: int) -> None:
        if self.current_meta is not None and self.diagnostics is not None:
            self.diagnostics.record("executions", {
                "chunk_id": self.current_meta["chunk_id"], "source": self.current_meta["source"],
                "chunk_update_mode": self.chunk_update_mode, "end_reason": reason,
                "planned_actions": self.horizon, "actions_selected": self._current_selected,
                "discarded_actions": discarded,
                "first_selected_mono_s": self._first_selected_mono,
                "last_selected_mono_s": self._last_selected_mono,
                "ended_mono_s": time.monotonic(),
            })
        self.current_meta = None

    def _infer(self, observation: dict[str, Any]) -> np.ndarray:
        # static_check enforces state_indices=[0], so every chunk needs only
        # this observation and has no cross-thread history dependency.
        self.agent.reset()
        self.agent.processor.update_state_buffer(observation)
        return self.agent._predict_chunk(observation, self.instruction)

    def _validate_actions(self, chunk: np.ndarray, expected_rows: int) -> np.ndarray:
        value = np.asarray(chunk, dtype=np.float32)
        expected = (expected_rows, 7)
        if value.shape != expected or not np.isfinite(value).all():
            raise RuntimeError(
                f"invalid action block: expected {expected}, got {value.shape}"
            )
        # Reject a bad future action before admitting any part of the block.
        for action in value:
            self.controller.validate_model_action(action)
        return value

    def _validate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        value = np.asarray(chunk, dtype=np.float32)
        expected = (int(self.agent.config.common.action_chunk_size), 7)
        if value.shape != expected or not np.isfinite(value).all():
            raise RuntimeError(
                f"invalid model chunk: expected {expected}, got {value.shape}"
            )
        # Alignment validates the actually selected window at admission,
        # including newly used tail actions. Legacy mode still uses the prefix.
        if not self.time_alignment:
            for action in value[: self.horizon]:
                self.controller.validate_model_action(action)
        return value

    def _validate_horizon(self, actions: np.ndarray) -> np.ndarray:
        return self._validate_actions(actions, self.horizon)

    def preview(self, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
        self.wait_idle()
        started = time.perf_counter()
        meta = self._metadata(observation, "preview")
        result, infer_started, completed = self._timed_infer(observation)
        meta.update(infer_started_mono_s=infer_started, completed_mono_s=completed)
        self.preview_meta = meta
        raw_chunk = self._validate_chunk(result)
        if self.time_alignment:
            self._validate_horizon(raw_chunk[: self.horizon])
        self.smoother.reset(self.controller.joint_target_snapshot())
        chunk = self._validate_horizon(
            self.smoother.smooth(raw_chunk[: self.horizon])
        )
        line = self.smoother.metrics_line("PREVIEW")
        self.preview_metrics = dict(self.smoother.last_metrics)
        self.preview_metrics.setdefault("raw_boundary_rad", float(np.max(np.abs(
            raw_chunk[0, :6] - self.controller.joint_target_snapshot()))))
        if line:
            print(line)
        return chunk, time.perf_counter() - started

    def start(self, preview_chunk: np.ndarray, observation: dict[str, Any] | None = None) -> None:
        if self.time_alignment and observation is None:
            raise ValueError("aligned start needs a fresh observation after the countdown")
        self.stop()
        if self.time_alignment:
            # The inspected preview is usually several seconds old. Refresh
            # while holding, without concurrently accessing the same agent.
            self.wait_idle()
            self.smoother.reset(self.controller.joint_target_snapshot())
            self._last_emitted_joint = self.controller.joint_target_snapshot().copy()
            meta = self._metadata(observation, "startup_refresh")
            result, started, completed = self._timed_infer(observation)
            meta.update(infer_started_mono_s=started, completed_mono_s=completed)
            self.pending = self._validate_chunk(result).copy()
            self._pending_meta = meta
            if not self._admit_pending():
                raise RuntimeError("fresh startup prediction expired; retry deployment with a fresh preview")
            return
        chunk = self._validate_horizon(preview_chunk)
        self.smoother.reset(self.controller.joint_target_snapshot())
        self.actions.extend(chunk.copy())
        self._last_emitted_joint = self.controller.joint_target_snapshot().copy()
        self._record_admission(self.preview_meta, chunk, self.preview_metrics)

    def _submit(self, observation: dict[str, Any]) -> None:
        if self.future is not None or self.pending is not None:
            return
        generation = self.generation
        self.future_generation = generation
        self._future_meta = self._metadata(observation, "background")
        self._future_meta["queue_remaining_at_submit"] = len(self.actions)
        self.future = self.executor.submit(self._timed_infer, observation)
        self._alignment_refresh_needed = False

    def _poll(self) -> None:
        if self.future is None or not self.future.done():
            return
        future = self.future
        generation = self.future_generation
        self.future = None
        self.future_generation = None
        result, infer_started, completed = future.result()
        meta, self._future_meta = self._future_meta, None
        if generation != self.generation:
            return
        chunk = self._validate_chunk(result)
        self.pending = (chunk if self.time_alignment else chunk[: self.horizon]).copy()
        meta.update(infer_started_mono_s=infer_started, completed_mono_s=completed,
                    polled_mono_s=time.monotonic())
        if self.diagnostics is not None:
            self.diagnostics.record("inferences", {
                **meta, "infer_s": completed - infer_started,
                "worker_wait_s": infer_started - meta["submitted_mono_s"],
                "poll_delay_s": meta["polled_mono_s"] - completed,
            })
        self._pending_meta = meta

    def _aligned_start(self, now: float) -> tuple[int, float]:
        stamp = float(self._pending_meta["observation_mono_s"])
        age = now - stamp
        if not math.isfinite(age) or age < -1e-6:
            raise ValueError("invalid prediction time origin for alignment")
        age = max(0.0, age)
        # Latest sample at/before now. The small epsilon handles floating
        # roundoff exactly at a sample boundary, not a latency fudge factor.
        return int(math.floor(age * self.action_rate + 1e-9)), age

    def _reject_aligned_pending(self, reason: str, start: int, age: float, attempts: int) -> bool:
        if self.diagnostics is not None:
            self.diagnostics.record("alignment_rejections", {
                "chunk_id": self._pending_meta["chunk_id"], "source": self._pending_meta["source"],
                "reason": reason, "rejected_mono_s": time.monotonic(),
                "observation_age_s": age, "aligned_start_index": start,
                "prediction_steps": len(self.pending), "required_horizon": self.horizon,
                "conditioning_attempts": attempts,
            })
        self.pending = None
        self._pending_meta = None
        self._alignment_refresh_needed = True
        return False

    def _admit_pending(self) -> bool:
        if self.pending is None:
            return False
        discarded = len(self.actions)
        if discarded and self.chunk_update_mode != "latest_ready":
            raise RuntimeError("drain_queue cannot replace an unconsumed action queue")
        previous_id = self.current_meta["chunk_id"] if self.current_meta else None
        previous_selected = self._current_selected
        conditioning_started = time.monotonic()
        alignment_metrics = {}
        for attempt in range(2 if self.time_alignment else 1):
            start, age = self._aligned_start(time.monotonic()) if self.time_alignment else (0, 0.0)
            if start + self.horizon > len(self.pending):
                if not self.time_alignment:
                    raise RuntimeError("pending prediction is shorter than the execution horizon")
                return self._reject_aligned_pending("insufficient_tail", start, age, attempt)
            raw_window = self.pending[start : start + self.horizon]
            if self.time_alignment:
                self._validate_horizon(raw_window)
            raw_boundary = float(np.max(np.abs(raw_window[0, :6] - self._last_emitted_joint)))
            chunk = self._validate_horizon(self.smoother.smooth(raw_window))
            if not self.time_alignment:
                break
            after_start, after_age = self._aligned_start(time.monotonic())
            if after_start + self.horizon > len(self.pending):
                return self._reject_aligned_pending("insufficient_tail", after_start, after_age, attempt + 1)
            if after_start != start:
                if attempt == 1:
                    return self._reject_aligned_pending("conditioning_overran_step", after_start, after_age, attempt + 1)
                continue
            command_snapshot = getattr(self.controller, "joint_command_snapshot", None)
            command_position = command_snapshot() if command_snapshot is not None else None
            skipped_gripper = self.pending[:start, 6]
            alignment_metrics = {
                "aligned_start_index": start, "aligned_end_index_exclusive": start + self.horizon,
                "alignment_reference_mono_s": float(self._pending_meta["observation_mono_s"]),
                "alignment_evaluated_age_s": after_age,
                "alignment_residual_age_s": after_age - start / self.action_rate,
                "alignment_conditioning_attempts": attempt + 1,
                "aligned_start_gripper_norm": float(raw_window[0, 6]),
                "skipped_gripper_min_norm": float(skipped_gripper.min()) if start else None,
                "skipped_gripper_max_norm": float(skipped_gripper.max()) if start else None,
                "aligned_raw_start_to_command_max_rad": float(np.max(np.abs(raw_window[0, :6] - command_position))) if command_position is not None else None,
                "aligned_start_to_command_max_rad": float(np.max(np.abs(chunk[0, :6] - command_position))) if command_position is not None else None,
            }
            break
        conditioning_s = time.monotonic() - conditioning_started
        line = self.smoother.metrics_line("CHUNK")
        if line:
            print(line)
        metrics = {"raw_boundary_rad": raw_boundary, **self.smoother.last_metrics,
                   "discarded_queue_actions": discarded, "previous_chunk_id": previous_id,
                   "previous_chunk_actions_selected": previous_selected,
                   "conditioning_s": conditioning_s, **alignment_metrics}
        # Fit and validate against the latest history BEFORE changing the queue.
        # Only targets actually selected by next_action enter that history.
        self._finish_current_chunk("replaced" if discarded else "exhausted", discarded)
        self.actions.clear()
        self._record_admission(self._pending_meta, chunk, metrics)
        self.actions.extend(chunk)
        self.pending = None
        self._pending_meta = None
        return True

    def next_action(self, observation: dict[str, Any]) -> np.ndarray | None:
        self._poll()
        if self.pending is not None and (not self.actions or self.chunk_update_mode == "latest_ready"):
            self._admit_pending()
        if not self.actions:
            self._submit(observation)
            return None
        action = self.actions.popleft()
        self.smoother.observe(action)
        self._last_emitted_joint = action[:6].copy()
        self._current_selected += 1
        self._last_selected_mono = time.monotonic()
        if self._first_selected_mono is None:
            self._first_selected_mono = self._last_selected_mono
        if (
            (len(self.actions) <= self.prefetch_threshold or self._alignment_refresh_needed)
            and self.pending is None
            and self.future is None
        ):
            self._submit(observation)
        return action

    def stop(self) -> None:
        self._finish_current_chunk("stopped", len(self.actions))
        self.generation += 1
        self.actions.clear()
        self.pending = None
        self._pending_meta = None
        self._alignment_refresh_needed = False
        self.current_meta = None
        if self.future is not None and self.future.cancel():
            self.future = None
            self.future_generation = None
            self._future_meta = None

    def wait_idle(self) -> None:
        if self.future is not None:
            # Preview is only allowed while the arm is holding. A running
            # inference is allowed to finish, then its stale result is dropped.
            self.future.result()
            self.future = None
            self.future_generation = None
            self.pending = None
            self._future_meta = None
            self._pending_meta = None

    def close(self) -> None:
        self.stop()
        self.executor.shutdown(wait=True, cancel_futures=True)


def build_agent(config: dict):
    import torch
    from robotwin_infer import RobotWinInference

    dtype_name = str(config["model"].get("dtype", "bfloat16"))
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    device = str(config["model"].get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but no CUDA GPU is visible")
    return RobotWinInference(
        config_path=config["model"]["config_path"],
        checkpoint_path=config["model"]["checkpoint_path"],
        norm_stats_path=config["model"]["norm_stats_path"],
        device=device,
        dtype=dtypes[dtype_name],
        task_name=str(config["model"]["task_name"]),
    )


def print_static_report(config: dict, checked: dict) -> None:
    meta = checked["stats_meta"]
    print("STATIC CHECK PASSED")
    print(f"  checkpoint: {config['model']['checkpoint_path']}")
    print(f"  stats files: {meta.get('valid_files', 'unknown')}/{meta.get('total_files', 'unknown')}")
    print(f"  model schema: action=7, state=8, camera={config['hardware']['camera_name']}")
    image_size = checked["train_config"]["dataset"]["image_size"]
    print(f"  inference resize: {image_size[0]}x{image_size[1]} (width x height)")
    print("  deployed action: action[0:6] joints + action[6] gripper")
    horizon = int(checked["train_config"]["common"]["action_execution_horizon"])
    threshold = int(config["runtime"].get("prefetch_threshold_actions", horizon // 2))
    rate = float(config["runtime"].get("action_rate_hz", 30.0))
    print(f"  async prefetch: horizon={horizon}, threshold={threshold}, nominal budget={threshold / rate:.3f}s")
    print(f"  chunk update mode: {_chunk_update_mode(config)}")
    aligned = _time_alignment_enabled(config, int(checked["train_config"]["common"]["action_chunk_size"]), horizon)
    print(f"  admission time alignment: {'enabled (fresh startup, floor(age * Hz))' if aligned else 'disabled'}")
    enabled = bool(config["runtime"].get("diagnostics", {}).get("enabled", False))
    print(f"  runtime diagnostics: {'enabled' if enabled else 'disabled'}")
    bspline = checked["joint_bspline"]
    if bspline["enabled"]:
        print(
            "  L1-L6 cubic B-spline: "
            f"method={bspline['method']}, "
            f"history={bspline['history_actions']}, "
            f"controls={bspline['control_points']}, "
            f"history_weight={bspline['history_weight']:.2f}"
        )
        if bspline["method"] == "prefix_refit":
            print(f"  local prefix: free_controls={bspline['refit_free_control_points']}, regularization={bspline['refit_regularization']}")
    else:
        print("  L1-L6 cubic B-spline: disabled")
    print(
        "  trained left action range: "
        f"min={checked['action_min'][:7].round(4).tolist()}, "
        f"max={checked['action_max'][:7].round(4).tolist()}"
    )


def run_execute(config: dict, checked: dict) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("--execute requires an interactive terminal")

    print("Loading VLA model before enabling any motor...")
    agent = build_agent(config)
    collector = checked["collector_config"]
    cameras = build_cameras(collector.get("cameras", []), dry_run=False)
    camera_name = config["hardware"]["camera_name"]
    camera = cameras[camera_name]
    controller = EDULITEVLAController(config, checked)
    diagnostics = RuntimeDiagnostics(config)
    controller.diagnostics = diagnostics
    visualizer = InferenceVisualizer(config, controller)
    instruction = str(config["model"]["instruction"])
    policy = AsyncActionChunks(agent, controller, instruction, config)
    camera_max_age = float(config["runtime"].get("camera_max_age_s", 0.25))
    action_period = 1.0 / float(config["runtime"].get("action_rate_hz", 30.0))
    max_episode = float(config["runtime"].get("max_episode_s", 40.0))
    start_delay = float(config["runtime"].get("start_delay_s", 3.0))
    require_preview = bool(config["runtime"].get("require_preview_before_execute", True))

    executing = False
    preview_ok = False
    episode_start = 0.0
    next_action = 0.0
    preview_chunk: np.ndarray | None = None
    stop_requested = threading.Event()
    last_tick = None
    last_action = None
    exit_reason = "program_exit"

    def finalize_episode(reason):
        diagnostics.finish(reason)
        visualizer.finalize(reason)

    def emergency() -> None:
        nonlocal executing
        executing = False
        controller.emergency_stop()
        print("\nEMERGENCY STOP: task-arm motors disabled", flush=True)

    def signal_stop(_signum, _frame) -> None:
        emergency()
        stop_requested.set()

    signal.signal(signal.SIGINT, signal_stop)
    signal.signal(signal.SIGTERM, signal_stop)

    key_reader: KeyReader | None = None
    try:
        camera.start()
        # Wait for a real frame before opening CAN.
        deadline = time.monotonic() + 3.0
        while True:
            try:
                frame = camera.latest(camera_max_age)
                print(f"Camera ready: RGB shape={tuple(frame.rgb.shape)}")
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        controller.connect()  # Connection alone never enables motors.
        print(f"Connected to task arm: {controller.arm_cfg['can_name']} (motors disabled)")
        print("Controls: a=enable+home, p=preview one model chunk, r=execute, s=hold, h=home")
        print("          e=E-STOP (independent key thread), q=quit")
        print("Keep the physical emergency-stop reachable at all times.")

        with raw_terminal(True):
            key_reader = KeyReader(emergency)
            key_reader.start()
            while not stop_requested.is_set():
                if controller.armed:
                    controller.check_health()
                key = key_reader.get()
                if key == "a":
                    if controller.armed:
                        print("Already armed")
                    else:
                        controller.arm_and_home()
                        agent.reset()
                        preview_ok = False
                        print("ARMED/HOME READY; press p to preview model output")
                elif key == "p":
                    if not controller.armed or executing:
                        print("Preview requires an armed, non-executing arm")
                    else:
                        frame = camera.latest(camera_max_age)
                        model_rgb = _model_camera_image(frame.rgb)
                        observation = controller.observation(model_rgb, frame.monotonic_time)
                        chunk, inference_s = policy.preview(observation)
                        warning_count = 0
                        for action in chunk:
                            _, warnings = controller.validate_model_action(action)
                            warning_count += len(warnings)
                        print(
                            "PREVIEW PASSED: "
                            f"action min={chunk.min(0).round(3).tolist()}, "
                            f"action max={chunk.max(0).round(3).tolist()}, "
                            f"clip_warnings={warning_count}, "
                            f"inference={inference_s:.3f}s"
                        )
                        preview_chunk = chunk.copy()
                        preview_ok = True
                        # Save the same resized RGB array passed to DINOv3.
                        visualizer.save_preview(
                            agent.processor.prepare_rgb(model_rgb), chunk
                        )
                elif key == "r":
                    if not controller.armed:
                        print("Press a before execution")
                    elif executing:
                        print("Already executing")
                    elif require_preview and not preview_ok:
                        print("Press p and pass model preview before execution")
                    else:
                        print(f"Execution starts in {start_delay:.1f}s; press e to abort")
                        end = time.monotonic() + start_delay
                        while time.monotonic() < end:
                            if not controller.armed:
                                break
                            time.sleep(0.02)
                        if controller.armed:
                            if preview_chunk is None:
                                raise RuntimeError("preview chunk is unexpectedly missing")
                            # A preview is single-use. After any motion, a new
                            # execution requires a new observation and preview.
                            preview_ok = False
                            start_frame = camera.latest(camera_max_age)
                            run_dir = visualizer.start_episode(start_frame.rgb)
                            diagnostics.start(run_dir)
                            if policy.time_alignment:
                                policy.wait_idle()
                                fresh_frame = camera.latest(camera_max_age)
                                fresh_observation = controller.observation(
                                    _model_camera_image(fresh_frame.rgb), fresh_frame.monotonic_time
                                )
                                print("Refreshing startup prediction while holding (time alignment enabled)")
                                policy.start(preview_chunk, observation=fresh_observation)
                            else:
                                policy.start(preview_chunk)
                            preview_chunk = None
                            last_tick = None
                            last_action = None
                            executing = True
                            episode_start = time.monotonic()
                            next_action = episode_start
                            print(f"VLA EXECUTION ACTIVE: {instruction}")
                elif key == "s":
                    executing = False
                    policy.stop()
                    if controller.armed:
                        controller.hold_current()
                        finalize_episode("operator_stop")
                        print("EXECUTION STOPPED: holding current pose")
                elif key == "h":
                    if executing:
                        print("Press s before homing")
                    elif not controller.armed:
                        print("Press a before homing")
                    else:
                        policy.stop()
                        finalize_episode("home_requested")
                        controller.return_home()
                        preview_ok = False
                        preview_chunk = None
                elif key == "e":
                    executing = False
                    policy.stop()
                    finalize_episode("emergency_stop")
                elif key == "q":
                    exit_reason = "quit"
                    break

                if executing and controller.armed:
                    now = time.monotonic()
                    if now - episode_start >= max_episode:
                        executing = False
                        policy.stop()
                        controller.hold_current()
                        finalize_episode("max_episode_time")
                        print(f"MAX EPISODE TIME {max_episode:.1f}s reached; holding pose")
                    elif now >= next_action:
                        tick_started = time.monotonic()
                        tick_lateness = max(0.0, tick_started - next_action)
                        frame = camera.latest(camera_max_age)
                        model_rgb = _model_camera_image(frame.rgb)
                        observation = controller.observation(model_rgb, frame.monotonic_time)
                        action = policy.next_action(observation)
                        warnings = []
                        command_time = None
                        if action is not None:
                            warnings = controller.set_model_action(action)
                            command_time = time.monotonic()
                            if warnings:
                                print("ACTION SAFETY: " + "; ".join(sorted(set(warnings))))
                        visualizer.record(
                            frame.rgb,
                            action,
                            controller.telemetry(),
                            np.asarray(observation["endpose"]["left_endpose"][:3]),
                        )
                        next_action += action_period
                        tick_done = time.monotonic()
                        schedule_reset = tick_done - next_action > action_period
                        if schedule_reset:
                            # Never catch up by bursting old actions.
                            next_action = time.monotonic() + action_period
                        meta = policy.current_meta or {}
                        aligned_index = meta.get("aligned_start_index")
                        prediction_index = (aligned_index + policy._current_selected - 1
                                            if aligned_index is not None and action is not None else None)
                        prediction_time = (meta["alignment_reference_mono_s"] + prediction_index / policy.action_rate
                                           if prediction_index is not None else None)
                        diagnostics.record("ticks", {
                            "tick_mono_s": tick_started,
                            "tick_interval_s": None if last_tick is None else tick_started - last_tick,
                            "action_interval_s": None if command_time is None or last_action is None else command_time - last_action,
                            "command_mono_s": command_time,
                            "work_s": tick_done - tick_started,
                            "lateness_s": tick_lateness, "schedule_reset": int(schedule_reset),
                            "action_available": int(action is not None),
                            "queue_remaining": len(policy.actions),
                            "pending_ready": int(policy.pending is not None),
                            "inference_in_flight": int(policy.future is not None),
                            "chunk_id": meta.get("chunk_id"),
                            "chunk_actions_selected": policy._current_selected,
                            "prediction_action_index": prediction_index,
                            "prediction_action_time_mono_s": prediction_time,
                            "alignment_phase_error_s": command_time - prediction_time if prediction_time is not None else None,
                            "camera_age_s": observation["_runtime_timing"]["observation_mono_s"] - frame.monotonic_time,
                            "observation_age_s": None if command_time is None or not meta else command_time - meta["observation_mono_s"],
                            "action_camera_age_s": None if command_time is None or not meta else command_time - meta["camera_mono_s"],
                            "safety_warning_count": len(warnings),
                        })
                        last_tick = tick_started
                        if command_time is not None:
                            last_action = command_time
                time.sleep(0.002 if executing else 0.01)
        return 0
    except BaseException:
        exit_reason = "exception"
        raise
    finally:
        if key_reader is not None:
            key_reader.close()
        # Hardware cleanup precedes disk/plot work, including on exceptions.
        try:
            controller.close()
        finally:
            try:
                camera.stop()
            finally:
                try:
                    policy.close()
                finally:
                    finalize_episode(exit_reason)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.real_single.yaml"))
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate paths/schema/stats only; never load GPU, camera, or CAN",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="allow the interactive program to enable and move the physical task arm",
    )
    parser.add_argument(
        "--task-name",
        help="override model.task_name (selects the matching task_cond.npy)",
    )
    parser.add_argument(
        "--instruction",
        help="override the human-readable task instruction shown in runtime logs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_only and args.execute:
        raise ValueError("choose either --check-only or --execute")
    if not args.check_only and not args.execute:
        raise ValueError("no operation selected; use --check-only first, then --execute")
    config, _ = load_deployment_config(args.config)
    if args.task_name:
        config["model"]["task_name"] = args.task_name
    if args.instruction:
        config["model"]["instruction"] = args.instruction
    checked = static_check(config)
    print_static_report(config, checked)
    if args.check_only:
        return 0
    return run_execute(config, checked)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1)
