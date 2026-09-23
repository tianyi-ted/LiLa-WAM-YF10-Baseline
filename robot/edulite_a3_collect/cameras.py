"""Timestamped camera capture for the EDULITE-A3 dataset collector."""

from __future__ import annotations

import threading
import time
import subprocess
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import yaml


def load_camera_profile(config: dict, config_path) -> dict:
    """Resolve an optional independent camera-control file relative to YAML."""
    profile_path = config.get("camera_controls_file")
    if profile_path:
        path = Path(profile_path).expanduser()
        if not path.is_absolute():
            path = Path(config_path).resolve().parent / path
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict) or not isinstance(profile.get("cameras"), dict):
            raise ValueError(f"invalid camera controls profile: {path}")
        for camera in config.get("cameras", []):
            camera["v4l2_controls"] = profile["cameras"].get(camera["name"], {})
    return config


def apply_camera_controls(source, profile: dict) -> None:
    if not profile.get("enabled", False):
        return
    device = f"/dev/video{source}" if isinstance(source, int) else str(source)
    controls = profile.get("controls", {})
    if not isinstance(controls, dict) or not controls:
        raise ValueError("enabled camera controls profile must contain controls")
    # Preserve YAML order: disable automatic controls before setting manual ones.
    for name, value in controls.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"V4L2 control {name} must be an integer")
        def run(option):
            result = subprocess.run(
                ["v4l2-ctl", "--device", device, option],
                capture_output=True, text=True, timeout=5, check=True,
            )
            return result.stdout.strip()
        try:
            run(f"--set-ctrl={name}={value}")
            actual = run(f"--get-ctrl={name}")
            # Menu controls append labels, e.g. "auto_exposure: 1 (Manual Mode)".
            # Match the requested control explicitly, retaining numeric readback checks.
            match = re.fullmatch(
                rf"\s*{re.escape(name)}\s*:\s*([+-]?\d+)(?:\s+\([^\r\n]*\))?\s*",
                actual,
            )
            if match is None:
                raise ValueError(f"unrecognized readback for {name}: {actual!r}")
            if int(match.group(1)) != value:
                raise RuntimeError(f"requested {name}={value}, read back {actual}")
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise RuntimeError(f"camera {device}: cannot verify {name}={value}: {detail}") from exc
        print(f"CAMERA CONTROL VERIFIED {device}: {actual}", flush=True)


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    monotonic_time: float


class OpenCVCamera:
    """Continuously reads a V4L2/file camera and exposes the newest RGB frame."""

    def __init__(self, name: str, source: Any, width: int, height: int, fps: float,
                 controls: dict | None = None):
        self.name = name
        self.source = int(source) if isinstance(source, str) and source.isdigit() else source
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.controls = controls or {}
        self._capture: cv2.VideoCapture | None = None
        self._frame: CameraFrame | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        capture = cv2.VideoCapture(self.source)
        if not capture.isOpened():
            raise RuntimeError(f"camera {self.name!r} cannot open source {self.source!r}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            # Apply after format negotiation, which can reset device controls.
            apply_camera_controls(self.source, self.controls)
        except Exception:
            capture.release()
            raise
        self._capture = capture
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"camera-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._capture is not None
        while self._running:
            ok, bgr = self._capture.read()
            if not ok:
                self._error = RuntimeError(f"camera {self.name!r} read failed")
                time.sleep(0.01)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = CameraFrame(np.ascontiguousarray(rgb), time.monotonic())
            with self._lock:
                self._frame = frame

    def latest(self, max_age_s: float) -> CameraFrame:
        if self._error is not None and self._frame is None:
            raise self._error
        with self._lock:
            frame = self._frame
        if frame is None:
            raise RuntimeError(f"camera {self.name!r} has not produced a frame")
        age = time.monotonic() - frame.monotonic_time
        if age > max_age_s:
            raise RuntimeError(
                f"camera {self.name!r} frame is stale: {age:.3f}s > {max_age_s:.3f}s"
            )
        return CameraFrame(frame.rgb.copy(), frame.monotonic_time)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                # VideoCapture.read() still owns the V4L2 file descriptor.
                # Releasing it here can cause ioctl(VIDIOC_QBUF): Bad file
                # descriptor as the blocked read queues its buffer.
                raise RuntimeError(
                    f"camera {self.name!r} read thread did not stop within 5 s; "
                    "capture was not released while read() is active"
                )
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class SyntheticCamera:
    """Deterministic RGB source used by dry-run and offline tests."""

    def __init__(self, name: str, width: int, height: int, fps: float):
        self.name = name
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._start = time.monotonic()

    def start(self) -> None:
        self._start = time.monotonic()

    def latest(self, max_age_s: float) -> CameraFrame:
        del max_age_s
        t = time.monotonic() - self._start
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[..., 0] = int((40.0 * t) % 255)
        image[..., 1] = np.linspace(0, 255, self.width, dtype=np.uint8)[None, :]
        image[..., 2] = np.linspace(0, 255, self.height, dtype=np.uint8)[:, None]
        cv2.putText(
            image,
            f"dry-run {self.name} {t:5.2f}s",
            (12, min(40, self.height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return CameraFrame(image, time.monotonic())

    def stop(self) -> None:
        return


def build_cameras(configs: list[dict], dry_run: bool):
    cameras = {}
    for cfg in configs:
        if not cfg.get("enabled", True):
            continue
        name = str(cfg["name"])
        if name in cameras:
            raise ValueError(f"duplicate camera name: {name}")
        cls = SyntheticCamera if dry_run else OpenCVCamera
        kwargs = {
            "name": name,
            "width": int(cfg.get("width", 640)),
            "height": int(cfg.get("height", 480)),
            "fps": float(cfg.get("fps", 30.0)),
        }
        if not dry_run:
            kwargs["source"] = cfg.get("source", 0)
            kwargs["controls"] = cfg.get("v4l2_controls", {})
        cameras[name] = cls(**kwargs)
    if not cameras:
        raise ValueError("at least one enabled camera is required")
    return cameras
