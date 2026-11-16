"""Buffered deployment measurements; no camera, CAN, or model dependencies."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import threading
import time

import numpy as np


class RuntimeDiagnostics:
    def __init__(self, config):
        self.config = config
        self.enabled = bool(config.get("runtime", {}).get("diagnostics", {}).get("enabled", False))
        self.lock = threading.Lock()
        self.directory = None
        self.started = 0.0
        self.interface = {}
        self.ticks, self.chunks, self.controls, self.inferences = [], [], [], []
        self.executions = []
        self.alignment_rejections = []

    def start(self, directory=None):
        if not self.enabled:
            return
        if self.directory is not None:
            self.finish("restarted")
        if directory is None:
            root = Path(self.config.get("visualization", {}).get("output_root", "./inference_outputs"))
            root.mkdir(parents=True, exist_ok=True)
            import tempfile
            directory = tempfile.mkdtemp(prefix="diagnostics_", dir=root)
        with self.lock:
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
            self.started = time.monotonic()
            self.ticks, self.chunks, self.controls, self.inferences = [], [], [], []
            self.executions = []
            self.alignment_rejections = []

    def record(self, kind, row):
        if not self.enabled:
            return
        with self.lock:
            if self.directory is not None:
                getattr(self, kind).append(dict(row))

    @staticmethod
    def _stats(values):
        values = np.asarray([v for v in values if v is not None], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return {"count": 0, "p50": None, "p95": None, "max": None}
        return {"count": len(values), "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)), "max": float(values.max())}

    @staticmethod
    def _csv(path, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def finish(self, reason):
        with self.lock:
            if self.directory is None:
                return
            directory, started = self.directory, self.started
            self.directory = None
            ticks, chunks, controls, inferences = self.ticks, self.chunks, self.controls, self.inferences
            executions, self.executions = self.executions, []
            alignment_rejections, self.alignment_rejections = self.alignment_rejections, []
            self.ticks, self.chunks, self.controls, self.inferences = [], [], [], []
        ended = time.monotonic()
        duration = ended - started
        # These are observed waiting intervals: empty queue at one scheduled
        # poll until recovery (or episode end), not GPU idle time.
        gaps, since = [], None
        for row in ticks:
            if not row["action_available"] and since is None:
                since = row["tick_mono_s"]
            elif row["action_available"] and since is not None:
                gaps.append(row["tick_mono_s"] - since)
                since = None
        if since is not None:
            gaps.append(ended - since)
        background = inferences
        summary = {
            "stop_reason": reason, "duration_s": duration,
            "background_inference_s": self._stats([c["infer_s"] for c in background]),
            "worker_queue_wait_s": self._stats([c["worker_wait_s"] for c in background]),
            "result_poll_delay_s": self._stats([c.get("poll_delay_s") for c in background]),
            "chunk_first_observation_age_s": self._stats([c["observation_age_s"] for c in chunks]),
            "background_chunk_first_observation_age_s": self._stats([c["observation_age_s"] for c in chunks if c["source"] == "background"]),
            "preview_chunk_first_observation_age_s": self._stats([c["observation_age_s"] for c in chunks if c["source"] == "preview"]),
            "startup_refresh_first_observation_age_s": self._stats([c["observation_age_s"] for c in chunks if c["source"] == "startup_refresh"]),
            "startup_refresh_inference_s": self._stats([c["infer_s"] for c in chunks if c["source"] == "startup_refresh"]),
            "action_observation_age_s": self._stats([r.get("observation_age_s") for r in ticks if r["action_available"]]),
            "action_tick_interval_s": self._stats([r["tick_interval_s"] for r in ticks]),
            "action_update_interval_s": self._stats([r.get("action_interval_s") for r in ticks]),
            "control_interval_s": self._stats([r["interval_s"] for r in controls]),
            "action_tick_work_s": self._stats([r["work_s"] for r in ticks]),
            "following_error_rad": self._stats([r["following_error_rad"] for r in controls]),
            "model_command_timeout_cycles": sum(r["model_command_timed_out"] for r in controls),
            "queue_wait_events": len(gaps), "queue_wait_total_s": float(sum(gaps)),
            "queue_wait_duration_s": self._stats(gaps),
            "schedule_resets": sum(r["schedule_reset"] for r in ticks),
            "model_updates": sum(r["action_available"] for r in ticks),
            "safety_warning_actions": sum(r["safety_warning_count"] > 0 for r in ticks),
            "bspline_fallback_chunks": sum(c.get("fallback", 0) for c in chunks),
            "chunk_replacements": sum(e["end_reason"] == "replaced" for e in executions),
            "replacement_discarded_actions": sum(e["discarded_actions"] for e in executions if e["end_reason"] == "replaced"),
            "stopped_discarded_actions": sum(e["discarded_actions"] for e in executions if e["end_reason"] == "stopped"),
            "background_chunk_actions_selected": self._stats([e["actions_selected"] for e in executions if e["source"] == "background"]),
            "chunk_conditioning_s": self._stats([c.get("conditioning_s") for c in chunks]),
            "bspline_fit_s": self._stats([c.get("fit_compute_s") for c in chunks]),
            "aligned_start_index": self._stats([c.get("aligned_start_index") for c in chunks]),
            "alignment_residual_age_s": self._stats([c.get("alignment_residual_age_s") for c in chunks]),
            "alignment_phase_error_s": self._stats([r.get("alignment_phase_error_s") for r in ticks]),
            "aligned_start_to_command_max_rad": self._stats([c.get("aligned_start_to_command_max_rad") for c in chunks]),
            "aligned_raw_start_to_command_max_rad": self._stats([c.get("aligned_raw_start_to_command_max_rad") for c in chunks]),
            "time_alignment_rejected_chunks": len(alignment_rejections),
            "settings": {"runtime": self.config["runtime"], "model": self.config["model"],
                         "safety": self.config["safety"], "model_interface": self.interface},
        }
        self._csv(directory / "runtime_ticks.csv", ticks)
        self._csv(directory / "inference_chunks.csv", chunks)
        self._csv(directory / "inference_completed.csv", inferences)
        self._csv(directory / "control_timing.csv", controls)
        self._csv(directory / "chunk_executions.csv", executions)
        self._csv(directory / "alignment_rejections.csv", alignment_rejections)
        (directory / "runtime_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(f"RUNTIME DIAGNOSTICS SAVED: {directory}; waits={len(gaps)}, total={sum(gaps):.3f}s", flush=True)
