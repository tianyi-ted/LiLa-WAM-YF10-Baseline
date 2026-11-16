"""Offline regression tests: fake model/controller, no GPU or robot interface."""
import json
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from real_vla import AsyncActionChunks, CubicBSplineJointSmoother, _joint_bspline_settings, _chunk_update_mode
from runtime_diagnostics import RuntimeDiagnostics
from trajectory_refit import fit_local_prefix


def config():
    return {"runtime": {"diagnostics": {"enabled": True}, "prefetch_threshold_actions": 2,
                        "joint_bspline": {"enabled": False}},
            "model": {}, "safety": {}}


class ManualExecutor:
    def submit(self, *args):
        return Future()

    def shutdown(self, **kwargs):
        pass


class FakeController:
    lower = np.full(6, -1000.0)
    upper = np.full(6, 1000.0)

    def joint_target_snapshot(self):
        return np.zeros(6)

    def validate_model_action(self, action):
        return action, []


class FakeAgent:
    action_execution_horizon = 4
    config = SimpleNamespace(common=SimpleNamespace(action_chunk_size=8))
    processor = SimpleNamespace(update_state_buffer=lambda obs: None)

    def reset(self):
        pass

    def _predict_chunk(self, *args):
        return np.repeat(np.arange(8, dtype=np.float32)[:, None], 7, axis=1)


class DiagnosticsTests(unittest.TestCase):
    def make_manual_policy(self, cfg, controller=None):
        policy = AsyncActionChunks(FakeAgent(), controller or FakeController(), "", cfg)
        policy.executor.shutdown()
        policy.executor = ManualExecutor()
        self.addCleanup(policy.close)
        return policy

    def test_wait_duration_and_episode_reset(self):
        d = RuntimeDiagnostics(config())
        with TemporaryDirectory() as directory, patch("runtime_diagnostics.time.monotonic", return_value=200.0):
            d.start(directory)
            for stamp, available in [(200.1, 0), (200.3, 1)]:
                d.record("ticks", {"tick_mono_s": stamp, "action_available": available,
                                  "tick_interval_s": None, "work_s": .002,
                                  "schedule_reset": 0, "safety_warning_count": 0})
            d.record("inferences", {"infer_s": .12, "worker_wait_s": .001})
            d.finish("operator_stop")
            summary = json.loads((Path(directory) / "runtime_summary.json").read_text())
            self.assertEqual(summary["queue_wait_events"], 1)
            self.assertAlmostEqual(summary["queue_wait_total_s"], .2)
            self.assertAlmostEqual(summary["background_inference_s"]["p95"], .12)
            self.assertTrue((Path(directory) / "runtime_ticks.csv").exists())
            # Recording after finish must not leak into a later episode.
            d.record("controls", {"interval_s": 99})
            d.start(Path(directory) / "second")
            d.finish("empty")
            self.assertEqual(json.loads((Path(directory) / "second/runtime_summary.json").read_text())["model_updates"], 0)

    def test_async_order_starvation_and_inference_measurement(self):
        cfg = config()
        d = RuntimeDiagnostics(cfg)
        controller = FakeController()
        controller.diagnostics = d
        policy = AsyncActionChunks(FakeAgent(), controller, "", cfg)
        policy.executor.shutdown()
        policy.executor = ManualExecutor()
        with TemporaryDirectory() as directory:
            d.start(directory)
            preview, _ = policy.preview({})
            policy.start(preview)
            try:
                obs = {"_runtime_timing": {"observation_mono_s": time.monotonic(),
                                            "camera_mono_s": time.monotonic() - .01}}
                actions = [policy.next_action(obs) for _ in range(4)]
                np.testing.assert_array_equal(np.stack(actions), preview)
                self.assertEqual(policy._future_meta["queue_remaining_at_submit"], 2)
                self.assertIsNone(policy.next_action(obs))
                submitted = policy._future_meta["submitted_mono_s"]
                predicted = FakeAgent()._predict_chunk() + 100
                policy.future.set_result((predicted, submitted + .001, submitted + .101))
                actual = policy.next_action(obs)
                np.testing.assert_array_equal(actual, predicted[0])
                self.assertEqual(len(d.inferences), 1)
                self.assertAlmostEqual(d.inferences[0]["infer_s"], .1)
                self.assertEqual(len(d.chunks), 2)
                self.assertEqual(d.chunks[0]["source"], "preview")
                self.assertEqual(d.chunks[1]["source"], "background")
                self.assertAlmostEqual(d.chunks[1]["raw_boundary_rad"], 97)
            finally:
                policy.close()
                d.finish("test")

    def test_stopped_generation_is_discarded(self):
        cfg = config()
        policy = AsyncActionChunks(FakeAgent(), FakeController(), "", cfg)
        policy.executor.shutdown()
        policy.executor = ManualExecutor()
        policy._submit({})
        future = policy.future
        future.set_running_or_notify_cancel()
        policy.stop()
        future.set_result((FakeAgent()._predict_chunk(), 1.0, 1.1))
        policy._poll()
        self.assertIsNone(policy.pending)
        self.assertIsNone(policy.current_meta)
        policy.close()

    def test_bspline_metrics_preserve_gripper(self):
        cfg = config()
        cfg["runtime"]["joint_bspline"] = {"enabled": True, "history_actions": 2,
                                            "control_points": 4, "history_weight": 4}
        policy = AsyncActionChunks(FakeAgent(), FakeController(), "", cfg)
        try:
            raw = FakeAgent()._predict_chunk()
            smoothed, _ = policy.preview({})
            np.testing.assert_array_equal(smoothed[:, 6], raw[:4, 6])
            self.assertIn("fallback", policy.preview_metrics)
            self.assertIn("raw_boundary_rad", policy.preview_metrics)
        finally:
            policy.close()

    def test_latest_ready_replaces_before_old_queue_drains(self):
        cfg = config()
        cfg["runtime"]["chunk_update_mode"] = "latest_ready"
        d = RuntimeDiagnostics(cfg)
        controller = FakeController()
        controller.diagnostics = d
        p = self.make_manual_policy(cfg, controller)
        with TemporaryDirectory() as directory:
            d.start(directory)
            preview, _ = p.preview({})
            p.start(preview)
            p.next_action({})
            p.next_action({})  # remaining=2, trigger one prefetch
            old_future = p.future
            predicted = FakeAgent()._predict_chunk() + 100
            old_future.set_result((predicted, time.monotonic(), time.monotonic()))
            output = p.next_action({})
            np.testing.assert_array_equal(output, predicted[0])
            self.assertEqual(len(p.actions), 3)
            self.assertIsNone(p.future)  # no immediate resubmission after refill
            self.assertEqual(d.chunks[-1]["discarded_queue_actions"], 2)
            self.assertEqual(d.chunks[-1]["previous_chunk_actions_selected"], 2)
            self.assertEqual(d.executions[0]["actions_selected"], 2)
            self.assertEqual(d.executions[0]["end_reason"], "replaced")
            # History contains only returned commands, never discarded old 2/3.
            np.testing.assert_array_equal(np.asarray(p.smoother.history)[-2:, 0], [1, 100])
            p.next_action({})
            self.assertIsNotNone(p.future)  # reaches threshold again, one future
            p.stop()
            d.finish("test")
            summary = json.loads((Path(directory) / "runtime_summary.json").read_text())
            self.assertEqual(summary["chunk_replacements"], 1)
            self.assertEqual(summary["replacement_discarded_actions"], 2)
            self.assertTrue((Path(directory) / "chunk_executions.csv").exists())

    def test_drain_queue_keeps_old_targets_when_result_is_early(self):
        p = self.make_manual_policy(config())
        preview, _ = p.preview({})
        p.start(preview)
        p.next_action({})
        p.next_action({})
        predicted = FakeAgent()._predict_chunk() + 100
        p.future.set_result((predicted, time.monotonic(), time.monotonic()))
        np.testing.assert_array_equal(p.next_action({}), preview[2])
        self.assertIsNotNone(p.pending)
        np.testing.assert_array_equal(p.next_action({}), preview[3])
        np.testing.assert_array_equal(p.next_action({}), predicted[0])

    def test_new_candidate_is_validated_before_replacing_queue(self):
        cfg = config()
        cfg["runtime"]["chunk_update_mode"] = "latest_ready"
        p = self.make_manual_policy(cfg)
        preview, _ = p.preview({})
        p.start(preview)
        p.next_action({})
        remaining = np.stack(p.actions).copy()
        p.pending = np.full((4, 7), np.nan)
        with self.assertRaises(RuntimeError):
            p.next_action({})
        np.testing.assert_array_equal(np.stack(p.actions), remaining)

    def test_prefix_refit_changes_only_first_three_controls(self):
        x = np.arange(16, dtype=float)
        future = np.repeat((.05 + .002 * x + .0001 * x**2)[:, None], 6, axis=1)
        history = np.repeat(np.array([-.002, 0])[:, None], 6, axis=1)
        fit = fit_local_prefix(future, history)
        self.assertLess(fit.history_rmse_after, fit.history_rmse_before)
        self.assertLess(abs(fit.future[0, 0]), abs(future[0, 0]))
        np.testing.assert_array_equal(fit.control_points[3:], fit.original_control_points[3:])
        # First 3 cubic basis functions have no support at/after sample 9.
        np.testing.assert_allclose(fit.future[9:], future[9:], atol=1e-12)

    def test_latest_prefix_fit_uses_history_at_admission(self):
        cfg = config()
        cfg["runtime"]["chunk_update_mode"] = "latest_ready"
        cfg["runtime"]["joint_bspline"] = {"enabled": True, "method": "prefix_refit",
            "control_points": 4, "history_actions": 2, "refit_free_control_points": 3}
        p = self.make_manual_policy(cfg)
        preview, _ = p.preview({})
        p.start(preview)
        p.next_action({})
        p.next_action({})
        submission_history = np.array(p.smoother.history)
        p.next_action({})  # another old action executes during inference
        admission_history = np.array(p.smoother.history)
        self.assertFalse(np.array_equal(submission_history, admission_history))
        p.future.set_result((FakeAgent()._predict_chunk() + .01, time.monotonic(), time.monotonic()))
        with patch("real_vla.fit_local_prefix", wraps=fit_local_prefix) as fitter:
            result = p.next_action({})
            np.testing.assert_array_equal(fitter.call_args.args[1], admission_history)
        self.assertEqual(len(p.actions), 3)
        self.assertTrue(np.isfinite(result).all())

    def test_prefix_refit_keeps_compatible_linear_trajectory(self):
        future = np.arange(16)[:, None] * np.full((1, 6), .002)
        history = np.array([-2, -1])[:, None] * np.full((1, 6), .002)
        fit = fit_local_prefix(future, history)
        np.testing.assert_allclose(fit.future, future, atol=1e-12)

    def test_prefix_smoother_preserves_gripper_and_limits(self):
        cfg = config()
        cfg["runtime"]["joint_bspline"] = {"enabled": True, "method": "prefix_refit",
            "control_points": 8, "history_actions": 2, "refit_free_control_points": 3}
        ctrl = FakeController()
        ctrl.lower = np.full(6, -.1)
        ctrl.upper = np.full(6, .1)
        smoother = CubicBSplineJointSmoother(cfg, ctrl, 16)
        smoother.reset(np.zeros(6))
        chunk = np.full((16, 7), .2, dtype=np.float32)
        chunk[:, 6] = np.linspace(0, .8, 16)
        result = smoother.smooth(chunk)
        np.testing.assert_array_equal(result[:, 6], chunk[:, 6])
        self.assertTrue(np.isfinite(result).all())
        self.assertLessEqual(result[:, :6].max(), .100001)
        self.assertEqual(smoother.last_metrics["bspline_method"], "prefix_refit")

    def test_bad_experiment_settings_are_rejected(self):
        cfg = config()
        cfg["runtime"]["chunk_update_mode"] = "typo"
        with self.assertRaises(ValueError):
            _chunk_update_mode(cfg)
        cfg["runtime"]["joint_bspline"] = {"enabled": True, "method": "prefix_refit", "control_points": 8}
        with self.assertRaises(ValueError):
            _joint_bspline_settings(cfg, 4)
        cfg["runtime"]["joint_bspline"]["refit_free_control_points"] = 8
        with self.assertRaises(ValueError):
            _joint_bspline_settings(cfg, 16)

    def test_prefix_rejects_stale_generation_in_latest_mode(self):
        cfg = config()
        cfg["runtime"]["chunk_update_mode"] = "latest_ready"
        p = self.make_manual_policy(cfg)
        p._submit({})
        future = p.future
        future.set_running_or_notify_cancel()
        p.stop()
        preview = FakeAgent()._predict_chunk()[:4]
        # Restart without waiting for the old inference, as a lifecycle stress test.
        p.start(preview)
        future.set_result((FakeAgent()._predict_chunk() + 100, 1.0, 1.1))
        np.testing.assert_array_equal(p.next_action({}), preview[0])


if __name__ == "__main__":
    unittest.main()
