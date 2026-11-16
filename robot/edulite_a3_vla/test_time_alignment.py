"""Offline time-origin, stale-result and unused-tail safety regressions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np

from real_vla import AsyncActionChunks, _time_alignment_enabled
from runtime_diagnostics import RuntimeDiagnostics
from test_runtime_diagnostics import FakeController, ManualExecutor


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, duration):
        self.now += duration


class Agent:
    action_execution_horizon = 16
    config = SimpleNamespace(common=SimpleNamespace(action_chunk_size=32))
    processor = SimpleNamespace(update_state_buffer=lambda obs: None)

    def reset(self):
        pass

    def _predict_chunk(self, *args):
        result = np.repeat(np.arange(32, dtype=np.float32)[:, None] / 100, 7, axis=1)
        result[:, 6] = np.linspace(0, .8, 32)
        return result


class Controller(FakeController):
    def joint_command_snapshot(self):
        return np.zeros(6)

    def validate_model_action(self, action):
        if np.max(np.abs(action[:6])) > 1:
            raise RuntimeError("unsafe selected joint")
        return action, []


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.clock_patch = patch("real_vla.time.monotonic", self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.cfg = {"runtime": {"action_rate_hz": 30, "chunk_update_mode": "latest_ready",
                    "time_alignment": {"enabled": True}, "prefetch_threshold_actions": 8,
                    "joint_bspline": {"enabled": False}, "diagnostics": {"enabled": True}},
                    "model": {}, "safety": {}}

    def obs(self, age=0):
        return {"_runtime_timing": {"observation_mono_s": self.clock() - age,
                                    "camera_mono_s": self.clock() - age - .02}}

    def policy(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        controller = Controller()
        controller.diagnostics = RuntimeDiagnostics(cfg)
        p = AsyncActionChunks(Agent(), controller, "", cfg)
        p.executor.shutdown()
        p.executor = ManualExecutor()
        self.addCleanup(p.close)
        return p

    def start(self, p):
        p.start(np.full((16, 7), -.5), observation=self.obs())

    def pending(self, p, age=.2, values=None):
        p.pending = Agent()._predict_chunk() if values is None else values.copy()
        p._pending_meta = p._metadata(self.obs(age), "background")
        p._pending_meta.update(infer_started_mono_s=self.clock(), completed_mono_s=self.clock())

    def test_start_refreshes_instead_of_executing_old_preview(self):
        p = self.policy()
        def infer(observation):
            started = self.clock()
            self.clock.advance(.2)
            return Agent()._predict_chunk(), started, self.clock()
        p._timed_infer = infer
        self.start(p)
        np.testing.assert_array_equal(np.stack(p.actions), Agent()._predict_chunk()[6:22])
        self.assertEqual(p.current_meta["source"], "startup_refresh")
        self.assertEqual(p.current_meta["aligned_start_index"], 6)
        self.assertAlmostEqual(p.current_meta["alignment_residual_age_s"], 0)
        self.assertAlmostEqual(p.current_meta["aligned_start_to_command_max_rad"], .06)

    def test_full_prediction_kept_and_offset_recomputed_at_admission(self):
        p = self.policy()
        self.start(p)
        p._submit(self.obs())
        p.future.set_result((Agent()._predict_chunk(), self.clock(), self.clock() + .2))
        self.clock.advance(.2)
        p._poll()
        self.assertEqual(p.pending.shape, (32, 7))
        self.clock.advance(.1)
        self.assertTrue(p._admit_pending())
        self.assertEqual(p.current_meta["aligned_start_index"], 9)
        np.testing.assert_array_equal(np.stack(p.actions), Agent()._predict_chunk()[9:25])

    def test_integer_boundary_uses_floor_not_round(self):
        p = self.policy()
        self.start(p)
        self.pending(p, age=.219)
        p._admit_pending()
        self.assertEqual(p.current_meta["aligned_start_index"], 6)
        self.assertAlmostEqual(p.current_meta["alignment_residual_age_s"], .019)

    def test_alignment_slices_before_prefix_refitting(self):
        self.cfg["runtime"]["joint_bspline"] = {"enabled": True, "method": "prefix_refit",
            "control_points": 8, "history_actions": 2, "refit_free_control_points": 3}
        p = self.policy()
        self.start(p)
        self.pending(p)
        from trajectory_refit import fit_local_prefix
        with patch("real_vla.fit_local_prefix", wraps=fit_local_prefix) as fitter:
            p._admit_pending()
        np.testing.assert_array_equal(fitter.call_args.args[0], Agent()._predict_chunk()[6:22, :6])
        np.testing.assert_array_equal(np.stack(p.actions)[:, 6], Agent()._predict_chunk()[6:22, 6])

    def test_drain_mode_aligns_when_queue_finishes_not_when_result_arrives(self):
        self.cfg["runtime"]["chunk_update_mode"] = "drain_queue"
        p = self.policy()
        self.start(p)
        old = np.stack(p.actions).copy()
        self.pending(p, age=0)
        for i in range(16):
            np.testing.assert_array_equal(p.next_action(self.obs()), old[i])
            self.clock.advance(1 / 30)
        returned = p.next_action(self.obs())
        np.testing.assert_array_equal(returned, Agent()._predict_chunk()[16])
        self.assertEqual(p.current_meta["aligned_start_index"], 16)

    def test_expired_result_preserves_old_queue_and_requests_fresh_inference(self):
        p = self.policy()
        with TemporaryDirectory() as directory:
            p.diagnostics.start(directory)
            self.start(p)
            old = np.stack(p.actions).copy()
            self.pending(p, age=.6)  # index18 + horizon16 is beyond32
            returned = p.next_action(self.obs())
            np.testing.assert_array_equal(returned, old[0])
            np.testing.assert_array_equal(np.stack(p.actions), old[1:])
            self.assertIsNotNone(p.future)
            self.assertEqual(p.diagnostics.alignment_rejections[0]["reason"], "insufficient_tail")
            p.stop()
            p.diagnostics.finish("test")
            summary = json.loads((Path(directory) / "runtime_summary.json").read_text())
            self.assertEqual(summary["time_alignment_rejected_chunks"], 1)
            self.assertTrue((Path(directory) / "alignment_rejections.csv").exists())

    def test_newly_used_tail_is_validated_before_queue_replacement(self):
        p = self.policy()
        self.start(p)
        old = np.stack(p.actions).copy()
        bad = Agent()._predict_chunk()
        bad[20, 0] = 9  # outside the old prefix but inside [6:22]
        self.pending(p, values=bad)
        with self.assertRaises(RuntimeError):
            p._admit_pending()
        np.testing.assert_array_equal(np.stack(p.actions), old)
        # An expired prefix is never sent; all seven selected dimensions use
        # identical temporal slicing, including the gripper.
        bad = Agent()._predict_chunk()
        bad[0, 0] = 9
        self.pending(p, values=bad)
        p._admit_pending()
        np.testing.assert_array_equal(np.stack(p.actions), Agent()._predict_chunk()[6:22])
        self.assertEqual(p.current_meta["skipped_gripper_min_norm"], 0)

    def test_last_full_window_is_valid_but_partial_tail_is_rejected(self):
        p = self.policy()
        self.start(p)
        self.pending(p, age=16 / 30)
        p._admit_pending()
        np.testing.assert_array_equal(np.stack(p.actions), Agent()._predict_chunk()[16:32])
        self.pending(p, age=17 / 30)
        old = np.stack(p.actions).copy()
        self.assertFalse(p._admit_pending())
        np.testing.assert_array_equal(np.stack(p.actions), old)

    def test_fit_overrun_is_bounded_and_does_not_clear_old_queue(self):
        p = self.policy()
        self.start(p)
        self.pending(p)
        old = np.stack(p.actions).copy()
        def slow_fit(window):
            self.clock.advance(1 / 30)
            return window.copy()
        with patch.object(p.smoother, "smooth", side_effect=slow_fit) as fitter:
            self.assertFalse(p._admit_pending())
            self.assertEqual(fitter.call_count, 2)
        np.testing.assert_array_equal(np.stack(p.actions), old)
        self.assertTrue(p._alignment_refresh_needed)

    def test_start_fails_safely_if_refresh_expires(self):
        p = self.policy()
        def slow_infer(observation):
            started = self.clock()
            self.clock.advance(.7)
            return Agent()._predict_chunk(), started, self.clock()
        p._timed_infer = slow_infer
        with self.assertRaisesRegex(RuntimeError, "startup prediction expired"):
            self.start(p)
        self.assertEqual(len(p.actions), 0)
        self.assertIsNone(p.current_meta)

    def test_disabled_feature_retains_original_prefix_and_start(self):
        for alignment in [None, {"enabled": False}]:
            cfg = {"runtime": dict(self.cfg["runtime"]), "model": {}, "safety": {}}
            if alignment is None:
                cfg["runtime"].pop("time_alignment")
            else:
                cfg["runtime"]["time_alignment"] = alignment
            p = self.policy(cfg)
            preview, _ = p.preview({})
            with patch.object(p, "_timed_infer", side_effect=AssertionError("legacy startup must not refresh")):
                p.start(preview)
            np.testing.assert_array_equal(np.stack(p.actions), Agent()._predict_chunk()[:16])
            p._submit({})
            p.future.set_result((Agent()._predict_chunk(), self.clock(), self.clock()))
            p._poll()
            self.assertEqual(p.pending.shape, (16, 7))

    def test_bad_config_and_time_origins_are_rejected(self):
        with self.assertRaises(ValueError):
            _time_alignment_enabled(self.cfg, 16, 16)
        with self.assertRaises(ValueError):
            _time_alignment_enabled({"runtime": {"time_alignment": {"enabled": "false"}}}, 32, 16)
        p = self.policy()
        for observation in [{}, self.obs(-1), self.obs(float("nan"))]:
            with self.assertRaises(ValueError):
                p._metadata(observation, "background")
        with self.assertRaises(ValueError):
            p.start(np.zeros((16, 7)))


if __name__ == "__main__":
    unittest.main()
