"""CPU tests for RoboTwin flat indexing and inference reuse; no simulation/GPU."""
import ast
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robotwin_action_prior import FrameIndices
from metaworld_vla.action_reuse import TailActionPrior


class Tests(unittest.TestCase):
    def test_flat_indexing_no_episode_leak(self):
        idx = FrameIndices([dict(global_start=0, global_end=40), dict(global_start=40, global_end=100)])
        self.assertEqual(idx[0], (0, 0))
        self.assertEqual(idx[39], (0, 39))
        self.assertEqual(idx[40], (1, 0))
        self.assertEqual(idx[99], (1, 59))

    def test_actual_robotwin_engine_tail_and_reset(self):
        tree = ast.parse((Path(__file__).resolve().parents[1]/'robotwin_infer.py').read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'RobotWinInference')
        ns = dict(torch=torch, np=np, Dict=dict, Any=object)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'robotwin_infer.py', 'exec'), ns)
        engine = ns['RobotWinInference'].__new__(ns['RobotWinInference'])
        engine.config = NS(common=NS(action_chunk_size=32, action_dim=14))
        engine.device, engine.dtype = 'cpu', torch.float32
        engine.use_task_cond, engine.task_cond = False, None
        engine.num_inference_steps, engine.action_execution_horizon = 10, 16
        engine.noise_generator = None
        engine.smooth_actions = False
        engine.action_prior = TailActionPrior(True, sigma=0)
        engine.action_queue = deque()
        calls = []
        class Action:
            use_global_intent = use_spatial_intent = False
            def __call__(self, **kwargs):
                calls.append(kwargs['noisy_actions'].clone())
                return dict(final_pred=torch.zeros_like(kwargs['noisy_actions']))
        engine.model = NS(action_model=Action(), normalize_state=lambda x:x,
                           get_vision_features=lambda x:[], denormalize_action=lambda x:x)
        engine.processor = NS(reset=lambda:None, update_state_buffer=lambda obs:None,
            process=lambda obs:dict(state=torch.zeros(1,1,16), pixel_values=torch.zeros(1,3,240,320)))
        for _ in range(16): self.assertEqual(engine.step({}).shape, (14,))
        tail = engine.action_prior.previous[:, 16:].clone()
        engine.step({})
        self.assertTrue(torch.equal(calls[10][:, :16], tail))
        self.assertEqual(engine.action_prior.last_overlap, 16)
        engine.reset()
        self.assertIsNone(engine.action_prior.previous)
        self.assertFalse(engine.action_queue)


if __name__ == '__main__':
    unittest.main()
