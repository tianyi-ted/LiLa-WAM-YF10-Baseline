"""CPU regression tests without loading DINO, checkpoints or MuJoCo."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_reuse import TailActionPrior


def policy_class():
    # Exercise the actual Policy methods while avoiding heavy module imports.
    tree = ast.parse(Path(__file__).with_name('eval.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Policy')
    namespace = dict(torch=torch, np=np, time=time,
                     cv2=NS(INTER_LINEAR=1, resize=lambda rgb, size, interpolation: rgb),
                     _normalise_rgb=lambda rgb: torch.zeros(3, 2, 2))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'eval.py', 'exec'), namespace)
    return namespace['Policy']


class Action:
    use_global_intent = False
    inject_global_intent = False

    def __init__(self):
        self.inputs = []

    def __call__(self, **kw):
        self.inputs.append(kw['noisy_actions'].clone())
        return dict(final_pred=torch.full_like(kw['noisy_actions'], 0.125))


def fake_policy(enabled=False, sigma=1.0):
    cls = policy_class()
    p = cls.__new__(cls)
    p.cfg = NS(common=NS(action_chunk_size=8, action_dim=4,
                        action_execution_horizon=4, num_inference_steps=10),
               dataset=NS(image_size=(2, 2)))
    p.device, p.dtype = torch.device('cpu'), torch.float32
    p.prior = TailActionPrior(enabled, sigma)
    p.generator, p.profile_generation = None, False
    p.task_cond, p.queue = None, []
    p.wrapper = NS(normalize_state=lambda x: x, get_vision_features=lambda x: [],
                   denormalize_action=lambda x: x)
    p.action = Action()
    p.reset_metrics()
    p.reset_episode()
    return p


class ReuseTests(unittest.TestCase):
    def test_source_alignment_and_reset(self):
        prior = TailActionPrior(True, 0.5)
        old = torch.arange(32.).reshape(1, 8, 4)
        prior.remember(old)
        for _ in range(4):
            prior.advance()
        noise = torch.ones_like(old)
        x = prior.source(noise)
        self.assertTrue(torch.equal(x[:, :4], old[:, 4:] + 0.5))
        self.assertTrue(torch.equal(x[:, 4:], noise[:, 4:]))
        self.assertEqual(prior.last_overlap, 4)
        self.assertTrue(torch.equal(noise, torch.ones_like(old)))
        prior.reset()
        self.assertIs(prior.source(noise), noise)

    def test_identity_and_full_execution(self):
        p = TailActionPrior(True, 1, 0)
        noise = torch.randn(1, 8, 4)
        p.remember(torch.randn_like(noise))
        p.advance()
        self.assertTrue(torch.equal(p.source(noise), noise))
        for _ in range(7):
            p.advance()
        self.assertTrue(torch.equal(p.source(noise), noise))

    def test_invalid(self):
        for sigma in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                TailActionPrior(True, sigma)
        p = TailActionPrior(True)
        p.remember(torch.zeros(1, 8, 4))
        with self.assertRaises(ValueError):
            p.source(torch.zeros(1, 8, 4))

    def test_disabled_matches_legacy_euler_and_rng(self):
        p = fake_policy(False)
        torch.manual_seed(123)
        expected = torch.randn(1, 8, 4)
        steps = torch.linspace(0, 1, 11)
        for i in range(10):
            expected += torch.full_like(expected, .125) * (steps[i+1]-steps[i])
        next_noise = torch.randn(3)
        torch.manual_seed(123)
        actual = p.chunk(np.zeros(4, np.float32), np.zeros((2, 2, 3)))
        self.assertTrue(np.array_equal(actual, expected[0].numpy()))
        self.assertTrue(torch.equal(torch.randn(3), next_noise))
        self.assertIsNone(p.prior.previous)

    def test_policy_queue_and_episode_reset(self):
        p = fake_policy(True, 0)
        obs, rgb = np.zeros(4, np.float32), np.zeros((2, 2, 3))
        for _ in range(4):
            p.step(obs, rgb)
        old_tail = p.prior.previous[:, 4:].clone()
        self.assertEqual(p.prior.executed, 4)
        p.step(obs, rgb)
        self.assertTrue(torch.equal(p.action.inputs[10][:, :4], old_tail))
        self.assertEqual(p.metrics['reused_chunks'], 1)
        self.assertEqual(p.metrics['reused_positions'], 4)
        p.reset_episode()
        self.assertIsNone(p.prior.previous)
        self.assertIsNone(p.last_sent)
        p.step(obs, rgb)
        self.assertEqual(p.prior.last_overlap, 0)
        self.assertEqual(len(p.queue), 3)

    def test_separate_rng_does_not_advance_global(self):
        p = fake_policy(True)
        p.generator = torch.Generator().manual_seed(456)
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        p.chunk(np.zeros(4, np.float32), np.zeros((2, 2, 3)))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_parallel_merge_validates_and_keeps_reuse_protocol(self):
        tree = ast.parse(Path(__file__).with_name('eval_parallel.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'merge_results')
        scope = dict(json=json, Path=Path)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'eval_parallel.py', 'exec'), scope)
        with tempfile.TemporaryDirectory() as d:
            ck = str(Path(d, 'ck.pt').resolve())
            files = []
            for task in ('a', 'b'):
                path = Path(d, task + '.json')
                path.write_text(json.dumps(dict(checkpoint=ck, seed=0, horizon=8, test_num=1,
                    rng_protocol='test', action_generation=dict(action_reuse=True),
                    tasks={task: dict(episodes=1, success=0, success_rate=0.0)})))
                files.append(path)
            merged = scope['merge_results'](files, ['a', 'b'], ck, 1)
            self.assertTrue(merged['action_generation']['action_reuse'])
            bad = json.loads(files[1].read_text())
            bad['action_generation']['action_reuse'] = False
            files[1].write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                scope['merge_results'](files, ['a', 'b'], ck, 1)


if __name__ == '__main__':
    unittest.main()
