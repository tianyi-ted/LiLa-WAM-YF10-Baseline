"""Regression tests for opt-in source loss, causal cache indexing and mixing."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from training_prior import (prepare_source_batch, PriorDataset, cache_spec,
                            cache_key, episode_record, sha256, expected_cache_spec)


def loss_function():
    path = Path(__file__).resolve().parents[1] / 'models/vla_model_fm.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'calc_flow_matching_loss')
    scope = dict(torch=torch, F=F)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    return scope['calc_flow_matching_loss']


class Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))

    def forward(self, t, noisy_actions, **kw):
        self.input = noisy_actions.detach().clone()
        self.time = t.detach().clone()
        return dict(final_pred=self.weight * noisy_actions, cond_tokens=noisy_actions)


class PriorTests(unittest.TestCase):
    def test_explicit_g2_cache_transfer_is_narrow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root/'stats.json'; stats.write_text('{}')
            ref = OmegaConf.create(dict(common=dict(action_chunk_size=8),
                model=dict(vision_encoder=dict(checkpoint_path='dino')), dataset=dict(camera_names=['corner2'])))
            path = root/'reference.yaml'; OmegaConf.save(ref, path)
            student = OmegaConf.create(OmegaConf.to_container(ref))
            student.model.global_intent = dict(enabled=True, pooling='task_attention')
            opts = dict(cache_reference_config=str(path))
            self.assertEqual(expected_cache_spec(student, stats, opts), cache_spec(ref, stats))
            student.dataset.camera_names = ['other']
            with self.assertRaises(ValueError): expected_cache_spec(student, stats, opts)

    def test_loss_identity_preserves_rng_grad_and_target(self):
        fn = loss_function()
        x = torch.ones(2, 8, 4)
        outputs = []
        for override in (False, True):
            model = Stub()
            torch.manual_seed(123)
            kw = dict(source_mean=torch.zeros_like(x), source_noise_scale=torch.ones_like(x)) if override else {}
            loss, info = fn(model, x, [], torch.zeros(2, 1, 4), **kw)
            loss.backward()
            outputs.append((loss.detach(), info['target_v'], model.weight.grad, torch.get_rng_state()))
        for a, b in zip(*outputs):
            self.assertTrue(torch.equal(a, b))

    def test_changed_source_uses_A_minus_Z(self):
        fn = loss_function()
        x, model = torch.ones(2, 8, 4), Stub()
        torch.manual_seed(4)
        eps = torch.randn_like(x)
        z = 2 + .5 * eps
        torch.manual_seed(4)
        loss, info = fn(model, x, [], torch.zeros(2, 1, 4),
                        source_mean=torch.full_like(x, 2), source_noise_scale=torch.full_like(x, .5))
        self.assertTrue(torch.equal(info['target_v'], x-z))
        s = model.time[:, None, None]
        self.assertTrue(torch.equal(model.input, (1-s)*z+s*x))
        loss.backward()
        self.assertTrue(torch.isfinite(model.weight.grad))

    def test_independent_selection_and_N_R_control(self):
        outputs = []
        initial = torch.get_rng_state().clone()
        for mode in ('scale', 'tail'):
            batch = dict(cached_prior_norm=torch.ones(3, 8, 4),
                         cached_prior_valid=torch.cat([torch.ones(3, 4, 1, dtype=torch.bool),
                                                       torch.zeros(3, 4, 1, dtype=torch.bool)], 1))
            opts = dict(mode=mode, cold_probability=0, strength=1, sigma=.5)
            prepare_source_batch(batch, opts, torch.Generator().manual_seed(1))
            self.assertTrue(torch.equal(batch['fm_source_scale'][:, 4:], torch.ones(3, 4, 4)))
            outputs.append(batch)
        self.assertTrue(torch.equal(initial, torch.get_rng_state()))
        self.assertTrue(torch.equal(outputs[0]['fm_source_scale'], outputs[1]['fm_source_scale']))
        self.assertEqual(outputs[0]['fm_source_mean'].sum(), 0)
        self.assertEqual(outputs[1]['fm_source_mean'].sum(), 48)

    def test_causal_index_and_termination_mask_and_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            epfile, stats = root/'episode.hdf5', root/'stats.json'
            epfile.touch(); stats.write_text('{}')
            cfg = OmegaConf.create(dict(common={}, model={}, dataset={}))
            ep = dict(path=str(epfile), task='x', length=10)
            class Base:
                episodes = [ep]
                indices = [(0, t) for t in range(10)]
                action_dim = 4
                def __len__(self): return 10
                def __getitem__(self, i): return dict(action_sequence=torch.zeros(8, 4))
            base = Base()
            key = cache_key(epfile)
            np.save(root/key, np.arange(160, dtype=np.float32).reshape(10, 4, 4))
            manifest = dict(spec=cache_spec(cfg, stats), complete=True, episodes={key:dict(
                source=episode_record(base, ep), sha256=sha256(root/key))})
            (root/'manifest.json').write_text(json.dumps(manifest))
            opts = dict(mode='tail', cache_dir=str(root), execution=4, overlap=4)
            ds = PriorDataset(base, cfg, stats, opts)
            self.assertFalse(ds[3]['cached_prior_valid'].any())
            self.assertEqual(ds[4]['cached_prior_norm'][0, 0], 64)
            self.assertEqual(ds[9]['cached_prior_valid'].sum(), 1)
            manifest['complete'] = False
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): PriorDataset(base, cfg, stats, opts)


if __name__ == '__main__':
    unittest.main()
