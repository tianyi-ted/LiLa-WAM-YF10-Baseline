"""CPU transfer regression: selective load, gradients, group floors and resume."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'metaworld_vla'))
from models.vla_model_fm import VLAModel
from models.planwam_g2 import planwam_flow_loss
from planwam_transfer import (
    initialize_action_from_teacher, build_transfer_optimizer,
    validate_transfer_config, UNUSED_ROOTS, PREDICTOR_ROOTS,
)
from train_planwam import VERSION, validate_config


class TransferTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'teacher.pt'
        self.kw = dict(action_dim=4, proprio_dim=4, hidden_dim=32, depth=1, action_len=8,
                       num_heads=4, dino_feat_dims=(32,), vlm_num_queries=4, adapter_depth=1,
                       task_cond_dim=32, use_global_intent=True, inject_global_intent=True,
                       intent_hidden_dim=32, intent_use_gate=False, inject_task_cond=False,
                       intent_pooling='task_attention', intent_pool_hidden_dim=32,
                       intent_pool_grid_size=(3, 4))
        torch.manual_seed(9)
        self.teacher = VLAModel(**self.kw)
        torch.nn.init.normal_(self.teacher.blocks[0].adaLN_modulation[-1].weight, std=0.05)
        torch.nn.init.normal_(self.teacher.output_proj.weight, std=0.05)
        torch.manual_seed(11)
        self.action = VLAModel(**self.kw)
        self.contract = {'test': 'inputs_v1'}
        self.spec = {'vision_dim': 32, 'grid_size': [3, 4]}
        self.ck = dict(planwam_stage='teacher', planwam_version=VERSION, epoch=40, epoch_complete=True,
                       input_contract=self.contract, posterior_spec=self.spec, model_state=self.teacher.state_dict())
        torch.save(self.ck, self.path)
        self.training = OmegaConf.create(dict(learning_rate=2e-4, lr_min=5e-5, betas=[0.9, 0.99], weight_decay=0.01))
        self.groups = OmegaConf.create(dict(inherited_lr=2e-5, inherited_lr_min=5e-6,
                                            predictor_lr=2e-4, predictor_lr_min=5e-5))

    def tearDown(self):
        self.tmp.cleanup()

    def initialize(self, action=None, **kwargs):
        return initialize_action_from_teacher(self.action if action is None else action, self.path,
                                               self.contract, self.spec, VERSION, **kwargs)

    def test_whitelist_fresh_predictor_unused_keys_and_rng(self):
        before = {k: v.clone() for k, v in self.action.state_dict().items()}
        rng = torch.get_rng_state().clone()
        source_bytes = self.path.read_bytes()
        report = self.initialize()
        after = self.action.state_dict()
        for k in report['copied_state_keys']:
            torch.testing.assert_close(after[k], self.teacher.state_dict()[k], atol=0, rtol=0)
        for k in report['fresh_predictor_state_keys'] + report['unused_compatibility_state_keys']:
            torch.testing.assert_close(after[k], before[k], atol=0, rtol=0)
        self.assertTrue(any(not torch.equal(after[k], self.teacher.state_dict()[k]) for k in report['fresh_predictor_state_keys']))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(source_bytes, self.path.read_bytes())
        self.assertFalse(any(k.startswith('task_cond_proj.') for k in report['copied_state_keys']))

    def test_resume_inspection_never_applies_teacher_weights(self):
        before = {k: v.clone() for k, v in self.action.state_dict().items()}
        report = self.initialize(apply_weights=False)
        self.assertTrue(report['copied_state_keys'])
        self.assertTrue(all(torch.equal(before[k], v) for k, v in self.action.state_dict().items()))

    def test_protocol_and_source_guards_before_copy(self):
        variants = [dict(planwam_stage='student'), dict(planwam_version='wrong'),
                    dict(epoch_complete=False), dict(input_contract={'wrong': True}),
                    dict(posterior_spec={'wrong': True})]
        before = {k: v.clone() for k, v in self.action.state_dict().items()}
        for update in variants:
            torch.save(dict(self.ck, **update), self.path)
            with self.assertRaises(ValueError):
                self.initialize()
            self.assertTrue(all(torch.equal(before[k], v) for k, v in self.action.state_dict().items()))
        state = copy.deepcopy(self.ck['model_state'])
        state['output_proj.weight'][0, 0] = float('nan')
        torch.save(dict(self.ck, model_state=state), self.path)
        with self.assertRaises(FloatingPointError):
            self.initialize()
        state = copy.deepcopy(self.ck['model_state'])
        state['output_proj.weight'] = torch.zeros(1)
        torch.save(dict(self.ck, model_state=state), self.path)
        with self.assertRaises(ValueError):
            self.initialize()

    def test_groups_are_disjoint_complete_and_reach_individual_floors(self):
        report = self.initialize()
        opt, scheduler, protocol = build_transfer_optimizer(self.action, self.training, self.groups, report, 10)
        grouped = [n for group in protocol['groups'] for n in group['parameter_names']]
        expected = {n for n, p in self.action.named_parameters() if p.requires_grad and n.split('.')[0] not in UNUSED_ROOTS}
        self.assertEqual(set(grouped), expected)
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertTrue(all(n.split('.')[0] in PREDICTOR_ROOTS for n in protocol['groups'][1]['parameter_names']))
        self.assertEqual(scheduler.get_last_lr(), [2e-5, 2e-4])
        for _ in range(5):
            opt.step(); scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1.25e-5)
        self.assertAlmostEqual(scheduler.get_last_lr()[1], 1.25e-4)
        for _ in range(5):
            opt.step(); scheduler.step()
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 5e-6)
        self.assertAlmostEqual(scheduler.get_last_lr()[1], 5e-5)

    def test_transfer_action_only_updates_predictor_and_inherited_dit(self):
        report = self.initialize()
        opt, _, _ = build_transfer_optimizer(self.action, self.training, self.groups, report, 10)
        features = torch.randn(2, 12, 32)
        state, task, actions = torch.randn(2, 1, 4), torch.randn(2, 32), torch.randn(2, 8, 4)
        loss, info = planwam_flow_loss(self.action, features, None, [features], state, task, actions,
                                      'action_only', noise=torch.randn_like(actions), flow_time=torch.full((2,), 0.5))
        self.assertEqual(info['loss_kd'], 0)
        loss.backward()
        chosen = [self.action.intent_pool.query[1].weight, self.action.output_proj.weight]
        for p in chosen:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(p.grad.norm(), 0)
        old = [p.clone() for p in chosen]
        opt.step()
        self.assertTrue(all(not torch.equal(p, b) for p, b in zip(chosen, old)))
        self.assertIsNone(self.action.task_cond_proj[1].weight.grad)

    def test_optimizer_scheduler_resume_matches_uninterrupted(self):
        report = self.initialize()
        opt, sched, _ = build_transfer_optimizer(self.action, self.training, self.groups, report, 4)
        def step(action, optimizer, scheduler):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                for p in group['params']:
                    p.grad = torch.full_like(p, 0.01)
            optimizer.step(); scheduler.step()
        for _ in range(2):
            step(self.action, opt, sched)
        saved = copy.deepcopy(dict(model=self.action.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict()))
        for _ in range(2):
            step(self.action, opt, sched)
        resumed = VLAModel(**self.kw)
        report2 = self.initialize(resumed, apply_weights=False)
        opt2, sched2, _ = build_transfer_optimizer(resumed, self.training, self.groups, report2, 4)
        resumed.load_state_dict(saved['model'], strict=True)
        opt2.load_state_dict(saved['optimizer']); sched2.load_state_dict(saved['scheduler'])
        for _ in range(2):
            step(resumed, opt2, sched2)
        self.assertEqual(sched.get_last_lr(), sched2.get_last_lr())
        self.assertTrue(all(torch.equal(v, resumed.state_dict()[k]) for k, v in self.action.state_dict().items()))

    def test_old_and_new_configs_and_model_keys(self):
        root = ROOT / 'metaworld_vla/configs'
        old = OmegaConf.load(root / 'metaworld_mt50_easy_planwam_action_only.yaml')
        self.assertEqual(validate_transfer_config(old), (False, False))
        for suffix, stage, grouped in [('action_only', 'action_only', True), ('student', 'student', True),
                                       ('action_only_uniformlr', 'action_only', False)]:
            cfg = OmegaConf.load(root / f'metaworld_mt50_easy_planwam_action_init_{suffix}.yaml')
            self.assertEqual(validate_config(cfg), stage)
            self.assertEqual(validate_transfer_config(cfg), (True, grouped))
            self.assertEqual(OmegaConf.to_container(cfg.model), OmegaConf.to_container(old.model))
            self.assertEqual(OmegaConf.to_container(cfg.training), OmegaConf.to_container(old.training))
        original_keys = set(self.action.state_dict())
        self.initialize()
        self.assertEqual(set(self.action.state_dict()), original_keys)
        cfg.planwam.grouped_lr.enabled = True
        cfg.planwam.grouped_lr.inherited_lr_min = 5e-5
        with self.assertRaises(ValueError):
            validate_transfer_config(cfg)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
