"""CPU checks for opt-in late pooling; no simulator or DINO weights needed."""
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
from models.vla_model_fm import VLAModel, LateTaskConditionedIntentPool
from models.planwam_g2 import FutureDrivenPosterior, freeze_module, planwam_flow_loss
from models.intent_pool_contract import teacher_input_contract
from metaworld_vla.planwam_transfer import initialize_action_from_teacher, build_transfer_optimizer
from metaworld_vla.train_planwam import VERSION, validate_config, load_frozen_teacher


class LatePoolTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(711)
        self.kw = dict(action_dim=4, proprio_dim=4, hidden_dim=32, depth=1,
                       action_len=8, num_heads=4, dino_feat_dims=(32,), vlm_num_queries=4,
                       adapter_depth=1, task_cond_dim=32, use_global_intent=True,
                       inject_global_intent=True, intent_hidden_dim=32,
                       intent_use_gate=False, inject_task_cond=False,
                       intent_pooling='task_attention', intent_pool_hidden_dim=32,
                       intent_pool_grid_size=(3, 4))
        self.source = VLAModel(**self.kw)
        self.kw['intent_pooling'] = 'task_attention_late'
        self.action = VLAModel(**self.kw)
        self.current, self.future = torch.randn(2, 12, 32), torch.randn(2, 12, 32)
        self.state, self.task, self.actions = torch.randn(2, 1, 4), torch.randn(2, 32), torch.randn(2, 8, 4)
        self.contract = dict(model=dict(global_intent=dict(pooling='task_attention_late',
                             pool_num_queries=4, pool_interaction_depth=1)),
                             stats_sha256='fixed-stats', dataset=dict(future_frame_offset=8))
        self.spec = dict(vision_dim=32, state_dim=4, task_dim=32, grid_size=[3, 4],
                         hidden_dim=32, num_heads=4, horizon_dim=32)
        self.posterior = freeze_module(FutureDrivenPosterior(**self.spec))
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'teacher.pt'
        self.ck = dict(planwam_stage='teacher', planwam_version=VERSION, epoch=40,
                       epoch_complete=True, input_contract=teacher_input_contract(self.contract),
                       posterior_spec=self.spec, model_state=self.source.state_dict(),
                       posterior_state=self.posterior.state_dict())
        torch.save(self.ck, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def initialize(self, contract=None):
        return initialize_action_from_teacher(self.action, self.path,
                                              self.contract if contract is None else contract,
                                              self.spec, VERSION)

    def test_pool_shape_attention_inputs_and_guards(self):
        pool = self.action.intent_pool.eval()
        pooled, weights = pool(self.current, self.state, self.task, return_attention=True)
        self.assertEqual(tuple(pooled.shape), (2, 32))
        self.assertEqual(tuple(weights.shape), (2, 12))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2))
        for args in [(self.current + torch.randn_like(self.current), self.state, self.task),
                     (self.current, self.state + 1, self.task),
                     (self.current, self.state, self.task + torch.randn_like(self.task))]:
            self.assertFalse(torch.allclose(pooled, pool(*args)))
        with self.assertRaises(ValueError):
            pool(self.current[:, :-1], self.state, self.task)
        with self.assertRaises(ValueError):
            pool(self.current, self.state, None)
        with self.assertRaises(ValueError):
            LateTaskConditionedIntentPool(32, 4, 32, (3, 4), hidden_dim=32, num_queries=1)
        # Existing attention-weighted-target mode is intentionally not supported.
        with self.assertRaises(ValueError):
            VLAModel(**dict(self.kw, intent_target_pooling='current_attention'))

    def test_bridge_transfers_only_compatible_action_modules(self):
        before = {k: v.clone() for k, v in self.action.state_dict().items()}
        report = self.initialize()
        self.assertEqual(report['predictor_bridge'], 'task_attention_late_v1')
        self.assertTrue(report['ignored_source_pool_state_keys'])
        for k in report['copied_state_keys']:
            torch.testing.assert_close(self.action.state_dict()[k], self.ck['model_state'][k], rtol=0, atol=0)
        for k in report['fresh_predictor_state_keys']:
            torch.testing.assert_close(self.action.state_dict()[k], before[k], rtol=0, atol=0)
        self.assertTrue(any(k.startswith('intent_pool.slots') for k in report['fresh_predictor_state_keys']))
        canonical = teacher_input_contract(self.contract)
        self.assertEqual(self.contract['model']['global_intent']['pooling'], 'task_attention_late')
        self.assertEqual(canonical['model']['global_intent'], dict(pooling='task_attention'))
        for mutation in [('stats_sha256', 'wrong'), ('dataset', dict(future_frame_offset=4)),
                         ('model', dict(global_intent=dict(pooling='mean')))]:
            bad = copy.deepcopy(self.contract)
            bad[mutation[0]] = mutation[1]
            with self.assertRaises(ValueError):
                self.initialize(bad)
        # Dropping the opt-in contract may NOT bypass the legacy key-set guard.
        with self.assertRaises(ValueError):
            self.initialize(canonical)
        bad = copy.deepcopy(self.ck)
        bad['model_state']['output_proj.weight'] = torch.zeros(1)
        torch.save(bad, self.path)
        with self.assertRaises(ValueError):
            self.initialize()

    def test_frozen_teacher_bridge_rng_and_targets(self):
        rng = torch.get_rng_state().clone()
        teacher = load_frozen_teacher(self.path, self.contract, self.spec, 'cpu', torch.float32)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(teacher.training)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        torch.testing.assert_close(teacher(self.current, self.future, self.state, self.task, 8),
                                   self.posterior(self.current, self.future, self.state, self.task, 8))
        bad = copy.deepcopy(self.contract)
        bad['stats_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            load_frozen_teacher(self.path, bad, self.spec, 'cpu', torch.float32)

    def test_bfloat16_forward_backward(self):
        self.action.to(dtype=torch.bfloat16).keep_control_gates_fp32()
        self.posterior.to(dtype=torch.bfloat16)
        current, future, state, task, actions = [x.to(torch.bfloat16) for x in
            (self.current, self.future, self.state, self.task, self.actions)]
        # Match the trainer's autocast context: sinusoidal time embeddings are
        # intentionally evaluated in float32 in the legacy action model.
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, info = planwam_flow_loss(self.action, current, future, [current], state,
                                           task, actions, 'student', teacher=self.posterior, horizon=8)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(self.action.intent_pool.slots.grad.float().norm(), 0)
        self.assertEqual(info['target_norm'] > 0, True)

    def test_student_causality_single_token_gradients_and_save(self):
        # Activate initially zero DiT gates/head to exercise the complete graph.
        torch.nn.init.normal_(self.action.blocks[0].adaLN_modulation[-1].weight, std=0.05)
        torch.nn.init.normal_(self.action.output_proj.weight, std=0.05)
        teacher_before = {k: v.clone() for k, v in self.posterior.state_dict().items()}
        noise, tau = torch.randn_like(self.actions), torch.tensor([.2, .8])
        z = self.action.predict_global_intent(self.current, self.state, self.task, 8)
        self.assertEqual(tuple(z.shape), (2, 32))
        token_shapes = []
        hook = self.action.blocks[0].register_forward_pre_hook(lambda m, args: token_shapes.append(args[0].shape))
        def call(stage, future):
            return planwam_flow_loss(self.action, self.current, future, [self.current], self.state,
                                     self.task, self.actions, stage, teacher=self.posterior,
                                     horizon=8, lambda_kd=.1, noise=noise, flow_time=tau)
        loss, info = call('student', self.future)
        _, changed = call('student', self.future + torch.randn_like(self.future))
        self.assertEqual(info['loss_action'], changed['loss_action'])
        self.assertNotEqual(info['loss_kd'], changed['loss_kd'])
        loss.backward()
        params = dict(self.action.named_parameters())
        for key in ['intent_pool.slots', 'intent_pool.read.in_proj_weight',
                    'intent_pool.interactions.0.self_attn.in_proj_weight',
                    'intent_pool.summary_query', 'intent_predictor.predictor.0.weight',
                    'intent_to_token.1.weight', 'output_proj.weight']:
            g = params[key].grad
            self.assertIsNotNone(g, key)
            self.assertTrue(torch.isfinite(g).all() and g.norm() > 0, key)
        self.assertTrue(all(torch.equal(v, teacher_before[k]) for k, v in self.posterior.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in self.posterior.parameters()))
        self.action.zero_grad(set_to_none=True)
        no_kd, no_kd_info = call('action_only', None)
        self.assertEqual(no_kd_info['loss_kd'], 0)
        no_kd.backward()
        self.assertGreater(params['intent_pool.slots'].grad.norm(), 0)
        # DiT still receives exactly the same number of tokens as the old pool.
        old = VLAModel(**dict(self.kw, intent_pooling='task_attention'))
        old_hook = old.blocks[0].register_forward_pre_hook(lambda m, args: token_shapes.append(args[0].shape))
        old_z = old.predict_global_intent(self.current, self.state, self.task, 8)
        old(tau, noisy_actions=noise, dino_features_list=[self.current], qpos_history=self.state,
            task_cond=self.task, global_intent=old_z)
        self.assertEqual(token_shapes[0], token_shapes[-1])
        hook.remove(); old_hook.remove()
        path = Path(self.tmp.name) / 'student.pt'
        torch.save(self.action.state_dict(), path)
        reloaded = VLAModel(**self.kw)
        reloaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
        torch.testing.assert_close(z, reloaded.predict_global_intent(self.current, self.state, self.task, 8))
        with self.assertRaises(RuntimeError):
            old.load_state_dict(reloaded.state_dict(), strict=True)

    def test_optimizer_covers_new_pool_and_legacy_default_is_unchanged(self):
        report = self.initialize()
        train = OmegaConf.create(dict(learning_rate=2e-4, lr_min=5e-5, betas=[.9,.99], weight_decay=.01))
        groups = OmegaConf.create(dict(inherited_lr=2e-5, inherited_lr_min=5e-6,
                                       predictor_lr=2e-4, predictor_lr_min=5e-5))
        _, _, policy = build_transfer_optimizer(self.action, train, groups, report, 10)
        new_names = policy['groups'][1]['parameter_names']
        self.assertIn('intent_pool.slots', new_names)
        self.assertIn('intent_pool.summary_query', new_names)
        # Defaults do not introduce parameters or consume RNG in the old path.
        kw = dict(self.kw, intent_pooling='task_attention')
        torch.manual_seed(12); a = VLAModel(**kw)
        torch.manual_seed(12); b = VLAModel(**dict(kw, intent_pool_num_queries=4, intent_pool_interaction_depth=1))
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))

    def test_configs_change_only_pool_and_kd(self):
        r = ROOT / 'metaworld_vla/configs'
        c = OmegaConf.load(r / 'metaworld_mt50_easy_planwam_latepool_q4_action_only_uniformlr.yaml')
        d = OmegaConf.load(r / 'metaworld_mt50_easy_planwam_latepool_q4_student_uniformlr.yaml')
        self.assertEqual(validate_config(c), 'action_only')
        self.assertEqual(validate_config(d), 'student')
        for cfg, name in [(c, 'action_init_action_only_uniformlr'), (d, 'action_init_student_uniformlr')]:
            old = OmegaConf.to_container(OmegaConf.load(r / f'metaworld_mt50_easy_planwam_{name}.yaml'), resolve=True)
            new = OmegaConf.to_container(cfg, resolve=True)
            new['model'] = teacher_input_contract(dict(model=new['model']))['model']
            self.assertEqual(new, old)
        d.planwam.stage = 'teacher'
        with self.assertRaises(ValueError):
            validate_config(d)

    def test_uniform_optimizer_resume_and_source_audit(self):
        report = self.initialize()
        noise, tau = torch.randn_like(self.actions), torch.tensor([.2, .8])
        def make_opt(action):
            opt = torch.optim.AdamW([p for p in action.parameters() if p.requires_grad], lr=2e-4)
            return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4, eta_min=5e-5)
        def step(action, opt, sched):
            opt.zero_grad(set_to_none=True)
            loss, _ = planwam_flow_loss(action, self.current, self.future, [self.current],
                                       self.state, self.task, self.actions, 'student',
                                       teacher=self.posterior, noise=noise, flow_time=tau)
            loss.backward(); opt.step(); sched.step()
        opt, sched = make_opt(self.action)
        step(self.action, opt, sched)
        saved = copy.deepcopy((self.action.state_dict(), opt.state_dict(), sched.state_dict()))
        step(self.action, opt, sched)
        resumed = VLAModel(**self.kw)
        before = {k: v.clone() for k, v in resumed.state_dict().items()}
        audited = initialize_action_from_teacher(resumed, self.path, self.contract,
                                                 self.spec, VERSION, apply_weights=False)
        self.assertEqual(report, audited)
        self.assertTrue(all(torch.equal(v, resumed.state_dict()[k]) for k, v in before.items()))
        resumed.load_state_dict(saved[0], strict=True)
        opt2, sched2 = make_opt(resumed)
        opt2.load_state_dict(saved[1]); sched2.load_state_dict(saved[2])
        step(resumed, opt2, sched2)
        self.assertEqual(sched.get_last_lr(), sched2.get_last_lr())
        self.assertTrue(all(torch.equal(v, resumed.state_dict()[k]) for k, v in self.action.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
