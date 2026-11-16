"""CPU tests for isolated PlanWAM branches; no DINO weights required."""
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'metaworld_vla'))
from models.vla_model_fm import VLAModel, fixed_layer_norm
from models.planwam_g2 import FutureDrivenPosterior, freeze_module, planwam_flow_loss
from train_planwam import load_frozen_teacher, validate_config, VERSION


class PlanWAMTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.kw = dict(action_dim=4, proprio_dim=4, hidden_dim=32, depth=1, action_len=8,
                       num_heads=4, dino_feat_dims=(32,), vlm_num_queries=4, adapter_depth=1,
                       task_cond_dim=32, use_global_intent=True, inject_global_intent=True,
                       intent_hidden_dim=32, intent_use_gate=False, inject_task_cond=False,
                       intent_pooling='task_attention', intent_pool_hidden_dim=32,
                       intent_pool_grid_size=(3, 4))
        self.action = VLAModel(**self.kw)
        # Make the originally zero-initialized action gates active for gradient routing tests.
        torch.nn.init.normal_(self.action.blocks[0].adaLN_modulation[-1].weight, std=0.05)
        torch.nn.init.normal_(self.action.output_proj.weight, std=0.05)
        self.spec = dict(vision_dim=32, state_dim=4, task_dim=32, grid_size=[3,4], hidden_dim=32, num_heads=4)
        self.post = FutureDrivenPosterior(**self.spec)
        self.current, self.future = torch.randn(2,12,32), torch.randn(2,12,32)
        self.state, self.task, self.actions = torch.randn(2,1,4), torch.randn(2,32), torch.randn(2,8,4)
        self.noise, self.t = torch.randn_like(self.actions), torch.tensor([0.1,0.5])

    def call(self, stage, **kwargs):
        return planwam_flow_loss(self.action, self.current, self.future, [self.current],
                                 self.state, self.task, self.actions, stage,
                                 noise=self.noise, flow_time=self.t, **kwargs)

    def test_teacher_future_residual_and_action_gradients(self):
        z = self.post(self.current, self.future, self.state, self.task, 8)
        z_wrong = self.post(self.current, self.current, self.state, self.task, 8)
        self.assertFalse(torch.allclose(z, z_wrong))
        loss, info = self.call('teacher', posterior=self.post)
        self.assertEqual(info['loss_kd'], 0)
        loss.backward()
        for name in ('visual.1.weight', 'cross.in_proj_weight', 'summary_query', 'output.1.weight'):
            g = dict(self.post.named_parameters())[name].grad
            self.assertIsNotNone(g, name)
            self.assertTrue(torch.isfinite(g).all() and g.norm()>0, name)
        self.assertIsNone(self.action.intent_predictor.predictor[0].weight.grad)
        old = self.post.summary_query.detach().clone()
        torch.optim.AdamW(list(self.action.parameters())+list(self.post.parameters()), lr=1e-3).step()
        self.assertFalse(torch.equal(old, self.post.summary_query))

    def test_student_is_causal_teacher_is_frozen(self):
        teacher = freeze_module(self.post)
        before = {k:v.clone() for k,v in teacher.state_dict().items()}
        self.future.requires_grad_(True)
        loss, info = self.call('student', teacher=teacher)
        future = self.future
        self.future = torch.randn_like(future)
        _, other = self.call('student', teacher=teacher)
        self.assertEqual(info['loss_action'], other['loss_action'])
        loss.backward()
        self.assertIsNone(future.grad)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        g = self.action.intent_pool.query[1].weight.grad
        self.assertTrue(g is not None and torch.isfinite(g).all() and g.norm()>0)
        torch.optim.AdamW(self.action.parameters(), lr=1e-3).step()
        self.assertTrue(all(torch.equal(before[k],v) for k,v in teacher.state_dict().items()))

    def test_zero_supervision_needs_no_future(self):
        self.future = None
        loss, info = self.call('action_only')
        self.assertEqual(info['loss_kd'], 0)
        loss.backward()
        self.assertGreater(self.action.intent_pool.query[1].weight.grad.norm(), 0)

    def test_mixed_precision_cpu(self):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss, _ = self.call('student', teacher=freeze_module(self.post))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_teacher_and_legacy_student_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'teacher.pt'
            torch.save(dict(planwam_stage='teacher', planwam_version=VERSION,
                            epoch_complete=True, input_contract={'id':1}, posterior_spec=self.spec,
                            posterior_state=self.post.state_dict()), path)
            loaded = load_frozen_teacher(path, {'id':1}, self.spec, 'cpu', torch.float32)
            self.assertFalse(loaded.training)
            self.post.eval()
            with torch.no_grad():
                a = loaded(self.current,self.future,self.state,self.task,8)
                b = self.post(self.current,self.future,self.state,self.task,8)
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            with self.assertRaises(ValueError):
                load_frozen_teacher(path, {'id':2}, self.spec, 'cpu', torch.float32)
        legacy = VLAModel(**self.kw)
        legacy.load_state_dict(self.action.state_dict(), strict=True)

    def test_config_only_protocol_changes(self):
        base = OmegaConf.to_container(OmegaConf.load(ROOT/'metaworld_vla/configs/metaworld_mt50_corner2_easy_g2.yaml'))
        for stage in ('teacher', 'student', 'action_only'):
            cfg = OmegaConf.load(ROOT/f'metaworld_vla/configs/metaworld_mt50_easy_planwam_{stage}.yaml')
            self.assertEqual(validate_config(cfg), stage)
            plain = OmegaConf.to_container(cfg)
            plain.pop('planwam')
            plain['training']['lambda_intent'] = base['training']['lambda_intent']
            self.assertEqual(plain, base)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
