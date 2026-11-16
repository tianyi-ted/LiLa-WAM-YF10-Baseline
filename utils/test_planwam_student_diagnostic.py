"""CPU-only regression tests for paired offline diagnostics (no pretrained DINO)."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'metaworld_vla'))
from diagnose_planwam_student import (
    chunk_metrics, paired_generate, paired_velocities, select_anchors, summarize,
    evaluate_teacher_pairs, summarize_matrix,
)
from models.planwam_g2 import planwam_flow_loss
from models.vla_model_fm import VLAModel


class ConstantField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, t, noisy_actions, dino_features_list, qpos_history, task_cond, global_intent):
        self.calls.append((global_intent.clone(), noisy_actions.clone()))
        return {'final_pred': global_intent[:, :4, None].transpose(1, 2).expand_as(noisy_actions)}


class StudentDiagnosticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.noise = torch.randn(2, 8, 4)
        self.actions = torch.randn_like(self.noise)
        self.features = [torch.randn(2, 12, 32)]
        self.state, self.task = torch.randn(2, 1, 4), torch.randn(2, 32)

    def test_same_noise_and_only_latent_changed(self):
        action = ConstantField()
        original = self.noise.clone()
        latents = {'ss': torch.zeros(2, 32), 'st': torch.ones(2, 32)}
        metrics = paired_velocities(action, self.features, self.state, self.task,
                                    self.actions, self.noise, latents, (0.1, 0.5))
        for first, second in ((0, 1), (2, 3)):
            torch.testing.assert_close(action.calls[first][1], action.calls[second][1])
        expected = (self.actions - self.noise).square().mean((1, 2))
        torch.testing.assert_close(metrics['ss/velocity_mse_tau_mean'], expected)
        self.assertFalse(torch.equal(metrics['ss/velocity_mse_tau_mean'], metrics['st/velocity_mse_tau_mean']))
        torch.testing.assert_close(original, self.noise)

    def test_euler_known_field_and_no_mutation(self):
        action = ConstantField()
        latent = torch.ones(2, 32)
        original = self.noise.clone()
        result = paired_generate(action, self.features, self.state, self.task,
                                 self.noise, {'ss': latent, 'st': latent}, 10)
        torch.testing.assert_close(result['ss'], self.noise + 1, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result['ss'], result['st'])
        torch.testing.assert_close(original, self.noise)

    def test_real_small_dit_matches_training_velocity_loss(self):
        action = VLAModel(action_dim=4, proprio_dim=4, hidden_dim=32, depth=1,
                          action_len=8, num_heads=4, dino_feat_dims=(32,),
                          vlm_num_queries=4, adapter_depth=1, task_cond_dim=32,
                          use_global_intent=True, inject_global_intent=True,
                          intent_hidden_dim=32, intent_use_gate=False, inject_task_cond=False,
                          intent_pooling='task_attention', intent_pool_hidden_dim=32,
                          intent_pool_grid_size=(3, 4)).eval()
        torch.nn.init.normal_(action.blocks[0].adaLN_modulation[-1].weight, std=0.05)
        torch.nn.init.normal_(action.output_proj.weight, std=0.05)
        before = {k: v.clone() for k, v in action.state_dict().items()}
        with torch.no_grad():
            z = action.predict_global_intent(self.features[0], self.state, self.task, horizon=8)
            metrics = paired_velocities(action, self.features, self.state, self.task,
                                        self.actions, self.noise, {'ss': z, 'st': z}, (0.5,))
            _, info = planwam_flow_loss(action, self.features[0], None, self.features,
                                       self.state, self.task, self.actions, 'action_only',
                                       noise=self.noise, flow_time=torch.full((2,), 0.5))
            self.assertAlmostEqual(metrics['ss/velocity_mse_tau_0.5'].mean().item(), info['loss_action'], places=6)
            torch.testing.assert_close(metrics['ss/velocity_mse_tau_0.5'], metrics['st/velocity_mse_tau_0.5'])
            generated = paired_generate(action, self.features, self.state, self.task,
                                        self.noise, {'ss': z, 'st': z + torch.randn_like(z)}, 3)
        self.assertFalse(torch.allclose(generated['ss'], generated['st']))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in action.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in action.parameters()))

    def test_padding_prefix_and_per_step_metrics(self):
        target = torch.zeros(2, 8, 4)
        pred = target.clone()
        pred[0, 4:] = 2
        pred[1, :, :3] = 1
        metrics = chunk_metrics({'ss': pred, 'st': target}, target, [4, 8], 4, lambda x: x)
        torch.testing.assert_close(metrics['ss/chunk_mse_normalized'], torch.tensor([2., 0.75]))
        torch.testing.assert_close(metrics['ss/recorded_steps_mse_normalized'], torch.tensor([0., 0.75]))
        torch.testing.assert_close(metrics['ss/prefix_mse_normalized'], torch.tensor([0., 0.75]))
        self.assertEqual(metrics['ss/command_clipped_gripper_mse'][0].item(), 0.5)

    def test_paired_summary_direction(self):
        rows = [{'metrics': {'ss/mse': 2., 'st/mse': 1.}},
                {'metrics': {'ss/mse': 4., 'st/mse': 2.}}]
        item = summarize(rows)['paired']['mse']
        self.assertEqual(item['delta_st_minus_ss'], -1.5)
        self.assertEqual(item['relative_mean_reduction'], 0.5)
        self.assertEqual(item['st_better_fraction'], 1.)
        self.assertEqual(summarize([])['samples'], 0)

    def test_teacher_pairs_use_student_latent_and_same_noise(self):
        action = ConstantField().eval()
        ss, tt = torch.zeros(2, 32), torch.ones(2, 32)
        original = self.noise.clone()
        metrics, generated = evaluate_teacher_pairs(
            action, self.features, self.state, self.task, self.actions, self.noise,
            ss, tt, (0.5,), 3)
        self.assertEqual(set(generated), {'ts', 'tt'})
        self.assertIn('ts/velocity_mse_tau_mean', metrics)
        self.assertIn('tt/velocity_mse_tau_mean', metrics)
        torch.testing.assert_close(action.calls[0][0], ss)
        torch.testing.assert_close(action.calls[1][0], tt)
        torch.testing.assert_close(action.calls[0][1], action.calls[1][1])
        torch.testing.assert_close(generated['ts'], self.noise)
        torch.testing.assert_close(generated['tt'], self.noise + 1)
        torch.testing.assert_close(self.noise, original)
        # A teacher's own legacy causal predictor is never called here.
        self.assertFalse(hasattr(action, 'predict_global_intent'))

    def test_full_matrix_summary_preserves_legacy_and_direction(self):
        rows = [{'metrics': {'ss/mse': 2., 'st/mse': 4., 'ts/mse': 3., 'tt/mse': 1.}}]
        summary = summarize_matrix(rows)
        self.assertEqual(summary['paired'], summarize(rows)['paired'])
        for name, delta in [('st_vs_ss', 2.), ('ts_vs_tt', 2.), ('tt_vs_ss', -1.), ('ts_vs_ss', 1.)]:
            self.assertEqual(summary['comparisons'][name]['metrics']['mse']['delta_candidate_minus_reference'], delta)
        self.assertEqual(summarize_matrix([])['comparisons']['tt_vs_ss']['metrics'], {})

    def test_task_balanced_unique_sampling(self):
        class Dataset:
            episodes = [{'task': 'a'}, {'task': 'b'}]
            indices = [(ep, i) for ep in (0, 1) for i in range(5)]
        chosen = select_anchors(Dataset(), ['a', 'b'], 3, 0)
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertEqual(sum(i < 5 for i in chosen), 3)
        self.assertEqual(chosen, select_anchors(Dataset(), ['a', 'b'], 3, 0))
        with self.assertRaises(ValueError):
            select_anchors(Dataset(), ['a'], 6, 0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
