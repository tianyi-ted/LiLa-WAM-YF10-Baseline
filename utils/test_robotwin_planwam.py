"""CPU RoboTwin PlanWAM port regression, with real HDF5 reader and tiny models."""
import copy
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_planwam_robotwin import (build_dataset, validate_config, input_contract,
                           deployment_fields, image_rotation, VERSION, main)
from robotwin_planwam_transfer import initialize_action_from_teacher
from metaworld_vla import train_planwam as meta_trainer
from dataloader.dataset import create_dataset
from models.model_runner import ModelFactory
from models.planwam_g2 import FutureDrivenPosterior, planwam_flow_loss


class RobotWinPlanWAMTest(unittest.TestCase):
    def test_training_entrypoints_do_not_import_each_other(self):
        for path, forbidden in (
            (ROOT / 'train_planwam_robotwin.py', ('metaworld_vla', 'train_planwam', 'planwam_transfer')),
            (ROOT / 'robotwin_planwam_transfer.py', ('metaworld_vla', 'train_planwam', 'planwam_transfer')),
            (ROOT / 'metaworld_vla/train_planwam.py', ('train_planwam_robotwin', 'robotwin_planwam_transfer', 'dataloader')),
        ):
            tree = ast.parse(path.read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imports.append(node.module or '')
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
            self.assertFalse(any(name == item or name.startswith(item + '.')
                                 for name in imports for item in forbidden), (path, imports))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = OmegaConf.load(ROOT / 'configs/robotwin_yf10_planwam_teacher.yaml')
        self.cfg.dataset.dataset_dir = str(self.root / 'data')
        self.cfg.dataset.task_cond_dir = str(self.root / 'vtt')
        self.cfg.dataset.task_names = ['example']
        self.cfg.dataset.image_size = [64, 48]
        vtt = self.root / 'vtt/example'
        vtt.mkdir(parents=True)
        np.save(vtt / 'task_cond.npy', np.linspace(-1, 1, 32, dtype=np.float32))
        (vtt.parent / 'meta.json').write_text('{}')
        self.stats = self.root / 'stats.json'
        self.stats.write_text('{}')
        ep = self.root / 'data/example/demo_clean/data/episode0.hdf5'
        ep.parent.mkdir(parents=True)
        with h5py.File(ep, 'w') as f:
            f['joint_action/vector'] = np.arange(8 * 14, dtype=np.float32).reshape(8, 14)
            f['endpose/left_endpose'] = np.zeros((8, 7), np.float32)
            f['endpose/right_endpose'] = np.ones((8, 7), np.float32)
            f['endpose/left_gripper'] = np.arange(8, dtype=np.float32)
            f['endpose/right_gripper'] = np.zeros(8, np.float32)
            rgb = f.create_dataset('observation/head_camera/rgb', (8,),
                                   dtype=h5py.vlen_dtype(np.dtype('uint8')))
            for i in range(8):
                image = np.full((48, 64, 3), i * 20, np.uint8)
                image[:24, :32, 0] = 255
                ok, encoded = cv2.imencode('.jpg', image)
                self.assertTrue(ok)
                rgb[i] = encoded.reshape(-1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_protocol_configs_and_eval_only_change(self):
        teacher = OmegaConf.load(ROOT / 'configs/robotwin_yf10_planwam_teacher.yaml')
        student = OmegaConf.load(ROOT / 'configs/robotwin_yf10_planwam_c_uniform.yaml')
        self.assertEqual(validate_config(teacher), 'teacher')
        self.assertEqual(validate_config(student), 'action_only')
        for field in ('model', 'dataset', 'training', 'common'):
            self.assertEqual(teacher[field], student[field])
        self.assertFalse(student.planwam.grouped_lr.enabled)
        self.assertEqual(student.training.time_sampler, 'logit_normal')
        self.assertEqual(student.training.learning_rate, 2e-4)
        self.assertEqual(student.training.lr_min, 5e-5)
        distilled = OmegaConf.load(ROOT / 'configs/robotwin_yf10_planwam_d_uniform.yaml')
        self.assertEqual(validate_config(distilled), 'student')
        self.assertEqual(distilled.planwam.lambda_kd, 0.1)
        distilled.planwam.stage = 'action_only'
        distilled.planwam.lambda_kd = 0.0
        self.assertEqual(distilled, student)
        g2 = OmegaConf.load(ROOT / 'configs/robotwin_yf10_easy_g2.yaml')
        reuse = OmegaConf.load(ROOT / 'configs/robotwin_yf10_easy_g2_reuse_eval.yaml')
        self.assertTrue(reuse.inference.action_reuse.enabled)
        reuse.inference.action_reuse.enabled = False
        self.assertEqual(g2, reuse)
        bad = copy.deepcopy(student)
        bad.common.action_dim = 4
        with self.assertRaises(ValueError):
            validate_config(bad)
        bad = copy.deepcopy(student)
        bad.dataset.strict_loading = False
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_original_reader_pixels_state_future_and_padding_are_unchanged(self):
        original = create_dataset(self.cfg)
        teacher, collate = build_dataset(self.cfg, 'teacher')
        causal, _ = build_dataset(self.cfg, 'action_only')
        distilled, _ = build_dataset(self.cfg, 'student')
        self.assertEqual(len(teacher), 8)
        self.assertFalse(causal.use_future_feat)
        self.assertTrue(distilled.use_future_feat)
        torch.testing.assert_close(distilled[0]['future_pixel_values'], teacher[0]['future_pixel_values'])
        for anchor in (0, 3, 7):
            expected, actual, no_future = original[anchor], teacher[anchor], causal[anchor]
            for key, value in expected.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value, actual[key], atol=0, rtol=0)
                    if key != 'future_pixel_values':
                        torch.testing.assert_close(value, no_future[key], atol=0, rtol=0)
            self.assertNotIn('future_pixel_values', no_future)
        batch = collate([teacher[0], teacher[7]])
        self.assertEqual(tuple(batch['state'].shape), (2, 1, 16))
        self.assertEqual(tuple(batch['action_sequence'].shape), (2, 32, 14))
        torch.testing.assert_close(batch['action_sequence'][1, -1], batch['action_sequence'][1, 0])
        self.assertEqual(image_rotation(self.cfg), 0)

    def test_contract_is_platform_bound_and_teacher_student_inputs_match(self):
        contract = input_contract(self.cfg, self.stats)
        student = copy.deepcopy(self.cfg)
        student.planwam.stage = 'action_only'
        self.assertEqual(contract, input_contract(student, self.stats))
        self.assertEqual(contract['image_rotation'], 0)
        self.assertEqual(contract['common']['action_dim'], 14)
        meta = OmegaConf.load(ROOT / 'metaworld_vla/configs/metaworld_mt50_easy_planwam_teacher.yaml')
        self.assertEqual(meta_trainer.image_rotation(meta), 180)
        with self.assertRaisesRegex(ValueError, 'RoboTwin-only'):
            validate_config(meta)
        with self.assertRaisesRegex(ValueError, 'MetaWorld-only'):
            meta_trainer.validate_config(self.cfg)

    def test_14d_teacher_transfer_fm_gradients_and_strict_deployment_reload(self):
        cfg = self.cfg
        cfg.model.action_expert.depth = 1
        cfg.model.action_expert.hidden_size = 32
        cfg.model.action_expert.num_heads = 4
        cfg.model.action_expert.vlm_adapter_num_queries = 4
        cfg.model.action_expert.adapter_depth = 1
        cfg.model.global_intent.hidden_size = 32
        cfg.model.global_intent.pool_hidden_size = 32
        def model():
            return ModelFactory.create_action_model(cfg, 32, 3, 32)
        teacher, student = model(), model()
        # Activate zero-init output so a one-step test can inspect upstream gradients.
        torch.nn.init.normal_(teacher.output_proj.weight, std=0.05)
        torch.nn.init.normal_(teacher.blocks[0].adaLN_modulation[-1].weight, std=0.05)
        spec = dict(vision_dim=32, state_dim=16, task_dim=32,
                    grid_size=[3, 4], hidden_dim=32, num_heads=4, horizon_dim=32)
        posterior = FutureDrivenPosterior(**spec)
        features = [torch.randn(2, 13, 32) for _ in range(3)]
        current, future = features[-1][:, 1:], torch.randn(2, 12, 32)
        state, task, actions = torch.randn(2, 1, 16), torch.randn(2, 32), torch.randn(2, 32, 14)
        loss, _ = planwam_flow_loss(teacher, current, future, features, state, task, actions,
                                    'teacher', posterior=posterior, horizon=32)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(posterior.output[-1].weight.grad.norm(), 0)
        self.assertIsNone(teacher.intent_pool.query[1].weight.grad)
        contract = input_contract(cfg, self.stats)
        path = self.root / 'teacher.pt'
        torch.save(dict(planwam_stage='teacher', planwam_version=VERSION, epoch=40,
                        epoch_complete=True, input_contract=contract, posterior_spec=spec,
                        model_state=teacher.state_dict()), path)
        report = initialize_action_from_teacher(student, path, contract, spec, VERSION)
        self.assertTrue(report['fresh_predictor_state_keys'])
        loss, info = planwam_flow_loss(student, current, None, features, state, task, actions,
                                       'action_only', horizon=32)
        self.assertEqual(info['loss_kd'], 0)
        loss.backward()
        self.assertGreater(student.intent_pool.query[1].weight.grad.norm(), 0)
        opt = torch.optim.AdamW(student.parameters(), lr=2e-4)
        self.assertEqual(len(opt.param_groups), 1)
        opt.step()
        fields = deployment_fields(cfg, 'action_only', student.state_dict())
        torch.save(fields, self.root / 'student.pt')
        restored = torch.load(self.root / 'student.pt', weights_only=False)
        deployed = model()
        deployed.load_state_dict(restored['model_state_dict'], strict=True)
        deployed.eval()
        with torch.no_grad():
            latent = deployed.predict_global_intent(current, state, task, horizon=32)
            result = deployed(torch.zeros(2), noisy_actions=actions, dino_features_list=features,
                              qpos_history=state, task_cond=task, global_intent=latent)['final_pred']
        self.assertEqual(tuple(result.shape), (2, 32, 14))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(deployment_fields(cfg, 'teacher', student.state_dict()), {})
        with self.assertRaises(ValueError):
            initialize_action_from_teacher(student, path, dict(contract, image_rotation=180), spec, VERSION)

    def test_complete_training_entrypoints_save_reload_and_resume(self):
        """Exercise the real trainer with tiny DINO stand-in; no GPU or network."""
        class TinyVision(torch.nn.Module):
            def forward(self, pixel_values, **kwargs):
                patches = torch.nn.functional.avg_pool2d(pixel_values, 16).flatten(2).transpose(1, 2)
                patches = patches.mean(-1, keepdim=True).expand(-1, -1, 32)
                full = torch.cat((patches.mean(1, keepdim=True), patches), dim=1)
                return SimpleNamespace(hidden_states=(full,) * 8)
        cfg = self.cfg
        cfg.model.action_expert.depth = 1
        cfg.model.action_expert.hidden_size = 32
        cfg.model.action_expert.num_heads = 4
        cfg.model.action_expert.vlm_adapter_num_queries = 4
        cfg.model.action_expert.adapter_depth = 1
        cfg.model.global_intent.hidden_size = 32
        cfg.model.global_intent.pool_hidden_size = 32
        cfg.planwam.posterior.hidden_size = 32
        cfg.training.batch_size = 2
        cfg.training.epochs = 1
        cfg.training.save_interval_epoch = 1
        cfg.system.num_workers = 0
        self.stats.write_text(json.dumps({'robotwin2': {
            'action': {'min': [0] * 14, 'max': [120] * 14},
            'state': {'min': [0] * 16, 'max': [8] * 16}}}))
        config_file = self.root / 'config.yaml'
        teacher_run, student_run = self.root / 'teacher', self.root / 'student'
        def run(config, output, extra=()):
            OmegaConf.save(config, config_file)
            args = ['train_planwam_robotwin.py', '--config', str(config_file), '--stats', str(self.stats),
                    '--save-dir', str(output), *extra]
            with patch('sys.argv', args), patch('torch.cuda.is_available', return_value=False), \
                    patch.object(ModelFactory, 'create_vision_encoder', return_value=(TinyVision(), 32, 0, 16)):
                main()
        run(cfg, teacher_run)
        teacher_path = teacher_run / 'checkpoint_epoch_1.pt'
        teacher = torch.load(teacher_path, weights_only=False)
        self.assertTrue(teacher['epoch_complete'])
        self.assertEqual(teacher['image_rotation_degrees'], 0)
        self.assertNotIn('model_state_dict', teacher)
        cfg.planwam.stage = 'action_only'
        cfg.planwam.action_init = {'enabled': True, 'checkpoint': str(teacher_path)}
        cfg.planwam.grouped_lr = {'enabled': False}
        run(cfg, student_run, ['--init-teacher-action', str(teacher_path)])
        student_path = student_run / 'checkpoint_epoch_1.pt'
        student = torch.load(student_path, weights_only=False)
        self.assertTrue(student['epoch_complete'])
        self.assertEqual(student['global_step'], 4)
        self.assertEqual(len(student['optimizer_state']['param_groups']), 1)
        self.assertAlmostEqual(student['optimizer_state']['param_groups'][0]['lr'], 5e-5)
        self.assertEqual(student['optimizer_policy']['kind'], 'legacy_uniform_cosine')
        restored = ModelFactory.create_action_model(cfg, 32, 3, 32)
        restored.load_state_dict(student['model_state_dict'], strict=True)
        self.assertTrue((student_run / 'action_init_report.json').is_file())
        from robotwin_infer import RobotWinInference
        with patch.object(ModelFactory, 'create_vision_encoder', return_value=(TinyVision(), 32, 0, 16)):
            engine = RobotWinInference(str(student_run / 'config.yaml'), str(student_path), str(self.stats),
                                       device='cpu', dtype=torch.float32, task_name='example')
            observation = {'endpose': {'left_endpose': np.zeros(7), 'left_gripper': 0.,
                                      'right_endpose': np.zeros(7), 'right_gripper': 0.},
                           'observation': {'head_camera': {'rgb': np.zeros((48, 64, 3), np.uint8)}}}
            for _ in range(17):
                action = engine.step(observation)
                self.assertEqual(action.shape, (14,))
                self.assertTrue(np.isfinite(action).all())
            engine.reset()
            with self.assertRaisesRegex(ValueError, 'cannot be evaluated online'):
                RobotWinInference(str(teacher_run / 'config.yaml'), str(teacher_path), str(self.stats),
                                   device='cpu', dtype=torch.float32)
        run(cfg, self.root / 'resumed', ['--resume', str(student_path)])
        # D-uniform starts from the SAME teacher, not from the trained C-uniform.
        cfg.planwam.stage = 'student'
        cfg.planwam.lambda_kd = 0.1
        d_run = self.root / 'd_uniform'
        teacher_bytes = teacher_path.read_bytes()
        run(cfg, d_run, ['--init-teacher-action', str(teacher_path),
                         '--teacher-checkpoint', str(teacher_path)])
        distilled_path = d_run / 'checkpoint_epoch_1.pt'
        distilled = torch.load(distilled_path, weights_only=False)
        self.assertEqual(teacher_bytes, teacher_path.read_bytes())
        self.assertEqual(distilled['planwam_stage'], 'student')
        self.assertEqual(distilled['planwam_protocol']['lambda_kd'], 0.1)
        self.assertEqual(distilled['planwam_protocol']['teacher_sha256'],
                         distilled['action_initialization']['source_sha256'])
        self.assertEqual(len(distilled['optimizer_state']['param_groups']), 1)
        self.assertEqual(distilled['optimizer_policy']['kind'], 'legacy_uniform_cosine')
        self.assertIsNone(distilled['posterior_state'])  # Frozen teacher is not part of the student.
        with patch.object(ModelFactory, 'create_vision_encoder', return_value=(TinyVision(), 32, 0, 16)):
            deployed = RobotWinInference(str(d_run / 'config.yaml'), str(distilled_path), str(self.stats),
                                         device='cpu', dtype=torch.float32, task_name='example')
            self.assertTrue(np.isfinite(deployed.step(observation)).all())
        run(cfg, self.root / 'd_resumed', ['--resume', str(distilled_path),
                                         '--teacher-checkpoint', str(teacher_path)])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
