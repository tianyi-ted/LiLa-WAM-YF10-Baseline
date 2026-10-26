"""Capture actual DINO forward inputs for native/YF data and native evaluation.

No training, checkpoints, or datasets are modified. The evaluation probe stops
at the DINO forward pre-hook, before action generation. YF online evaluation
remains uncalibrated; only its offline dataset inputs are captured.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.model_runner import ModelFactory, VLAWrapper
from dataset import IMAGENET_MEAN, IMAGENET_STD
from dataset_native import NativeMetaWorldDataset
from dataset_yf import YFMetaWorldDataset
from episode_io import episode_keys
from camera_config import apply_profile, load_profile, render_rgb
from eval import Policy


class CapturedInput(Exception):
    """Stop immediately after recording the real visual encoder input."""


def reconstruct(tensor):
    x = tensor.detach().cpu().float()[0].permute(1, 2, 0).numpy()
    x = (x * IMAGENET_STD + IMAGENET_MEAN) * 255
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-config', default='metaworld_vla/configs/metaworld_mt50_corner2_480_50ep_b4_h512.yaml')
    p.add_argument('--yf-config', default='metaworld_vla/configs/metaworld_yf_mt50_corner_b4_h512.yaml')
    p.add_argument('--task', default='reach-v3')
    p.add_argument('--output-dir', type=Path, default=Path('eval_result/vision_input_audit'))
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    torch.set_num_threads(2)
    dtype = getattr(torch, args.dtype)
    device = torch.device('cpu')
    cfg = OmegaConf.load(args.native_config)
    yf_cfg = OmegaConf.load(args.yf_config)
    vision, _, regs, _ = ModelFactory.create_vision_encoder(
        cfg.model.vision_encoder.checkpoint_path, dtype=dtype, device=str(device))
    # Only the wrapper's vision/state preprocessing is needed for this probe.
    # No action weights/checkpoint are needed: the hook stops before inference.
    wrapper = VLAWrapper.__new__(VLAWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.vision_encoder = vision
    wrapper.device, wrapper.dtype = device, dtype
    wrapper.feat_layers = list(cfg.model.vision_encoder.feat_layers)
    wrapper.future_feat_target_layer = int(cfg.model.future_feat.target_layer)
    wrapper.include_cls_register = bool(cfg.model.vision_encoder.include_cls_register)
    wrapper.num_register_tokens = regs
    # State normalization is irrelevant to the captured RGB; these buffers only
    # allow the unchanged Policy.chunk path to reach its vision call.
    wrapper.register_buffer('state_min', torch.full((4,), -1.0))
    wrapper.register_buffer('state_max', torch.full((4,), 1.0))
    tag = ''
    captured = {}
    report = {'task': args.task, 'probe_device': str(device), 'probe_dtype': args.dtype,
              'capture_point': 'vision_encoder.forward pre-hook, after device/dtype conversion',
              'yf_online_evaluation': 'not run; camera pose/assets/orientation uncalibrated',
              'inputs': {}}

    def hook(_module, _inputs, kwargs):
        x = kwargs['pixel_values'].detach().cpu().clone()
        captured[tag] = x
        rgb = reconstruct(x)
        torch.save(x, args.output_dir / f'{tag}.pt')
        Image.fromarray(rgb).save(args.output_dir / f'{tag}.png')
        report['inputs'][tag] = dict(shape=list(x.shape), dtype=str(x.dtype),
                                    min=float(x.min()), max=float(x.max()))
        raise CapturedInput()

    handle = vision.register_forward_pre_hook(hook, with_kwargs=True)

    def capture(name, callback):
        nonlocal tag
        tag = name
        try:
            callback()
        except CapturedInput:
            return
        raise AssertionError(f'{name}: DINO capture hook was not reached')

    def dataset_probe(config, cls, camera, name):
        ds = cls(str(config.dataset.dataset_dir), [args.task],
                 image_size=tuple(config.dataset.image_size), camera_name=camera,
                 action_chunk_size=int(config.common.action_chunk_size),
                 future_frame_offset=int(config.dataset.future_frame_offset))
        sample = ds[0]
        capture(name, lambda: wrapper.get_vision_features(sample['pixel_values'][None]))
        capture(name + '_future', lambda: wrapper.get_future_target_features(sample['future_pixel_values'][None]))
        path = ds.episodes[0]['path']
        with h5py.File(path, 'r') as f:
            _, _, rgb_key = episode_keys(f, camera)
            payload = bytes(f[rgb_key][0])
            pil = np.asarray(Image.open(io.BytesIO(payload)).convert('RGB'))
            cv_rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(payload, np.uint8), 1), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(pil, cv_rgb)
            Image.fromarray(pil).save(args.output_dir / f'{name}_stored.png')
            expected = cv2.resize(pil, tuple(config.dataset.image_size), interpolation=cv2.INTER_LINEAR)
            recovered = reconstruct(captured[name])
            err = np.abs(recovered.astype(float) - expected.astype(float)).mean()
            swap_err = np.abs(recovered.astype(float) - expected[..., ::-1].astype(float)).mean()
            assert err < 1.0, (name, err)
            report['inputs'][name].update(episode=path, camera=camera,
                stored_shape=list(pil.shape), PIL_OpenCV_RGB_equal=True,
                mean_pixel_error_vs_stored_RGB=err, mean_pixel_error_vs_swapped_RB=swap_err)
            seed = int(f.attrs.get('seed', 0))
        return seed

    try:
        seed = dataset_probe(cfg, NativeMetaWorldDataset, str(cfg.dataset.camera_names[0]), 'native_train')
        for camera in ('corner', 'behind'):
            dataset_probe(yf_cfg, YFMetaWorldDataset, camera, f'yf_{camera}_train')
        # Actual native eval.py preprocessing, stopped at the DINO entry. The
        # seed is taken from the native dataset to compare matching reset views.
        import gymnasium as gym
        import metaworld  # noqa: F401
        profile = load_profile(cfg.dataset.get('camera_profile', None))
        size = cfg.dataset.get('render_size', cfg.dataset.image_size)
        env = gym.make('Meta-World/MT1', env_name=args.task, seed=seed,
                       render_mode='rgb_array', camera_name=profile['slot'] if profile else str(cfg.dataset.camera_names[0]),
                       width=int(size[0]), height=int(size[1]))
        try:
            obs, _ = env.reset(seed=seed)
            apply_profile(env, profile)
            raw = env.render()
            Image.fromarray(raw).save(args.output_dir / 'native_eval_raw.png')
            rgb = render_rgb(env, profile)
            Image.fromarray(rgb).save(args.output_dir / 'native_eval_oriented.png')
            policy = Policy.__new__(Policy)
            policy.cfg, policy.device, policy.dtype, policy.wrapper = cfg, device, dtype, wrapper
            capture('native_eval', lambda: policy.chunk(obs, rgb))
            expected = cv2.resize(rgb, tuple(cfg.dataset.image_size), interpolation=cv2.INTER_LINEAR)
            err = np.abs(reconstruct(captured['native_eval']).astype(float) - expected.astype(float)).mean()
            assert err < 1.0
            report['inputs']['native_eval'].update(seed=seed, mean_pixel_error_vs_render_RGB=err)
            diff = np.abs(reconstruct(captured['native_train']).astype(float) - reconstruct(captured['native_eval']).astype(float)).mean()
            report['native_same_seed_train_eval_mean_pixel_difference'] = diff
        finally:
            env.close()
    finally:
        handle.remove()
    # Display-only hypothetical RB swap: never fed to the model.
    for camera in ('corner', 'behind'):
        rgb = reconstruct(captured[f'yf_{camera}_train'])
        Image.fromarray(rgb[..., ::-1]).save(args.output_dir / f'yf_{camera}_hypothetical_RB_swap.png')
    rows = [
        ('Native: stored / DINO train / DINO eval', ['native_train_stored', 'native_train', 'native_eval']),
        ('YF corner: stored / DINO train / hypothetical R-B swap', ['yf_corner_train_stored', 'yf_corner_train', 'yf_corner_hypothetical_RB_swap']),
        ('YF behind: stored / DINO train / hypothetical R-B swap', ['yf_behind_train_stored', 'yf_behind_train', 'yf_behind_hypothetical_RB_swap']),
        ('Future target inputs: native / YF corner / YF behind', ['native_train_future', 'yf_corner_train_future', 'yf_behind_train_future']),
    ]
    sheet = Image.new('RGB', (3 * 280, len(rows) * 280), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (title, names) in enumerate(rows):
        draw.text((5, i * 280 + 4), title, fill='black')
        for j, name in enumerate(names):
            im = Image.open(args.output_dir / f'{name}.png').convert('RGB').resize((256, 256))
            sheet.paste(im, (j * 280, i * 280 + 22))
    sheet.save(args.output_dir / 'comparison.png')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Artifacts: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
