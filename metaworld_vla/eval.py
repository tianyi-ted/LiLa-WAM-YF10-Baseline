"""Standalone MT10 evaluator for the parallel Meta-World pipeline."""
from __future__ import annotations
import argparse, json, logging
import os
import random
import time
from pathlib import Path
import sys
import cv2, numpy as np, torch
from omegaconf import OmegaConf

# Keep direct script execution (``python metaworld_vla/eval.py``) working
# without requiring callers to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_runner import ModelFactory, VLAWrapper
from camera_config import load_profile, apply_profile, render_rgb
from image_orientation import IMAGE_ROTATION, check_task_conditions
from dataset import _normalise_rgb
from action_reuse import TailActionPrior
from reuse_options import add_reuse_arguments, validate_reuse_arguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

class Policy:
    def __init__(self, cfg, stats, checkpoint, device, *, action_reuse=False,
                 reuse_sigma=1.0, reuse_strength=1.0, policy_seed=None,
                 profile_generation=False):
        self.cfg, self.device = cfg, torch.device(device)
        self.prior = TailActionPrior(action_reuse, reuse_sigma, reuse_strength)
        self.generator = None
        if policy_seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(policy_seed)
        self.policy_seed = policy_seed
        self.profile_generation = profile_generation
        if not 1 <= int(cfg.common.action_execution_horizon) <= int(cfg.common.action_chunk_size):
            raise ValueError('execution horizon must be between 1 and chunk size')
        if cfg.model.get('spatial_intent', {}).get('enabled', False):
            raise ValueError('Meta-World spatial-intent evaluation is not wired yet')
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        mc, cc = cfg.model, cfg.common
        self.vision, hidden, regs, patch = ModelFactory.create_vision_encoder(mc.vision_encoder.checkpoint_path, self.dtype, str(self.device))
        self.action = ModelFactory.create_action_model(cfg, hidden, len(mc.vision_encoder.feat_layers), task_cond_dim=hidden if mc.get("use_task_cond", False) else None, patch_size=patch, num_register_tokens=regs).to(device=self.device, dtype=self.dtype)
        self.action.keep_control_gates_fp32()
        self.wrapper = VLAWrapper(self.vision, self.action, cfg.training.time_sampler, mc.vision_encoder.feat_layers, bool(mc.vision_encoder.include_cls_register), regs, self.device, self.dtype, stats, None, int(mc.future_feat.target_layer)).to(self.device).eval()
        ck = torch.load(checkpoint, map_location="cpu")
        if ck.get("image_rotation_degrees") != IMAGE_ROTATION:
            raise ValueError("Checkpoint uses old/unknown image orientation. Retrain for rotation=180.")
        state = ck.get("model_state", ck.get("model_state_dict", ck)); self.action.load_state_dict(state, strict=True); self.action.eval()
        self.task_cond = None
        self.queue = []
        self.reset_metrics()
    def reset_metrics(self):
        self.metrics = dict(chunks=0, reused_chunks=0, reused_positions=0,
                            latency_ms=[], boundary_xyz=[], boundary_gripper=[])
    def reset_episode(self):
        self.queue.clear()
        self.prior.reset()
        self.last_sent = None
    def metrics_summary(self):
        def mean(values):
            return float(np.mean(values)) if values else None
        m = self.metrics
        return dict(chunks=m['chunks'], reused_chunks=m['reused_chunks'],
                    reused_positions=m['reused_positions'],
                    generation_ms_mean=mean(m['latency_ms']),
                    generation_ms_p95=float(np.percentile(m['latency_ms'], 95)) if m['latency_ms'] else None,
                    boundary_xyz_l2_mean=mean(m['boundary_xyz']),
                    boundary_gripper_abs_mean=mean(m['boundary_gripper']))
    def set_task(self, task):
        check_task_conditions(self.cfg.dataset.task_cond_dir, self.cfg.dataset.image_size)
        path = Path(self.cfg.dataset.task_cond_dir) / task / "task_cond.npy"
        self.task_cond = torch.from_numpy(np.load(path).astype(np.float32)).to(self.device, self.dtype).unsqueeze(0)
        self.reset_episode()
        self.reset_metrics()
        if self.generator is not None:
            self.generator.manual_seed(self.policy_seed)
    @torch.no_grad()
    def chunk(self, obs, rgb):
        if self.profile_generation and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter() if self.profile_generation else None
        state = torch.from_numpy(np.asarray(obs[:4], np.float32)).to(self.device, self.dtype).unsqueeze(0)
        state = self.wrapper.normalize_state(state).unsqueeze(1)
        pix = _normalise_rgb(cv2.resize(rgb, tuple(self.cfg.dataset.image_size), interpolation=cv2.INTER_LINEAR)).unsqueeze(0).to(self.device, self.dtype)
        feats = self.wrapper.get_vision_features(pix)
        global_intent = None
        if self.action.use_global_intent and self.action.inject_global_intent:
            # Predict once per chunk from current observations only. Reuse
            # across Euler steps; no future image is read during evaluation.
            current = self.wrapper.get_current_intent_features(feats)
            global_intent = self.action.predict_global_intent(
                current, qpos_history=state, task_cond=self.task_cond,
                horizon=self.action.intent_horizon)
        n = int(self.cfg.common.action_chunk_size)
        # Keep the legacy draw exactly unchanged when no separate RNG is requested.
        noise_args = {} if self.generator is None else {'generator': self.generator}
        x = torch.randn(1, n, int(self.cfg.common.action_dim), device=self.device, dtype=self.dtype, **noise_args)
        x = self.prior.source(x)
        steps = torch.linspace(0, 1, int(self.cfg.common.num_inference_steps) + 1, device=self.device, dtype=self.dtype)
        for i in range(len(steps) - 1):
            pred = self.action(t=steps[i].view(1), noisy_actions=x, qpos_history=state, dino_features_list=feats, task_cond=self.task_cond, global_intent=global_intent)["final_pred"]
            x = x + pred * (steps[i + 1] - steps[i])
        self.prior.remember(x)
        actions = self.wrapper.denormalize_action(x)[0].float().cpu().numpy()
        self.metrics['chunks'] += 1
        if self.prior.last_overlap:
            self.metrics['reused_chunks'] += 1
            self.metrics['reused_positions'] += self.prior.last_overlap
        if self.profile_generation:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.metrics['latency_ms'].append((time.perf_counter() - started) * 1000)
        return actions
    def step(self, obs, rgb):
        boundary = not self.queue
        if boundary: self.queue = list(self.chunk(obs, rgb)[:int(self.cfg.common.action_execution_horizon)])
        action = np.clip(self.queue.pop(0), -1, 1)
        if boundary and self.last_sent is not None:
            self.metrics['boundary_xyz'].append(float(np.linalg.norm(action[:3] - self.last_sent[:3])))
            self.metrics['boundary_gripper'].append(float(abs(action[3] - self.last_sent[3])))
        self.last_sent = action.copy()
        self.prior.advance()
        return action

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--stats", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True); p.add_argument("--tasks", nargs="+"); p.add_argument("--test-num", type=int, default=50); p.add_argument("--horizon", type=int, default=500); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda")
    p.add_argument("--viewer", action="store_true", help="show interactive MuJoCo free camera alongside policy evaluation")
    add_reuse_arguments(p)
    a = p.parse_args()
    validate_reuse_arguments(p, a)
    cfg = OmegaConf.load(a.config)
    if a.num_inference_steps is not None:
        cfg.common.num_inference_steps = a.num_inference_steps
    tasks = a.tasks or list(cfg.dataset.task_names); device = a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    if cfg.dataset.get('online_eval_verified', True) is False:
        raise ValueError('External dataset rendering is not calibrated to this simulator. Verify camera pose, robot assets and image orientation before enabling dataset.online_eval_verified.')
    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium as gym
    # Importing Meta-World registers the ``Meta-World/MT1`` Gymnasium
    # namespace. The collector imports policy modules first, but evaluation
    # calls gym.make directly, so registration must be explicit here.
    import metaworld  # noqa: F401
    profile = load_profile(cfg.dataset.get("camera_profile", None))
    # Render at the collection resolution, then resize in Policy.chunk.
    # Older experiment configs without render_size retain their old behavior.
    render_size = tuple(cfg.dataset.get("render_size", cfg.dataset.image_size))
    policy = Policy(cfg, a.stats, a.checkpoint, device,
                    action_reuse=a.action_reuse, reuse_sigma=a.reuse_sigma,
                    reuse_strength=a.reuse_strength, policy_seed=a.policy_seed,
                    profile_generation=a.profile_generation)
    protocol = dict(action_reuse=a.action_reuse, reuse_sigma=a.reuse_sigma,
                    reuse_strength=a.reuse_strength, policy_seed=a.policy_seed,
                    num_inference_steps=int(cfg.common.num_inference_steps),
                    chunk_size=int(cfg.common.action_chunk_size),
                    execution_horizon=int(cfg.common.action_execution_horizon),
                    profile_generation=a.profile_generation)
    log.info('Action generation protocol: %s', protocol)
    result = {"checkpoint": str(Path(a.checkpoint).resolve()), "seed": a.seed, "horizon": a.horizon, "test_num": a.test_num, "rng_protocol": "reset_python_numpy_torch_per_task_v1", "action_generation": protocol, "tasks": {}}
    result["stop_on_success"] = True
    for task in tasks:
        # Task sharding must not change the policy noise stream based on which
        # other tasks a process evaluated first. This also seeds Flow Matching.
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        policy.set_task(task); successes = []
        env = gym.make("Meta-World/MT1", env_name=task, render_mode="rgb_array", camera_name=profile["slot"] if profile else str(cfg.dataset.camera_names[0]), width=int(render_size[0]), height=int(render_size[1]), seed=a.seed)
        viewer = None
        if a.viewer:
            from view_scene import launch_free_viewer
            # Initialize the scene before opening the interactive viewer.
            env.reset(seed=a.seed)
            viewer = launch_free_viewer(env)
        episodes = []
        for ep in range(a.test_num):
            obs, _ = env.reset(seed=a.seed + ep); apply_profile(env, profile); policy.reset_episode(); ok = False
            for _ in range(a.horizon):
                rgb = render_rgb(env, profile); obs, _, term, trunc, info = env.step(policy.step(obs, rgb)); ok |= bool(info.get("success", 0))
                if viewer is not None:
                    if not viewer.is_running():
                        env.close()
                        raise SystemExit("Viewer closed; evaluation stopped without writing partial success rates.")
                    viewer.sync()
                if ok or term or trunc: break
            episodes.append(dict(environment_seed=a.seed + ep, success=int(ok)))
            successes.append(int(ok)); log.info("%s episode %d success=%s", task, ep, ok)
        if viewer is not None:
            viewer.close()
        env.close(); result["tasks"][task] = {"success": int(sum(successes)), "episodes": len(successes), "success_rate": float(np.mean(successes)), "episode_results": episodes, "generation_metrics": policy.metrics_summary()}; log.info("%s success rate %.3f", task, result["tasks"][task]["success_rate"])
    result["overall_success_rate"] = float(np.mean([x["success_rate"] for x in result["tasks"].values()]))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
