"""Standalone MT10 evaluator for the parallel Meta-World pipeline."""
from __future__ import annotations
import argparse, json, logging
import os
import random
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

class Policy:
    def __init__(self, cfg, stats, checkpoint, device):
        self.cfg, self.device = cfg, torch.device(device)
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
    def set_task(self, task):
        check_task_conditions(self.cfg.dataset.task_cond_dir, self.cfg.dataset.image_size)
        path = Path(self.cfg.dataset.task_cond_dir) / task / "task_cond.npy"
        self.task_cond = torch.from_numpy(np.load(path).astype(np.float32)).to(self.device, self.dtype).unsqueeze(0)
        self.queue.clear()
    @torch.no_grad()
    def chunk(self, obs, rgb):
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
        n = int(self.cfg.common.action_chunk_size); x = torch.randn(1, n, int(self.cfg.common.action_dim), device=self.device, dtype=self.dtype)
        steps = torch.linspace(0, 1, int(self.cfg.common.num_inference_steps) + 1, device=self.device, dtype=self.dtype)
        for i in range(len(steps) - 1):
            pred = self.action(t=steps[i].view(1), noisy_actions=x, qpos_history=state, dino_features_list=feats, task_cond=self.task_cond, global_intent=global_intent)["final_pred"]
            x = x + pred * (steps[i + 1] - steps[i])
        return self.wrapper.denormalize_action(x)[0].float().cpu().numpy()
    def step(self, obs, rgb):
        if not self.queue: self.queue = list(self.chunk(obs, rgb)[:int(self.cfg.common.action_execution_horizon)])
        return np.clip(self.queue.pop(0), -1, 1)

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--stats", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True); p.add_argument("--tasks", nargs="+"); p.add_argument("--test-num", type=int, default=50); p.add_argument("--horizon", type=int, default=500); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cuda")
    p.add_argument("--viewer", action="store_true", help="show interactive MuJoCo free camera alongside policy evaluation")
    a = p.parse_args(); cfg = OmegaConf.load(a.config); tasks = a.tasks or list(cfg.dataset.task_names); device = a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
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
    policy = Policy(cfg, a.stats, a.checkpoint, device); result = {"checkpoint": str(Path(a.checkpoint).resolve()), "seed": a.seed, "horizon": a.horizon, "test_num": a.test_num, "rng_protocol": "reset_python_numpy_torch_per_task_v1", "tasks": {}}
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
        for ep in range(a.test_num):
            obs, _ = env.reset(seed=a.seed + ep); apply_profile(env, profile); policy.queue.clear(); ok = False
            for _ in range(a.horizon):
                rgb = render_rgb(env, profile); obs, _, term, trunc, info = env.step(policy.step(obs, rgb)); ok |= bool(info.get("success", 0))
                if viewer is not None:
                    if not viewer.is_running():
                        env.close()
                        raise SystemExit("Viewer closed; evaluation stopped without writing partial success rates.")
                    viewer.sync()
                if term or trunc: break
            successes.append(int(ok)); log.info("%s episode %d success=%s", task, ep, ok)
        if viewer is not None:
            viewer.close()
        env.close(); result["tasks"][task] = {"success": int(sum(successes)), "episodes": len(successes), "success_rate": float(np.mean(successes))}; log.info("%s success rate %.3f", task, result["tasks"][task]["success_rate"])
    result["overall_success_rate"] = float(np.mean([x["success_rate"] for x in result["tasks"].values()]))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
