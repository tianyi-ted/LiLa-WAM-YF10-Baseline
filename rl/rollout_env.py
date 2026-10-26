"""Thin RoboTwin macro-step adapter; RoboTwin itself is not modified."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .reward import RoboTwinReward


@dataclass
class RoboTwinMacroEnv:
    """Adapt one RoboTwin task instance to reset/step semantics.

    A step means one VLA action followed by one ``TASK_ENV.take_action`` call,
    exactly matching the existing evaluation bridge.  The adapter is kept
    generic because task construction and instruction generation are owned by
    RoboTwin's evaluation code.
    """

    task_env: Any
    task_args: Dict[str, Any]
    reward_fn: RoboTwinReward
    action_type: str = "qpos"
    episode_id: int = 0
    seed: int = 0
    step_count: int = 0
    previous_stage: Optional[float] = None

    def reset(self, seed: Optional[int] = None, instruction: Optional[str] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
        args = dict(self.task_args)
        args.update({"eval_mode": True, "collect_data": False})
        self.task_env.setup_demo(
            now_ep_num=self.episode_id,
            seed=self.seed,
            is_test=True,
            **args,
        )
        if instruction is not None and hasattr(self.task_env, "set_instruction"):
            self.task_env.set_instruction(instruction=instruction)
        self.step_count = 0
        self.previous_stage = None
        return self.task_env.get_obs()

    def step(self, action):
        self.task_env.take_action(action, action_type=self.action_type)
        self.step_count += 1
        done = bool(getattr(self.task_env, "eval_success", False)) or (
            self.step_count >= int(getattr(self.task_env, "step_lim", 0))
        )
        reward, self.previous_stage = self.reward_fn(
            self.task_env, done, self.previous_stage
        )
        next_obs = self.task_env.get_obs() if not done else self.task_env.get_obs()
        info = {
            "success": bool(getattr(self.task_env, "eval_success", False)),
            "step_count": self.step_count,
            "seed": self.seed,
        }
        return next_obs, reward, done, info
