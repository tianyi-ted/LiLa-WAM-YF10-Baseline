"""Reward adapter for RoboTwin's non-Gym task API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class RoboTwinReward:
    """Conservative reward shaping with terminal success as the anchor.

    ``stage_reward`` is used only when explicitly enabled.  It may expose
    privileged simulator progress to the learner, but that progress is never
    passed to the policy.  The default is binary terminal reward, matching the
    simple RoboTwin-compatible RL setup and avoiding task-specific assumptions.
    """

    success_reward: float = 1.0
    failure_reward: float = 0.0
    step_penalty: float = 0.0
    use_stage_reward: bool = False
    stage_scale: float = 0.0

    def __call__(
        self,
        task_env: Any,
        done: bool,
        previous_stage: Optional[float] = None,
    ) -> tuple[float, Optional[float]]:
        success = bool(getattr(task_env, "eval_success", False))
        stage = None
        if self.use_stage_reward and hasattr(task_env, "stage_reward"):
            value = task_env.stage_reward()
            stage = float(value)
        reward = self.success_reward if success else self.failure_reward
        if done and not success:
            reward += self.failure_reward
        reward -= self.step_penalty
        if stage is not None and previous_stage is not None:
            reward += self.stage_scale * (stage - previous_stage)
        return float(reward), stage
