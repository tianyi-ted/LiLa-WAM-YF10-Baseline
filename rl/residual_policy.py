"""Low-memory RL residual policy.

This module does not modify the SFT VLA model.  It learns a bounded correction
to an already working action:

    action = base_action + residual_scale * residual

The final layer is zero initialized, so a newly created controller is exactly
the baseline policy at initialization.  The policy consumes proprioception and
optionally a task condition; it intentionally does not duplicate DINO or the
Action Expert in the first experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def _state_from_observation(observation: Dict[str, Any], state_dim: int) -> np.ndarray:
    """Extract the same endpose ordering used by ``robotwin_infer``."""
    endpose = observation["endpose"]
    left = np.concatenate([
        np.asarray(endpose["left_endpose"], dtype=np.float32),
        np.asarray([endpose["left_gripper"]], dtype=np.float32),
    ])
    if state_dim == 8:
        return left
    right = np.concatenate([
        np.asarray(endpose["right_endpose"], dtype=np.float32),
        np.asarray([endpose["right_gripper"]], dtype=np.float32),
    ])
    state = np.concatenate([left, right])
    if state.shape != (state_dim,):
        raise ValueError(f"expected {state_dim}D state, got {state.shape}")
    return state


class ResidualActionPolicy(nn.Module):
    """Small Gaussian residual actor with a shared value head.

    ``forward`` returns bounded residuals in normalized action units, a value
    estimate, and a Normal distribution for PPO-style training.  The actor's
    output layer is zero initialized; ``log_std`` starts conservatively at
    ``init_log_std``.
    """

    def __init__(
        self,
        state_dim: int = 16,
        action_dim: int = 14,
        task_cond_dim: int = 0,
        hidden_dim: int = 128,
        residual_limit: float = 0.10,
        init_log_std: float = -2.5,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.task_cond_dim = int(task_cond_dim)
        self.residual_limit = float(residual_limit)
        input_dim = self.state_dim + self.task_cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.residual_head = nn.Linear(hidden_dim, self.action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(init_log_std)))

        # Exact baseline behavior before RL updates.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        state: torch.Tensor,
        task_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Normal]:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if self.task_cond_dim:
            if task_cond is None:
                task_cond = state.new_zeros((state.shape[0], self.task_cond_dim))
            if task_cond.ndim == 1:
                task_cond = task_cond.unsqueeze(0)
            x = torch.cat([state, task_cond], dim=-1)
        else:
            x = state
        h = self.encoder(x)
        residual = torch.tanh(self.residual_head(h)) * self.residual_limit
        value = self.value_head(h).squeeze(-1)
        std = self.log_std.clamp(-6.0, 1.0).exp().expand_as(residual)
        return residual, value, Normal(residual, std)

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        task_cond: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, value, distribution = self(state, task_cond)
        sampled = residual if deterministic else distribution.sample().clamp(
            -self.residual_limit, self.residual_limit
        )
        log_prob = distribution.log_prob(sampled).sum(dim=-1)
        return sampled, value, log_prob


@dataclass
class ResidualVLAController:
    """Inference/rollout wrapper around an existing ``RobotWinInference``.

    It calls the untouched SFT controller first, then adds a small residual.
    Set ``enabled=False`` to return the baseline action byte-for-byte (up to
    NumPy conversion).  This wrapper is not used by ``eval_vla.py``.
    """

    base_policy: Any
    residual_policy: ResidualActionPolicy
    device: str = "cuda"
    enabled: bool = True
    residual_scale: float = 1.0
    state_dim: int = 16
    task_cond: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self.base_policy.reset()

    def set_task(self, task_name: str) -> None:
        if hasattr(self.base_policy, "set_task"):
            self.base_policy.set_task(task_name)
        # The task condition is optional for the first low-memory experiment.
        if getattr(self.residual_policy, "task_cond_dim", 0):
            self.task_cond = getattr(self.base_policy, "task_cond", None)

    def step(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        base_action = np.asarray(self.base_policy.step(observation, instruction), dtype=np.float32)
        if not self.enabled:
            return base_action
        state = torch.from_numpy(_state_from_observation(observation, self.state_dim))
        state = state.to(self.device, dtype=torch.float32)
        cond = self.task_cond.to(self.device) if self.task_cond is not None else None
        residual, _, _ = self.residual_policy.act(state, cond, deterministic=True)
        correction = residual[0].float().cpu().numpy() * float(self.residual_scale)
        if correction.shape != base_action.shape:
            raise ValueError(f"residual shape {correction.shape} != base action {base_action.shape}")
        # The first prototype applies a bounded correction in the environment
        # action units.  A later version can move this operation to normalized
        # action space once the action statistics are threaded into the wrapper.
        return base_action + correction
