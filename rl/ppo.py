"""Framework-independent PPO utilities for the isolated residual policy.

The functions operate on CPU/GPU tensors and deliberately do not import
RoboTwin or the baseline VLA.  They are enough to build a small trainer later
without coupling the normal SFT entry points to RL dependencies such as Ray.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RolloutBatch:
    states: Tensor
    actions: Tensor
    old_log_probs: Tensor
    returns: Tensor
    advantages: Tensor


@torch.no_grad()
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    last_value: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[Tensor, Tensor]:
    """Compute GAE advantages and returns for a single rollout stream."""
    rewards, values, dones = [x.reshape(-1) for x in (rewards, values, dones)]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value.reshape(()))
    next_value = last_value.reshape(())
    for t in range(rewards.numel() - 1, -1, -1):
        nonterminal = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return advantages, returns


def clipped_ppo_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    values: Tensor,
    returns: Tensor,
    old_values: Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy: Tensor | None = None,
    entropy_coef: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Return PPO actor/value loss and detached scalar diagnostics."""
    ratio = (new_log_probs - old_log_probs).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    actor_loss = -torch.minimum(unclipped, clipped).mean()

    value_delta = values - old_values
    clipped_values = old_values + value_delta.clamp(-clip_ratio, clip_ratio)
    value_loss = 0.5 * torch.maximum(
        (values - returns).square(), (clipped_values - returns).square()
    ).mean()
    entropy_term = entropy.mean() if entropy is not None else values.new_zeros(())
    loss = actor_loss + value_coef * value_loss - entropy_coef * entropy_term
    diagnostics = {
        "loss": float(loss.detach()),
        "actor_loss": float(actor_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "approx_kl": float((old_log_probs - new_log_probs).mean().detach()),
        "clip_fraction": float(((ratio - 1.0).abs() > clip_ratio).float().mean().detach()),
    }
    return loss, diagnostics
