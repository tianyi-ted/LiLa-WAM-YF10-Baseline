"""Isolated, lightweight RL extensions for LiLa-WAM.

The package is deliberately not imported by the normal SFT/evaluation entry
points.  This keeps existing checkpoints and commands backward compatible.
"""

from .residual_policy import ResidualActionPolicy, ResidualVLAController
from .reward import RoboTwinReward
from .ppo import RolloutBatch, compute_gae, clipped_ppo_loss

__all__ = [
    "ResidualActionPolicy", "ResidualVLAController", "RoboTwinReward",
    "RolloutBatch", "compute_gae", "clipped_ppo_loss",
]
