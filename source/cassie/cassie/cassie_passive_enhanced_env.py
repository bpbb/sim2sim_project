# cassie_passive_enhanced_env.py

from __future__ import annotations

import torch
import gymnasium as gym

from .cassie_passive_env import CassiePassiveEnv
from .cassie_sim2sim_passive_enhanced_cfg import CassieSim2SimPassiveEnhancedEnvCfg


class CassiePassiveEnhancedEnv(CassiePassiveEnv):
    """Enhanced Cassie passive ankle environment with additional reward terms.

    Adds two new reward terms targeting the identified deployment failure modes:
      1. Forward drift penalty: penalizes vx when cmd_x ≈ 0
      2. Standing bonus: rewards low velocity when commanded to stand

    These rewards help the policy learn stable standing behavior that transfers
    better to MuJoCo.
    """

    cfg: CassieSim2SimPassiveEnhancedEnvCfg

    def __init__(self, cfg: CassieSim2SimPassiveEnhancedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Store reward weights for new terms
        self._forward_drift_penalty_weight = cfg.forward_drift_penalty_weight
        self._standing_bonus_weight = cfg.standing_bonus_weight

        print(f"[CassiePassiveEnhancedEnv] Forward drift penalty: {self._forward_drift_penalty_weight}")
        print(f"[CassiePassiveEnhancedEnv] Standing bonus: {self._standing_bonus_weight}")

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards with enhanced terms for sim-to-sim transfer."""

        # Call base class rewards (all existing terms)
        rewards = super()._get_rewards()

        # NEW: Forward drift penalty
        # Penalize forward velocity when command is near zero
        # Target: when cmd_x < 0.1, vx should also be near 0
        cmd_x = self._commands[:, 0]  # forward velocity command
        vel_x = self._robot.data.root_lin_vel_b[:, 0]  # forward velocity

        # Mask: only apply when command is small (standing or slow walking)
        standing_mask = torch.abs(cmd_x) < 0.1
        forward_drift = torch.where(
            standing_mask,
            vel_x ** 2,  # L2 penalty on forward velocity
            torch.zeros_like(vel_x)
        )
        rewards += self._forward_drift_penalty_weight * forward_drift

        # NEW: Standing bonus
        # Reward low total velocity when commanded to stand/move slowly
        vel_magnitude = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
        cmd_magnitude = torch.norm(self._commands[:, :2], dim=1)

        # Mask: command magnitude < 0.1 (standing)
        # Reward: velocity magnitude < 0.1 (actually standing still)
        standing_cmd_mask = cmd_magnitude < 0.1
        standing_vel_mask = vel_magnitude < 0.1
        standing_success = standing_cmd_mask & standing_vel_mask

        standing_bonus = torch.where(
            standing_success,
            torch.ones_like(vel_magnitude),
            torch.zeros_like(vel_magnitude)
        )
        rewards += self._standing_bonus_weight * standing_bonus

        return rewards
