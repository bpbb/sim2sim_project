# cassie_balance_standing_env.py

from __future__ import annotations

import torch
import gymnasium as gym

from .cassie_passive_env import CassiePassiveEnv
from .cassie_balance_standing_cfg import CassieBalanceStandingEnvCfg


class CassieBalanceStandingEnv(CassiePassiveEnv):
    """Cassie balance standing environment (passive ankles, zero commands).

    Simplified task focusing ONLY on standing balance:
      - Commands always [0, 0, 0] (no locomotion)
      - Heavy penalties on any movement
      - Bonus for complete stillness
      - Strong orientation penalty

    This should transfer much better to MuJoCo than locomotion because:
      1. Simpler task (fewer failure modes)
      2. No gait dynamics (no step timing issues)
      3. Small actions (less sensitive to PD gains)
      4. Clear success criterion (velocity ≈ 0)
    """

    cfg: CassieBalanceStandingEnvCfg

    def __init__(self, cfg: CassieBalanceStandingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Store reward weights
        self._lin_vel_xy_l2_weight = cfg.lin_vel_xy_l2_weight
        self._ang_vel_z_l2_weight = cfg.ang_vel_z_l2_weight
        self._stillness_bonus_weight = cfg.stillness_bonus_weight

        print(f"[CassieBalanceStandingEnv] Standing-only task")
        print(f"  Lin vel XY penalty: {self._lin_vel_xy_l2_weight}")
        print(f"  Ang vel Z penalty: {self._ang_vel_z_l2_weight}")
        print(f"  Stillness bonus: {self._stillness_bonus_weight}")

    def _resample_commands(self, env_ids: torch.Tensor):
        """Override: commands are always [0, 0, 0] for standing task."""
        # Set all commands to zero (standing still)
        self._commands[env_ids, :] = 0.0

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards focused on standing still."""

        # Call base class rewards (orientation, action penalties, etc.)
        # Note: base class track_lin_vel_xy_exp and track_ang_vel_z_exp are disabled
        rewards = super()._get_rewards()

        # Get velocities
        lin_vel_xy = self._robot.data.root_lin_vel_b[:, :2]  # [N, 2]
        lin_vel_z = self._robot.data.root_lin_vel_b[:, 2]    # [N]
        ang_vel_xy = self._robot.data.root_ang_vel_b[:, :2]  # [N, 2] (pitch, roll rates)
        ang_vel_z = self._robot.data.root_ang_vel_b[:, 2]    # [N] (yaw rate)

        # Horizontal velocity penalty (ANY movement is bad)
        lin_vel_xy_penalty = torch.sum(lin_vel_xy ** 2, dim=1)
        rewards += self._lin_vel_xy_l2_weight * lin_vel_xy_penalty

        # Yaw velocity penalty (no turning)
        ang_vel_z_penalty = ang_vel_z ** 2
        rewards += self._ang_vel_z_l2_weight * ang_vel_z_penalty

        # Stillness bonus: reward for ALL velocities being near zero
        # Define "still" as: |v_xy| < 0.05 m/s, |v_z| < 0.02 m/s, |w| < 0.1 rad/s
        vel_xy_magnitude = torch.norm(lin_vel_xy, dim=1)
        vel_z_magnitude = torch.abs(lin_vel_z)
        ang_vel_magnitude = torch.norm(self._robot.data.root_ang_vel_b, dim=1)

        is_still_xy = vel_xy_magnitude < 0.05
        is_still_z = vel_z_magnitude < 0.02
        is_still_ang = ang_vel_magnitude < 0.1

        is_completely_still = is_still_xy & is_still_z & is_still_ang

        stillness_bonus = torch.where(
            is_completely_still,
            torch.ones_like(vel_xy_magnitude),
            torch.zeros_like(vel_xy_magnitude)
        )
        rewards += self._stillness_bonus_weight * stillness_bonus

        return rewards
