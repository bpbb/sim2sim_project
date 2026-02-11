#!/usr/bin/env python3
"""RMA Environment Wrapper.

This module provides utilities for privileged information management
and observation history buffering for RMA training and deployment.

Privileged Information for Go2 Sim2Sim (4 dims):
  [0] friction_scale  — ground friction scale (1.0 = nominal)
  [1] mass_offset     — base mass offset in kg
  [2] kp_scale        — stiffness / nominal (1.0 = Kp=20)
  [3] kd_scale        — damping / nominal (1.0 = Kd=0.5)

These match the DR parameters in go2_sim2sim_moderate_cfg.py:
  - friction: static_friction_range (0.5, 1.5)
  - mass:     mass_distribution_params (-0.5, 1.0) kg additive
  - stiffness: stiffness_distribution_params (0.8, 1.2) scale
  - damping:   damping_distribution_params (0.8, 1.2) scale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class PrivilegedInfoCfg:
    """Configuration for privileged information.

    Aligned with go2_sim2sim_moderate_cfg.py DR ranges.
    """

    # Ground friction scale factor
    friction_range: Tuple[float, float] = (0.5, 1.5)

    # Base mass offset (kg, additive)
    mass_offset_range: Tuple[float, float] = (-0.5, 1.0)

    # Actuator stiffness scale factor (nominal Kp=20.0)
    kp_scale_range: Tuple[float, float] = (0.8, 1.2)

    # Actuator damping scale factor (nominal Kd=0.5)
    kd_scale_range: Tuple[float, float] = (0.8, 1.2)

    @property
    def total_dim(self) -> int:
        """Total dimension of privileged info vector."""
        return 4  # friction_scale, mass_offset, kp_scale, kd_scale


class PrivilegedInfoBuffer:
    """Buffer to store and manage privileged information.

    Stores the ground-truth environment parameters that are used by the
    RMA encoder during training or recorded for Phase 2 data collection.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: PrivilegedInfoCfg,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.cfg = cfg
        self.device = device
        self.dim = cfg.total_dim

        # Initialize buffers (nominal values)
        self.friction_scale = torch.ones(num_envs, 1, device=device)
        self.mass_offset = torch.zeros(num_envs, 1, device=device)
        self.kp_scale = torch.ones(num_envs, 1, device=device)
        self.kd_scale = torch.ones(num_envs, 1, device=device)

    def randomize(self, env_ids: Optional[torch.Tensor] = None):
        """Randomize privileged info for specified environments.

        Args:
            env_ids: Environment indices to randomize. If None, randomize all.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        n = len(env_ids)
        cfg = self.cfg

        self.friction_scale[env_ids] = torch.empty(
            n, 1, device=self.device
        ).uniform_(*cfg.friction_range)

        self.mass_offset[env_ids] = torch.empty(
            n, 1, device=self.device
        ).uniform_(*cfg.mass_offset_range)

        self.kp_scale[env_ids] = torch.empty(
            n, 1, device=self.device
        ).uniform_(*cfg.kp_scale_range)

        self.kd_scale[env_ids] = torch.empty(
            n, 1, device=self.device
        ).uniform_(*cfg.kd_scale_range)

    def get_privileged_obs(self) -> torch.Tensor:
        """Get concatenated privileged observation vector.

        Returns:
            privileged_obs: [num_envs, 4]
        """
        return torch.cat([
            self.friction_scale,   # [N, 1]
            self.mass_offset,      # [N, 1]
            self.kp_scale,         # [N, 1]
            self.kd_scale,         # [N, 1]
        ], dim=-1)

    def normalize(self, privileged_obs: torch.Tensor) -> torch.Tensor:
        """Normalize privileged observations to [-1, 1] range.

        Args:
            privileged_obs: Raw privileged observations [N, 4]

        Returns:
            Normalized privileged observations [N, 4]
        """
        cfg = self.cfg
        ranges = [
            cfg.friction_range,
            cfg.mass_offset_range,
            cfg.kp_scale_range,
            cfg.kd_scale_range,
        ]

        normalized = privileged_obs.clone()
        for idx, (low, high) in enumerate(ranges):
            normalized[:, idx] = 2 * (privileged_obs[:, idx] - low) / (high - low) - 1

        return normalized


class ObservationHistoryBuffer:
    """Buffer to store observation history for adaptation module.

    Maintains a sliding window of past observations for each environment.
    """

    def __init__(
        self,
        num_envs: int,
        obs_dim: int,
        history_length: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.history_length = history_length
        self.device = device

        # History buffer: [num_envs, history_length, obs_dim]
        self.buffer = torch.zeros(
            num_envs, history_length, obs_dim, device=device
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None):
        """Reset history for specified environments.

        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self.buffer.zero_()
        else:
            self.buffer[env_ids] = 0.0

    def update(self, obs: torch.Tensor):
        """Add new observation to history.

        Args:
            obs: New observation [num_envs, obs_dim]
        """
        # Shift history left (oldest dropped)
        self.buffer = torch.roll(self.buffer, shifts=-1, dims=1)
        # Add new observation at the end
        self.buffer[:, -1, :] = obs

    def get_history(self) -> torch.Tensor:
        """Get full observation history.

        Returns:
            history: [num_envs, history_length, obs_dim]
        """
        return self.buffer

    def get_history_flat(self) -> torch.Tensor:
        """Get flattened observation history.

        Returns:
            history_flat: [num_envs, history_length * obs_dim]
        """
        return self.buffer.reshape(self.num_envs, -1)
