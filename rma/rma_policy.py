"""RMA Policy Architecture.

This module implements the RMA (Rapid Motor Adaptation) policy architecture:
1. Environment Encoder: Maps privileged info → latent vector
2. Base Policy: Maps (obs, latent) → action
3. Adaptation Module: Maps observation history → estimated latent

Reference: https://arxiv.org/abs/2107.04034
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RMAPolicyCfg:
    """Configuration for RMA policy.

    Aligned with rma/configs/go2.yaml and source/go2 baseline:
    - obs_dim=48: matches Go2Sim2SimObservationsCfg
    - privileged_dim=4: friction_scale, mass_offset, kp_scale, kd_scale
    - latent_dim=4: compact encoding of 4-dim privileged info
    """

    # Observation dimensions
    obs_dim: int = 48  # Standard locomotion observations
    privileged_dim: int = 4  # friction_scale, mass_offset, kp_scale, kd_scale
    action_dim: int = 12  # Joint actions

    # Latent dimension (environment encoding)
    latent_dim: int = 4

    # Base policy architecture
    policy_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    policy_activation: str = "elu"

    # Environment encoder architecture
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [32, 16])

    # Adaptation module architecture
    adaptation_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    history_length: int = 50  # Number of past observations to use

    # Training
    init_noise_std: float = 1.0


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "elu": nn.ELU(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
    }
    return activations.get(name.lower(), nn.ELU())


class EnvironmentEncoder(nn.Module):
    """Encodes privileged environment information into latent vector.

    Used during training to provide ground-truth environment encoding.

    Input: privileged_info (friction, mass, actuator gains, etc.)
    Output: latent vector e
    """

    def __init__(
        self,
        privileged_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = [64, 32],
        activation: str = "elu",
    ):
        super().__init__()

        layers = []
        prev_dim = privileged_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                get_activation(activation),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, privileged_info: torch.Tensor) -> torch.Tensor:
        """Encode privileged information.

        Args:
            privileged_info: [batch, privileged_dim]

        Returns:
            latent: [batch, latent_dim]
        """
        return self.encoder(privileged_info)


class AdaptationModule(nn.Module):
    """Estimates environment encoding from observation history.

    Used during deployment when privileged info is not available.

    Input: observation history (last N observations)
    Output: estimated latent vector ê
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        history_length: int = 50,
        hidden_dims: List[int] = [256, 128],
        activation: str = "elu",
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.history_length = history_length
        input_dim = obs_dim * history_length

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                get_activation(activation),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.adapter = nn.Sequential(*layers)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Estimate environment encoding from observation history.

        Args:
            obs_history: [batch, history_length, obs_dim] or [batch, history_length * obs_dim]

        Returns:
            estimated_latent: [batch, latent_dim]
        """
        # Flatten history if needed
        if obs_history.dim() == 3:
            obs_history = obs_history.reshape(obs_history.shape[0], -1)

        return self.adapter(obs_history)


class BasePolicy(nn.Module):
    """Base policy network that takes observations and environment encoding.

    Input: (observations, latent_encoding)
    Output: action mean
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "elu",
    ):
        super().__init__()

        input_dim = obs_dim + latent_dim
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                get_activation(activation),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.policy = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute action from observations and environment encoding.

        Args:
            obs: [batch, obs_dim]
            latent: [batch, latent_dim]

        Returns:
            action_mean: [batch, action_dim]
        """
        x = torch.cat([obs, latent], dim=-1)
        return self.policy(x)


class RMAPolicy(nn.Module):
    """Complete RMA Policy with encoder, base policy, and adaptation module.

    Training Phase 1: Use encoder to provide latent, train base policy
    Training Phase 2: Train adaptation module to match encoder output
    Deployment: Use adaptation module to estimate latent
    """

    def __init__(self, cfg: RMAPolicyCfg):
        super().__init__()

        self.cfg = cfg

        # Environment encoder (privileged → latent)
        self.encoder = EnvironmentEncoder(
            privileged_dim=cfg.privileged_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=cfg.encoder_hidden_dims,
        )

        # Base policy (obs + latent → action)
        self.base_policy = BasePolicy(
            obs_dim=cfg.obs_dim,
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
            hidden_dims=cfg.policy_hidden_dims,
            activation=cfg.policy_activation,
        )

        # Adaptation module (obs_history → estimated latent)
        self.adaptation = AdaptationModule(
            obs_dim=cfg.obs_dim,
            latent_dim=cfg.latent_dim,
            history_length=cfg.history_length,
            hidden_dims=cfg.adaptation_hidden_dims,
        )

        # Action noise for exploration
        self.action_std = nn.Parameter(
            torch.ones(cfg.action_dim) * cfg.init_noise_std
        )

        # Observation history buffer (for deployment)
        self._obs_history = None

    def reset_history(self, batch_size: int, device: torch.device):
        """Reset observation history buffer."""
        self._obs_history = torch.zeros(
            batch_size,
            self.cfg.history_length,
            self.cfg.obs_dim,
            device=device,
        )

    def update_history(self, obs: torch.Tensor):
        """Update observation history with new observation."""
        if self._obs_history is None:
            self.reset_history(obs.shape[0], obs.device)

        # Shift history and add new observation
        self._obs_history = torch.roll(self._obs_history, shifts=-1, dims=1)
        self._obs_history[:, -1, :] = obs

    def forward_train(
        self,
        obs: torch.Tensor,
        privileged_info: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass during training (Phase 1).

        Uses encoder to get ground-truth latent.

        Args:
            obs: [batch, obs_dim]
            privileged_info: [batch, privileged_dim]

        Returns:
            action_mean: [batch, action_dim]
            latent: [batch, latent_dim] (for Phase 2 supervision)
        """
        latent = self.encoder(privileged_info)
        action_mean = self.base_policy(obs, latent)
        return action_mean, latent

    def forward_adapt(
        self,
        obs: torch.Tensor,
        obs_history: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using adaptation module (Phase 2 / Deployment).

        Uses observation history to estimate latent.

        Args:
            obs: [batch, obs_dim]
            obs_history: [batch, history_length, obs_dim] or None (use internal buffer)

        Returns:
            action_mean: [batch, action_dim]
            estimated_latent: [batch, latent_dim]
        """
        # Use internal history if not provided
        if obs_history is None:
            self.update_history(obs)
            obs_history = self._obs_history

        # Estimate latent from history
        estimated_latent = self.adaptation(obs_history)

        # Get action
        action_mean = self.base_policy(obs, estimated_latent)

        return action_mean, estimated_latent

    def forward(
        self,
        obs: torch.Tensor,
        privileged_info: Optional[torch.Tensor] = None,
        obs_history: Optional[torch.Tensor] = None,
        use_adaptation: bool = False,
    ) -> torch.Tensor:
        """Unified forward pass.

        Args:
            obs: Current observation
            privileged_info: Privileged info (training only)
            obs_history: Observation history (deployment only)
            use_adaptation: Whether to use adaptation module

        Returns:
            action_mean
        """
        if use_adaptation or privileged_info is None:
            action_mean, _ = self.forward_adapt(obs, obs_history)
        else:
            action_mean, _ = self.forward_train(obs, privileged_info)

        return action_mean

    def get_adaptation_loss(
        self,
        obs_history: torch.Tensor,
        privileged_info: torch.Tensor,
    ) -> torch.Tensor:
        """Compute adaptation module supervision loss.

        The adaptation module is trained to match the encoder output.

        Args:
            obs_history: [batch, history_length, obs_dim]
            privileged_info: [batch, privileged_dim]

        Returns:
            MSE loss between estimated and true latent
        """
        with torch.no_grad():
            true_latent = self.encoder(privileged_info)

        estimated_latent = self.adaptation(obs_history)

        return nn.functional.mse_loss(estimated_latent, true_latent)


class RMACritic(nn.Module):
    """Critic network for RMA (uses privileged info during training)."""

    def __init__(
        self,
        obs_dim: int,
        privileged_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        activation: str = "elu",
    ):
        super().__init__()

        input_dim = obs_dim + privileged_dim
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                get_activation(activation),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))

        self.critic = nn.Sequential(*layers)

    def forward(
        self,
        obs: torch.Tensor,
        privileged_info: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate value.

        Args:
            obs: [batch, obs_dim]
            privileged_info: [batch, privileged_dim]

        Returns:
            value: [batch, 1]
        """
        x = torch.cat([obs, privileged_info], dim=-1)
        return self.critic(x)
