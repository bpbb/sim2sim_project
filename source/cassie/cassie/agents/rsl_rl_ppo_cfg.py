# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL PPO configuration for Cassie sim2sim."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CassiePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Base PPO config for Cassie."""
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "cassie_sim2sim"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CassieSim2SimPPORunnerCfg(CassiePPORunnerCfg):
    """PPO config for Cassie sim2sim baseline."""
    experiment_name = "cassie_sim2sim"


@configclass
class CassieSim2SimModeratePPORunnerCfg(CassiePPORunnerCfg):
    """PPO config for Cassie sim2sim with moderate DR."""
    experiment_name = "cassie_sim2sim_moderate"


@configclass
class CassieSim2SimAggressivePPORunnerCfg(CassiePPORunnerCfg):
    """PPO config for Cassie sim2sim with aggressive DR."""
    experiment_name = "cassie_sim2sim_aggressive"
