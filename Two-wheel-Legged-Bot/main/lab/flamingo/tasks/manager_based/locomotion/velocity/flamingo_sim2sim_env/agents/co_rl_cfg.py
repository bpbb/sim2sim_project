# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
CO-RL PPO Configuration for Sim2Sim Compatible Training.

This configuration produces policies that can be directly transferred to MuJoCo
using the sim2sim_onnx framework.
"""

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
)


@configclass
class FlamingoSim2SimPPORunnerCfg(CoRlPolicyRunnerCfg):
    """Base PPO configuration for sim2sim training."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "Flamingo_Sim2Sim"
    experiment_description = "Sim2sim compatible training"
    empirical_normalization = False

    policy = CoRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FlamingoSim2SimFlatPPORunnerCfg(FlamingoSim2SimPPORunnerCfg):
    """PPO configuration for flat environment sim2sim training."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "Flamingo_Sim2Sim_Flat"
        self.save_interval = 500
