# Copyright (c) 2024, Boonyaporn Preechasuth
# RSL-RL PPO configuration for Go2 sim2sim transfer

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2Sim2SimPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Go2 sim2sim transfer, network [512, 256, 128]."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "go2_sim2sim"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
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
class Go2Sim2SimModeratePPORunnerCfg(Go2Sim2SimPPORunnerCfg):
    """PPO runner for moderate DR. More iterations for harder task."""

    max_iterations = 8000
    experiment_name = "go2_sim2sim_moderate"


@configclass
class Go2Sim2SimAggressivePPORunnerCfg(Go2Sim2SimPPORunnerCfg):
    """PPO runner for aggressive DR. Most iterations for hardest task."""

    max_iterations = 10000
    experiment_name = "go2_sim2sim_aggressive"


@configclass
class Go2Sim2SimActuatorNetPPORunnerCfg(Go2Sim2SimPPORunnerCfg):
    """PPO runner for ActuatorNet variant. Same as baseline, different name."""

    experiment_name = "go2_sim2sim_actuator_net"


@configclass
class Go2Sim2SimRMAPPORunnerCfg(Go2Sim2SimPPORunnerCfg):
    """PPO runner for RMA Phase 1.

    Both actor and critic see policy obs (48) + privileged (4) = 52 dims.
    The actor learns to CONDITION behavior on privileged physics params
    (friction, mass, kp, kd), so that in Phase 2 an adaptation module
    can replace those 4 dims with estimates from observation history.
    """

    max_iterations = 5000
    experiment_name = "go2_rma"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )


##############################################################################
# TERRAIN PPO RUNNERS
##############################################################################


@configclass
class Go2Sim2SimTerrainPPORunnerCfg(Go2Sim2SimPPORunnerCfg):
    """PPO runner for terrain baseline. More iterations for harder task."""

    max_iterations = 8000
    experiment_name = "go2_sim2sim_terrain"


@configclass
class Go2Sim2SimTerrainModeratePPORunnerCfg(Go2Sim2SimPPORunnerCfg):
    """PPO runner for terrain + moderate DR."""

    max_iterations = 10000
    experiment_name = "go2_sim2sim_terrain_moderate"


@configclass
class Go2Sim2SimTerrainAggressivePPORunnerCfg(Go2Sim2SimPPORunnerCfg):
    """PPO runner for terrain + aggressive DR. Most iterations."""

    max_iterations = 12000
    experiment_name = "go2_sim2sim_terrain_aggressive"
