# Copyright (c) 2024, Boonyaporn Preechasuth
# Go2 sim2sim config for RMA (Rapid Motor Adaptation) training.
#
# Extends moderate DR env with privileged observations for the critic.
# The privileged obs expose ground-truth randomized physics parameters:
#   - ground friction coefficient
#   - base mass offset
#   - actuator stiffness scale
#   - actuator damping scale
#
# RMA Phase 1 training:
#   - Actor sees: policy obs (48) + privileged (4) = 52 dims
#   - Critic sees: policy obs (48) + privileged (4) = 52 dims
#   The actor learns to USE the privileged info, so in Phase 2
#   an adaptation module can replace those 4 dims with estimates.

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .go2_sim2sim_moderate_cfg import Go2Sim2SimModerateEnvCfg
from .mdp import observations as go2_obs


##############################################################################
# OBSERVATIONS — 52-dim policy (48 standard + 4 privileged) for RMA Phase 1
##############################################################################

@configclass
class Go2Sim2SimRMAObservationsCfg:
    """Observation config for RMA Phase 1 training.

    Policy group (52 dims): standard locomotion obs (48) + privileged (4).
    Critic group (52 dims): same as policy (both see privileged info).

    The actor must see privileged info so it learns to condition on it.
    In Phase 2, the adaptation module replaces the 4 privileged dims.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations — 48 dims + 4 privileged = 52 dims."""

        # 1. Standard Locomotion Obs (48 dims)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        # 2. Privileged Info (4 dims) - Explicitly given to Actor for RMA Phase 1
        ground_friction = ObsTerm(
            func=go2_obs.ground_friction,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_mass_offset = ObsTerm(
            func=go2_obs.body_mass_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_mass": 6.921,
            },
        )
        kp_scale = ObsTerm(
            func=go2_obs.actuator_stiffness_scale,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_kp": 20.0,
            },
        )
        kd_scale = ObsTerm(
            func=go2_obs.actuator_damping_scale,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_kd": 0.5,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations — policy obs + privileged info (52 dims).

        Privileged info (4 dims):
          [0] ground_friction  — mean static friction
          [1] base_mass_offset — mass offset from nominal (kg)
          [2] kp_scale         — stiffness / nominal
          [3] kd_scale         — damping / nominal
        """

        # Same as policy observations (48 dims)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        # Privileged info (4 dims)
        ground_friction = ObsTerm(
            func=go2_obs.ground_friction,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        base_mass_offset = ObsTerm(
            func=go2_obs.body_mass_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_mass": 6.921,
            },
        )
        kp_scale = ObsTerm(
            func=go2_obs.actuator_stiffness_scale,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_kp": 20.0,
            },
        )
        kd_scale = ObsTerm(
            func=go2_obs.actuator_damping_scale,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "nominal_kd": 0.5,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##############################################################################
# MAIN ENVIRONMENT CONFIGS
##############################################################################

@configclass
class Go2Sim2SimRMAEnvCfg(Go2Sim2SimModerateEnvCfg):
    """Go2 RMA env — moderate DR with privileged observations for actor+critic.

    Inherits everything from moderate DR (friction, mass, gains, push).
    Both actor and critic see 52 dims (48 standard + 4 privileged).
    """

    observations: Go2Sim2SimRMAObservationsCfg = Go2Sim2SimRMAObservationsCfg()


@configclass
class Go2Sim2SimRMAEnvCfg_PLAY(Go2Sim2SimRMAEnvCfg):
    """Play configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
