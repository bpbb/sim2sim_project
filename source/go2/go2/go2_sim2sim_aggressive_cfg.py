# Copyright (c) 2024, Boonyaporn Preechasuth
# Go2 sim2sim config with AGGRESSIVE domain randomization

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .go2_sim2sim_cfg import Go2Sim2SimEnvCfg, Go2Sim2SimEventCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class Go2Sim2SimAggressiveEventCfg(Go2Sim2SimEventCfg):
    """Aggressive domain randomization for maximum sim2sim robustness.

    Compared to moderate:
    - Even wider friction range
    - Larger base mass variation
    - Wider actuator gain variation
    - Stronger and more frequent push perturbations
    - Joint friction randomization
    """

    # Override: widest friction range
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.8),
            "dynamic_friction_range": (0.3, 1.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Larger base mass randomization
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    # Wider actuator gain randomization
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
        },
    )

    # Stronger and more frequent push perturbations
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "velocity_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7)},
        },
    )

    # Override: wider joint position reset range
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.3, 1.7),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class Go2Sim2SimAggressiveEnvCfg(Go2Sim2SimEnvCfg):
    """Go2 env config with aggressive domain randomization."""

    events: Go2Sim2SimAggressiveEventCfg = Go2Sim2SimAggressiveEventCfg()


@configclass
class Go2Sim2SimAggressiveEnvCfg_PLAY(Go2Sim2SimAggressiveEnvCfg):
    """Play configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
