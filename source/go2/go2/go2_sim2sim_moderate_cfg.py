# Copyright (c) 2024, Boonyaporn Preechasuth
# Go2 sim2sim config with MODERATE domain randomization

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .go2_sim2sim_cfg import Go2Sim2SimEnvCfg, Go2Sim2SimEventCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class Go2Sim2SimModerateEventCfg(Go2Sim2SimEventCfg):
    """Moderate domain randomization for sim2sim transfer.

    Adds to baseline:
    - Wider friction range
    - Base mass randomization
    - Actuator gain (Kp/Kd) randomization
    - External push perturbations
    """

    # Override: wider friction range
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # NEW: base mass randomization (Go2 base mass ~6.9 kg)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.5, 1.0),
            "operation": "add",
        },
    )

    # NEW: actuator gain randomization (Kp=20, Kd=0.5 nominal)
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # NEW: external push perturbations
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class Go2Sim2SimModerateEnvCfg(Go2Sim2SimEnvCfg):
    """Go2 env config with moderate domain randomization."""

    events: Go2Sim2SimModerateEventCfg = Go2Sim2SimModerateEventCfg()


@configclass
class Go2Sim2SimModerateEnvCfg_PLAY(Go2Sim2SimModerateEnvCfg):
    """Play configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
