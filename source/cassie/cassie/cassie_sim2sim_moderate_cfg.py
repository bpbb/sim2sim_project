# Copyright (c) 2024, Boonyaporn Preechasuth
# Cassie sim2sim config with MODERATE domain randomization
#
# Wider DR ranges than baseline for stronger sim2sim robustness:
#   - Friction [0.4, 1.35] (vs baseline [0.5, 1.25])
#   - Mass [-1.5, 2.5] kg (vs baseline [-1.0, 2.0])
#   - Actuator gains x[0.7, 1.3] (vs baseline x[0.8, 1.2])
#   - Joint position ±0.6 rad (vs baseline ±0.5)
#   - Push 0.6 m/s every 8-13s (vs baseline 0.5 m/s every 10-15s)
#
# Usage:
#   python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-Moderate-v0

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .cassie_sim2sim_cfg import CassieSim2SimEnvCfg, CassieSim2SimEventCfg


@configclass
class CassieSim2SimModerateEventCfg(CassieSim2SimEventCfg):
    """Moderate DR — wider ranges than baseline."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.35),
            "dynamic_friction_range": (0.4, 1.35),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.5, 2.5),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.6, 0.6),
            "velocity_range": (-0.6, 0.6),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 13.0),
        params={
            "velocity_range": {
                "x": (-0.6, 0.6),
                "y": (-0.6, 0.6),
            },
        },
    )


@configclass
class CassieSim2SimModerateEnvCfg(CassieSim2SimEnvCfg):
    """Cassie sim2sim with moderate DR."""

    events: CassieSim2SimModerateEventCfg = CassieSim2SimModerateEventCfg()


@configclass
class CassieSim2SimModerateEnvCfg_PLAY(CassieSim2SimModerateEnvCfg):
    """Play configuration with fewer envs."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )

    add_noise = False
