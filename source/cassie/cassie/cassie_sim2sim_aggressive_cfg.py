# Copyright (c) 2024, Boonyaporn Preechasuth
# Cassie sim2sim config with AGGRESSIVE domain randomization
#
# Widest DR ranges for maximum sim2sim robustness:
#   - Friction [0.3, 1.5] (vs baseline [0.5, 1.25])
#   - Mass [-2.0, 3.0] kg (vs baseline [-1.0, 2.0])
#   - Actuator gains x[0.6, 1.4] (vs baseline x[0.8, 1.2])
#   - Joint position ±0.75 rad (vs baseline ±0.5)
#   - Push 0.75 m/s every 6-10s (vs baseline 0.5 m/s every 10-15s)
#
# Usage:
#   python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-Aggressive-v0

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .cassie_sim2sim_cfg import CassieSim2SimEnvCfg, CassieSim2SimEventCfg


@configclass
class CassieSim2SimAggressiveEventCfg(CassieSim2SimEventCfg):
    """Aggressive DR — widest ranges for maximum transfer robustness."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.6, 1.4),
            "damping_distribution_params": (0.6, 1.4),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.75, 0.75),
            "velocity_range": (-0.75, 0.75),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.75, 0.75),
                "y": (-0.75, 0.75),
            },
        },
    )


@configclass
class CassieSim2SimAggressiveEnvCfg(CassieSim2SimEnvCfg):
    """Cassie sim2sim with aggressive DR."""

    events: CassieSim2SimAggressiveEventCfg = CassieSim2SimAggressiveEventCfg()


@configclass
class CassieSim2SimAggressiveEnvCfg_PLAY(CassieSim2SimAggressiveEnvCfg):
    """Play configuration with fewer envs."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )

    add_noise = False
