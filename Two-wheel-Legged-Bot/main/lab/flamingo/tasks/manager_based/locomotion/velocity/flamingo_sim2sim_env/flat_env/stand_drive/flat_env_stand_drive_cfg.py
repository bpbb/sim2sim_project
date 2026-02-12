# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Sim2Sim Compatible Flat Environment Configuration for Flamingo.

This configuration produces policies that can be directly transferred to MuJoCo
using the sim2sim_onnx framework.
"""

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp
from lab.flamingo.tasks.manager_based.locomotion.velocity.flamingo_sim2sim_env.velocity_env_cfg import (
    Sim2SimVelocityEnvCfg,
)

from lab.flamingo.assets.flamingo.flamingo_rev03_1_1 import FLAMINGO_CFG


@configclass
class FlamingoSim2SimRewardsCfg:
    """Reward configuration for sim2sim training."""

    # Velocity tracking (matched to working Flamingo_Flat_Stand_Drive config)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_link_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Velocity penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)

    # Orientation
    flat_orientation = RewTerm(func=mdp.flat_euler_angle_l2, weight=-5.0)
    base_height = RewTerm(
        func=mdp.base_height_adaptive_l2,
        weight=-25.0,
        params={
            "target_height": 0.35842,
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )

    # Joint penalties
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation_shoulder = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_joint"])},
    )
    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_zero_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_joint"])},
    )

    # Limit penalties
    dof_pos_limits_hip = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    dof_pos_limits_shoulder = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_shoulder_joint")},
    )
    dof_pos_limits_leg = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_leg_joint")},
    )

    # Contact penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_shoulder_link", ".*_leg_link"]),
            "threshold": 1.0,
        },
    )

    # Regularization
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
    )
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Termination
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # Alive bonus
    alive_bonus = RewTerm(func=mdp.is_alive, weight=2.0)


@configclass
class FlamingoSim2SimFlatEnvCfg(Sim2SimVelocityEnvCfg):
    """Sim2Sim compatible flat environment."""

    rewards: FlamingoSim2SimRewardsCfg = FlamingoSim2SimRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 20.0
        self.scene.robot = FLAMINGO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Update termination
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            ".*_hip_link",
            ".*_shoulder_link",
            ".*_leg_link",
        ]

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)


@configclass
class FlamingoSim2SimFlatEnvCfg_PLAY(FlamingoSim2SimFlatEnvCfg):
    """Play configuration for sim2sim env."""

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 30.0
        self.sim.render_interval = self.decimation

        # Disable observation noise for play
        self.observations.stack_policy.enable_corruption = False
        self.observations.none_stack_policy.enable_corruption = False

        # Less aggressive domain randomization for play
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.5, 1.0)
        self.events.physics_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.8, 1.0)

        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
