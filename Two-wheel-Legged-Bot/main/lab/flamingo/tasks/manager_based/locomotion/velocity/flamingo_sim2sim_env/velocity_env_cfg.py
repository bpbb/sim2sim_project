# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Sim2Sim Compatible Velocity Environment Configuration.

This configuration is designed to be compatible with the sim2sim_onnx MuJoCo transfer.
Key differences from the standard config:
1. Uses Euler angles (roll, pitch, yaw) instead of projected gravity
2. Uses 3-dim velocity commands (vx, vy, wz) instead of 4-dim (with pos_z)
3. Joint positions without gear ratio for observation
4. Observation structure: joint_pos(6) + joint_vel(8) + ang_vel(3) + euler(3) + action(8) = 28 per frame
5. Total observation: 28 * 3 (stacked) + 3 (commands) = 87 dims
"""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

import lab.flamingo.tasks.manager_based.locomotion.velocity.mdp as mdp


##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=4000.0),
    )


##
# MDP settings
##

@configclass
class Sim2SimCommandsCfg:
    """Sim2Sim compatible command configuration - 3 dims only (vx, vy, wz)."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 8.0),
        rel_standing_envs=0.01,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-2.5, 2.5),
        ),
    )


@configclass
class Sim2SimActionsCfg:
    """Sim2Sim compatible action configuration."""

    # All 6 joint positions (hip, shoulder, leg)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip_joint", "right_hip_joint",
                     "left_shoulder_joint", "right_shoulder_joint",
                     "left_leg_joint", "right_leg_joint"],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    # Wheel velocity with scale 20.0 (matching existing training config)
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["left_wheel_joint", "right_wheel_joint"],
        scale=20.0,
        use_default_offset=False,
        preserve_order=True
    )


@configclass
class Sim2SimObservationsCfg:
    """Sim2Sim compatible observation configuration.

    Observation structure per frame (28 dims):
    - joint_pos (6): left_hip, right_hip, left_shoulder, right_shoulder, left_leg, right_leg
    - joint_vel (8): all joints including wheels
    - base_ang_vel (3): angular velocity in body frame
    - base_euler (3): roll, pitch, yaw angles
    - actions (8): last action

    Stacked 3 frames = 84 dims
    Commands (3 dims): vx, vy, wz
    Total: 87 dims
    """

    @configclass
    class StackPolicyCfg(ObsGroup):
        """Observations for policy - stacked frames."""

        # Joint positions (6) - hip, shoulder, leg (NO gear ratio)
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_hip_joint", "right_hip_joint",
                    "left_shoulder_joint", "right_shoulder_joint",
                    "left_leg_joint", "right_leg_joint"
                ])
            },
        )

        # Joint velocities (8) - all joints including wheels, scaled by 0.15
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_hip_joint", "right_hip_joint",
                    "left_shoulder_joint", "right_shoulder_joint",
                    "left_leg_joint", "right_leg_joint",
                    "left_wheel_joint", "right_wheel_joint"
                ])
            },
            scale=0.15,
        )

        # Base angular velocity (3), scaled by 0.25
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel_link,
            scale=0.25,
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )

        # Euler angles (3) instead of projected gravity
        base_euler = ObsTerm(
            func=mdp.base_euler_angle_link,
            noise=Unoise(n_min=-0.125, n_max=0.125),
        )

        # Last action (8)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoneStackPolicyCfg(ObsGroup):
        """Non-stacked observations - velocity commands only (3 dims)."""

        # Velocity commands (3 dims: vx, vy, wz) - no pos_z
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StackCriticCfg(ObsGroup):
        """Observations for critic - same as policy."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_hip_joint", "right_hip_joint",
                    "left_shoulder_joint", "right_shoulder_joint",
                    "left_leg_joint", "right_leg_joint"
                ])
            },
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "left_hip_joint", "right_hip_joint",
                    "left_shoulder_joint", "right_shoulder_joint",
                    "left_leg_joint", "right_leg_joint",
                    "left_wheel_joint", "right_wheel_joint"
                ])
            },
            scale=0.15,
        )

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel_link, scale=0.25)
        base_euler = ObsTerm(func=mdp.base_euler_angle_link)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class NoneStackCriticCfg(ObsGroup):
        """Non-stacked critic observations."""

        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel_link)
        base_pos_z = ObsTerm(func=mdp.base_pos_z_rel_link, params={"sensor_cfg": None})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups
    stack_policy: StackPolicyCfg = StackPolicyCfg()
    none_stack_policy: NoneStackPolicyCfg = NoneStackPolicyCfg()
    stack_critic: StackCriticCfg = StackCriticCfg()
    none_stack_critic: NoneStackCriticCfg = NoneStackCriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


##
# Environment configuration
##

@configclass
class Sim2SimVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Sim2Sim compatible velocity environment configuration."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: Sim2SimObservationsCfg = Sim2SimObservationsCfg()
    actions: Sim2SimActionsCfg = Sim2SimActionsCfg()
    commands: Sim2SimCommandsCfg = Sim2SimCommandsCfg()
    rewards = None  # Will be defined in task config
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
