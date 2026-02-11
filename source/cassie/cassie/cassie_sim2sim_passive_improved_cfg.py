# Copyright (c) 2024, Boonyaporn Preechasuth
# Improved Cassie sim2sim PASSIVE config with curriculum learning
#
# Key changes from passive_enhanced:
#   1. REDUCE DR aggressiveness (below active level initially)
#   2. Curriculum: start easy, gradually increase difficulty
#   3. More balanced reward weights
#   4. Conservative command ranges to encourage learning
#
# Usage:
#   python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Improved-v0

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.cassie import CASSIE_CFG

import isaaclab.envs.mdp as mdp


##
# Domain Randomization (Conservative → Curriculum)
##


@configclass
class CassiePassiveImprovedEventCfg:
    """Conservative domain randomization for passive ankle training.

    Start EASIER than active config, then curriculum will increase difficulty.
    Passive ankles make the task fundamentally harder, so we need gentler DR.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Start with NARROWER friction range
            "static_friction_range": (0.7, 1.15),
            "dynamic_friction_range": (0.7, 1.15),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            # Gentler mass variation
            "mass_distribution_params": (-0.5, 1.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # Start NARROWER than active (will curriculum up)
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # NEW: Joint friction randomization (helps with passive joints)
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 1.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            # Start SMALLER than active (±0.5 → ±0.3)
            "position_range": (-0.3, 0.3),
            "velocity_range": (-0.3, 0.3),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        # LESS frequent than active (15-20s vs 10-15s)
        interval_range_s=(15.0, 20.0),
        params={
            "velocity_range": {
                # Gentler pushes
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
            },
        },
    )


##
# Environment Configuration
##


@configclass
class CassiePassiveImprovedEnvCfg(DirectRLEnvCfg):
    """Improved passive Cassie config with curriculum learning.

    46-dim obs: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3)
                + joint_pos(12) + joint_vel(12) + actions(10)
    10-dim actions: joint position offsets (NO ankles)
    """

    # Environment
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5  # Keep same as active
    action_space = 10   # No ankles
    observation_space = 46  # 48 - 2 (ankle actions)
    state_space = 0

    # Domain Randomization
    events: CassiePassiveImprovedEventCfg = CassiePassiveImprovedEventCfg()

    # Simulation (200 Hz physics)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain (flat ground)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Robot
    robot: ArticulationCfg = CASSIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Contact Sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )

    # Commands (CONSERVATIVE ranges to encourage learning)
    # Start easier than active, curriculum will increase
    lin_vel_x_range = (-0.3, 0.6)  # Bias forward
    lin_vel_y_range = (-0.2, 0.2)  # Less lateral
    ang_vel_z_range = (-0.3, 0.3)  # Less rotation
    command_resampling_time = 10.0

    # Reward Weights (Same as active - proven to work!)
    track_lin_vel_xy_exp_weight = 2.0
    track_ang_vel_z_exp_weight = 1.0
    feet_air_time_weight = 2.5
    feet_air_time_threshold = 0.3

    # Penalties (Use active config values - NOT enhanced!)
    lin_vel_z_l2_weight = -2.0
    ang_vel_xy_l2_weight = -0.05  # NOT -0.1!
    dof_torques_l2_weight = -5.0e-6
    dof_acc_l2_weight = -3.75e-7
    action_rate_l2_weight = -0.015  # NOT -0.03!
    flat_orientation_l2_weight = -2.5  # NOT -5.0!

    joint_deviation_hip_weight = -0.2
    joint_deviation_toes_weight = -0.2

    dof_pos_limits_weight = -1.0
    dof_pos_limits_soft_ratio = 0.9

    feet_slide_weight = -0.1  # NOT -0.2!

    termination_penalty = -200.0

    # Observation Noise (Use active config - NOT enhanced!)
    add_noise = True
    noise_scales = {
        "lin_vel": 0.1,  # NOT 0.15
        "ang_vel": 0.2,  # NOT 0.3
        "projected_gravity": 0.05,
        "joint_pos": 0.01,  # NOT 0.02
        "joint_vel": 1.5,  # NOT 2.0
    }

    # Reset Randomization (Conservative)
    reset_position_noise = 0.3  # Start less than active (0.5)
    reset_yaw_noise = math.pi / 2  # Start less than active (π)


@configclass
class CassiePassiveImprovedEnvCfg_PLAY(CassiePassiveImprovedEnvCfg):
    """Play configuration with fewer envs for visualization."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Disable noise during playback
    add_noise = False