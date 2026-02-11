# Copyright (c) 2024, Boonyaporn Preechasuth
# Cassie sim2sim baseline config (Isaac Lab → MuJoCo transfer)
#
# This is the PRIMARY config for the Cassie sim2sim pipeline:
#   train (Isaac Lab) → export_policy → deploy (MuJoCo)
#
# DR settings are tuned for sim2sim transfer robustness:
#   - Friction [0.5, 1.25] — covers contact model differences
#   - Mass [-1.0, 2.0] kg — covers inertia differences
#   - Actuator gains x[0.8, 1.2] — covers motor model differences
#   - Joint position ±0.5 rad — covers MuJoCo pose mismatch
#   - Push perturbations every 10-15s at 0.5 m/s
#
# Usage:
#   python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-v0
#   python scripts/cassie/export_policy.py <log_dir>
#   python scripts/cassie/deploy_mujoco.py scripts/cassie/cassie_isaaclab.yaml

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
# Domain Randomization
##


@configclass
class CassieSim2SimEventCfg:
    """Domain randomization for sim2sim transfer.

    Ranges tuned from Legged Gym / Unitree methodology:
    - Moderate friction + mass covers physics engine differences
    - Actuator gain randomization covers motor model differences
    - Joint position randomization is CRITICAL for MuJoCo pose mismatch
    - Push perturbations encourage balance recovery
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )


##
# Environment Configuration
##


@configclass
class CassieSim2SimEnvCfg(DirectRLEnvCfg):
    """Cassie sim2sim baseline — DirectRL with moderate DR.

    48-dim obs: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3)
                + joint_pos(12) + joint_vel(12) + actions(12)
    12-dim actions: joint position offsets
    """

    # Environment
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 48
    state_space = 0

    # Domain Randomization
    events: CassieSim2SimEventCfg = CassieSim2SimEventCfg()

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

    # Commands
    lin_vel_x_range = (-1.0, 1.0)
    lin_vel_y_range = (-1.0, 1.0)
    ang_vel_z_range = (-1.0, 1.0)
    command_resampling_time = 10.0

    # Reward Weights
    track_lin_vel_xy_exp_weight = 2.0
    track_ang_vel_z_exp_weight = 1.0
    feet_air_time_weight = 2.5
    feet_air_time_threshold = 0.3

    lin_vel_z_l2_weight = -2.0
    ang_vel_xy_l2_weight = -0.05
    dof_torques_l2_weight = -5.0e-6
    dof_acc_l2_weight = -3.75e-7
    action_rate_l2_weight = -0.015
    flat_orientation_l2_weight = -2.5

    joint_deviation_hip_weight = -0.2
    joint_deviation_toes_weight = -0.2

    dof_pos_limits_weight = -1.0
    dof_pos_limits_soft_ratio = 0.9

    feet_slide_weight = -0.1

    termination_penalty = -200.0

    # Observation Noise
    add_noise = True
    noise_scales = {
        "lin_vel": 0.1,
        "ang_vel": 0.2,
        "projected_gravity": 0.05,
        "joint_pos": 0.01,
        "joint_vel": 1.5,
    }

    # Reset Randomization
    reset_position_noise = 0.5
    reset_yaw_noise = math.pi


@configclass
class CassieSim2SimEnvCfg_PLAY(CassieSim2SimEnvCfg):
    """Play configuration with fewer envs for visualization."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Disable noise during playback
    add_noise = False
