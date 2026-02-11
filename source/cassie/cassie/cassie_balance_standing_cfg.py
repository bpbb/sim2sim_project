# cassie_balance_standing_cfg.py
#
# Focused task: Balance standing ONLY (no locomotion)
#
# Simplified task for easier sim-to-sim transfer:
#   - Zero velocity commands only (standing still)
#   - Heavy penalty on drift/movement
#   - Reward upright pose and stillness
#   - Reduced action space complexity
#
# This should achieve >10s standing in MuJoCo since the policy only learns
# one thing: stand still and maintain balance.
#
# Usage:
#   python scripts/rsl_rl/train.py --task Isaac-Balance-Standing-Cassie-v0 --headless

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
# Domain Randomization for Standing Task
##


@configclass
class CassieBalanceStandingEventCfg:
    """DR for balance standing task.

    Focus: train a policy that can stand still on different surfaces
    and with varying dynamics, without any locomotion.
    """

    # Physics material randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.4),
            "dynamic_friction_range": (0.4, 1.4),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )

    # Mass randomization (pelvis only)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
        },
    )

    # Leg mass randomization (COM shift)
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*thigh.*|.*shin.*|.*tarsus.*"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # Actuator gains (wide range for PhysX/MuJoCo gap)
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Joint friction (energy dissipation)
    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 2.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # Joint position reset (moderate noise)
    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.8, 0.8),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # Push perturbations (balance recovery training)
    # More frequent than locomotion task since standing is easier
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 8.0),  # every 3-8 seconds
        params={
            "velocity_range": {
                "x": (-0.6, 0.6),
                "y": (-0.6, 0.6),
            },
        },
    )


##
# Environment Configuration
##


@configclass
class CassieBalanceStandingEnvCfg(DirectRLEnvCfg):
    """Cassie balance standing task with passive ankles.

    Task: Stand still and maintain balance. No locomotion.

    46-dim obs: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3)
                + joint_pos(12) + joint_vel(12) + actions(10)
    10-dim actions: joint position offsets (ankles passive)
    3-dim cmd: ALWAYS [0, 0, 0] (no locomotion commands)
    """

    # Environment
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.4  # Conservative action scale for standing
    action_space = 10  # passive ankles
    observation_space = 46
    state_space = 0

    # Domain Randomization
    events: CassieBalanceStandingEventCfg = CassieBalanceStandingEventCfg()

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

    # Commands: ALWAYS ZERO (standing only)
    # We still provide command interface for observation compatibility,
    # but commands are fixed at [0, 0, 0]
    lin_vel_x_range = (0.0, 0.0)  # NO forward velocity
    lin_vel_y_range = (0.0, 0.0)  # NO lateral velocity
    ang_vel_z_range = (0.0, 0.0)  # NO turning
    command_resampling_time = 1000.0  # Never resample (commands are always zero)

    # Reward Weights: FOCUS ON STANDING STILL
    # Key philosophy: heavily penalize ANY movement, reward stillness

    # Velocity tracking (should be zero)
    # These become penalties since desired velocity is zero
    track_lin_vel_xy_exp_weight = 0.0  # Disabled (use L2 penalty instead)
    track_ang_vel_z_exp_weight = 0.0   # Disabled (use L2 penalty instead)

    # Direct velocity penalties (L2 on actual velocity, not tracking error)
    lin_vel_xy_l2_weight = -5.0  # NEW: Strong penalty on horizontal movement
    lin_vel_z_l2_weight = -2.0   # Vertical velocity penalty
    ang_vel_xy_l2_weight = -0.2  # Pitch/roll velocity penalty
    ang_vel_z_l2_weight = -0.2   # NEW: Yaw velocity penalty

    # Stillness bonus (NEW)
    # Reward for being completely still (all velocities near zero)
    stillness_bonus_weight = 2.0  # Large bonus for |v| < 0.05 m/s

    # Feet air time (both feet should stay on ground)
    feet_air_time_weight = 0.0  # Disabled for standing task
    feet_air_time_threshold = 0.0

    # Orientation (must stay upright)
    flat_orientation_l2_weight = -8.0  # Even stronger than enhanced (was -5.0)

    # Action penalties
    dof_torques_l2_weight = -1.0e-5  # Small actions preferred
    dof_acc_l2_weight = -1.0e-6
    action_rate_l2_weight = -0.05  # Strong smoothness (was -0.03)

    # Joint deviation penalties (stay near default pose)
    joint_deviation_hip_weight = -0.5   # Stronger than locomotion
    joint_deviation_toes_weight = -0.3

    # Joint limits
    dof_pos_limits_weight = -2.0  # Stronger penalty
    dof_pos_limits_soft_ratio = 0.9

    # Feet contact (both feet on ground)
    feet_slide_weight = -0.3  # Stronger penalty on sliding

    # Termination
    termination_penalty = -200.0

    # Observation Noise (moderate)
    add_noise = True
    noise_scales = {
        "lin_vel": 0.1,
        "ang_vel": 0.2,
        "projected_gravity": 0.05,
        "joint_pos": 0.02,
        "joint_vel": 1.5,
    }

    # Reset Randomization
    reset_position_noise = 0.8  # Moderate position noise
    reset_yaw_noise = math.pi   # Full yaw randomization

    def __post_init__(self):
        super().__post_init__()

        # Remove ankle actuators (passive ankles for MuJoCo compatibility)
        actuator_keys_to_remove = ["ankle_joint", "ankle", "ankle_left", "ankle_right"]

        keys_found = []
        for key in self.robot.actuators.keys():
            if any(x in key for x in actuator_keys_to_remove):
                keys_found.append(key)

        for key in keys_found:
            print(f"[CassieBalanceStandingEnvCfg] Removing actuator: {key}")
            del self.robot.actuators[key]


@configclass
class CassieBalanceStandingEnvCfg_PLAY(CassieBalanceStandingEnvCfg):
    """Play configuration for visualization."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Disable noise during playback
    add_noise = False
