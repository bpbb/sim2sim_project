# cassie_balance_standing_simple_cfg.py
#
# SIMPLIFIED balance standing task with LEARNABLE rewards
#
# Previous version was TOO HARD:
#   - Penalties too aggressive (-5.0 vel, -8.0 orientation)
#   - Robot couldn't learn to stand
#
# This version:
#   - Start with minimal penalties
#   - Focus on NOT falling (termination avoidance)
#   - Let policy learn balance naturally
#   - Add bonuses for stillness

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
# Simpler Domain Randomization
##


@configclass
class CassieBalanceStandingSimpleEventCfg:
    """Simpler DR for standing - focus on learning first."""

    # Physics material (moderate range)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Mass randomization (moderate)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-1.0, 1.5),
            "operation": "add",
        },
    )

    # Actuator gains (moderate range)
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

    # Joint position reset (small noise to start)
    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.3, 0.3),  # Small variation
            "velocity_range": (-0.2, 0.2),
        },
    )

    # Push perturbations (help learn balance recovery)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),  # Gentle pushes
                "y": (-0.3, 0.3),
            },
        },
    )


##
# Environment Configuration
##


@configclass
class CassieBalanceStandingSimpleEnvCfg(DirectRLEnvCfg):
    """SIMPLIFIED Cassie balance standing - focus on learning.

    Key changes from aggressive version:
      1. Weaker penalties (robot can actually learn)
      2. Positive rewards for success (not just penalties for failure)
      3. Simpler reward structure
      4. Focus on episode length (don't fall = good)
    """

    # Environment
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5  # Normal action scale
    action_space = 10  # passive ankles
    observation_space = 46
    state_space = 0

    # Domain Randomization (simpler)
    events: CassieBalanceStandingSimpleEventCfg = CassieBalanceStandingSimpleEventCfg()

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
    lin_vel_x_range = (0.0, 0.0)
    lin_vel_y_range = (0.0, 0.0)
    ang_vel_z_range = (0.0, 0.0)
    command_resampling_time = 1000.0

    # ========================================================================
    # SIMPLIFIED REWARD STRUCTURE
    # ========================================================================
    #
    # Philosophy: Make it EASY to learn standing
    #   1. Big reward for NOT terminating (survival)
    #   2. Small bonuses for good behavior (stillness, upright)
    #   3. Weak penalties for bad behavior
    #
    # Let the policy learn "don't fall" first, then refine to "stand still"

    # === POSITIVE REWARDS (encourage good behavior) ===

    # Survival bonus (main reward signal)
    # Just staying alive = good
    # Implemented via episode_length reward (built into DirectRLEnv)

    # Upright bonus (positive reward for good orientation)
    # Instead of penalty, give reward for being upright
    flat_orientation_l2_weight = -1.5  # Moderate penalty (was -8.0)

    # Stillness bonus (positive reward for low velocity)
    stillness_bonus_weight = 1.0  # Bonus when velocity < threshold

    # === WEAK PENALTIES (guide behavior gently) ===

    # Velocity penalties (weak - just guide, don't dominate)
    lin_vel_xy_l2_weight = -0.5  # Was -5.0 (10x weaker!)
    lin_vel_z_l2_weight = -0.5   # Was -2.0 (4x weaker!)
    ang_vel_xy_l2_weight = -0.05 # Was -0.2 (4x weaker!)
    ang_vel_z_l2_weight = -0.05  # Was -0.2 (4x weaker!)

    # Action penalties (very weak - allow learning)
    dof_torques_l2_weight = -2.0e-6   # Was -1e-5 (5x weaker!)
    dof_acc_l2_weight = -5.0e-7       # Was -1e-6 (2x weaker!)
    action_rate_l2_weight = -0.01     # Was -0.05 (5x weaker!)

    # Joint deviation (weak)
    joint_deviation_hip_weight = -0.1   # Was -0.5 (5x weaker!)
    joint_deviation_toes_weight = -0.1  # Was -0.3 (3x weaker!)

    # Joint limits (moderate - important for safety)
    dof_pos_limits_weight = -1.0
    dof_pos_limits_soft_ratio = 0.9

    # Feet (weak)
    feet_slide_weight = -0.05  # Was -0.3 (6x weaker!)

    # Disable tracking rewards (not needed for standing)
    track_lin_vel_xy_exp_weight = 0.0
    track_ang_vel_z_exp_weight = 0.0
    feet_air_time_weight = 0.0
    feet_air_time_threshold = 0.0

    # Termination penalty (moderate - don't dominate)
    termination_penalty = -50.0  # Was -200.0 (4x weaker!)

    # === OBSERVATION NOISE (moderate) ===
    add_noise = True
    noise_scales = {
        "lin_vel": 0.05,   # Reduced
        "ang_vel": 0.1,    # Reduced
        "projected_gravity": 0.02,  # Reduced
        "joint_pos": 0.01,
        "joint_vel": 1.0,  # Reduced
    }

    # Reset Randomization (small)
    reset_position_noise = 0.3  # Small
    reset_yaw_noise = math.pi

    def __post_init__(self):
        super().__post_init__()

        # Remove ankle actuators (passive ankles)
        actuator_keys_to_remove = ["ankle_joint", "ankle", "ankle_left", "ankle_right"]

        keys_found = []
        for key in self.robot.actuators.keys():
            if any(x in key for x in actuator_keys_to_remove):
                keys_found.append(key)

        for key in keys_found:
            print(f"[CassieBalanceStandingSimpleEnvCfg] Removing actuator: {key}")
            del self.robot.actuators[key]


@configclass
class CassieBalanceStandingSimpleEnvCfg_PLAY(CassieBalanceStandingSimpleEnvCfg):
    """Play configuration for visualization."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Disable noise during playback
    add_noise = False
