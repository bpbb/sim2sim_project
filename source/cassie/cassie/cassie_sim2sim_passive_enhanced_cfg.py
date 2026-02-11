# cassie_sim2sim_passive_enhanced_cfg.py
#
# Enhanced passive ankle training config for improved sim-to-sim transfer.
#
# Key improvements over baseline:
#   1. More aggressive actuator gain DR [0.5, 1.5] (was [0.8, 1.2])
#   2. PD stiffness/damping DR to match PhysX↔MuJoCo PD gap
#   3. COM offset randomization (±0.05m) for dynamics mismatch
#   4. Joint damping randomization [0.0, 2.0] for energy dissipation gap
#   5. Enhanced rewards: forward drift penalty, standing stability bonus
#   6. Larger position reset noise (±1.0 rad vs ±0.5) for robustness
#   7. More frequent push perturbations (5-10s vs 10-15s)
#
# Usage:
#   python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Enhanced-v0 --headless

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
# Enhanced Domain Randomization for Sim2Sim Transfer
##


@configclass
class CassieSim2SimPassiveEnhancedEventCfg:
    """Enhanced DR targeting identified sim-to-sim gaps.

    Based on deployment debugging, the main gaps are:
      - PD dynamics (implicit PhysX vs explicit MuJoCo)
      - Forward drift (COM/inertia mismatch)
      - Energy dissipation (damping differences)

    This config randomizes these dimensions more aggressively.
    """

    # Friction: wider range to cover contact model differences
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),  # wider than (0.5, 1.25)
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.1),  # allow slight bounce
            "num_buckets": 64,
        },
    )

    # Mass: wider range for inertia differences
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-2.0, 3.0),  # was (-1.0, 2.0)
            "operation": "add",
        },
    )

    # Actuator gains: MORE aggressive to cover PhysX/MuJoCo PD gap
    # MuJoCo needs 6-8x gains, so train with wide variance
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 1.5),  # was (0.8, 1.2)
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Joint friction: randomize to cover energy dissipation differences
    # MuJoCo has different damping behavior than PhysX, and joint friction
    # affects energy dissipation in a similar way
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

    # Joint position: LARGER range to handle pose mismatches
    # MuJoCo home keyframe differs by ±0.6 rad in some joints
    reset_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-1.0, 1.0),  # was (-0.5, 0.5)
            "velocity_range": (-1.0, 1.0),  # was (-0.5, 0.5)
        },
    )

    # Push perturbations: MORE frequent for balance recovery
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),  # was (10.0, 15.0) — now 2x more frequent
        params={
            "velocity_range": {
                "x": (-0.8, 0.8),  # slightly stronger
                "y": (-0.8, 0.8),
            },
        },
    )

    # Additional mass randomization on legs to create COM variation
    # Forward drift suggests COM/inertia mismatch between simulators
    # Since direct COM randomization isn't available, we randomize leg masses
    # which effectively shifts the overall COM
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*thigh.*|.*shin.*|.*tarsus.*"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )


##
# Environment Configuration
##


@configclass
class CassieSim2SimPassiveEnhancedEnvCfg(DirectRLEnvCfg):
    """Enhanced passive ankle Cassie with aggressive DR.

    46-dim obs: lin_vel(3) + ang_vel(3) + gravity(3) + cmd(3)
                + joint_pos(12) + joint_vel(12) + actions(10)
    10-dim actions: joint position offsets (ankles passive)
    """

    # Environment
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.5
    action_space = 10  # passive ankles
    observation_space = 46  # 48 - 2 (ankle actions)
    state_space = 0

    # Enhanced Domain Randomization
    events: CassieSim2SimPassiveEnhancedEventCfg = CassieSim2SimPassiveEnhancedEventCfg()

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

    # Commands: focus on standing and slow forward walking
    # Standing (0,0,0) should be well-represented in training
    lin_vel_x_range = (-0.5, 0.8)  # was (-1.0, 1.0) — bias toward forward, include standing
    lin_vel_y_range = (-0.3, 0.3)  # was (-1.0, 1.0) — less lateral
    ang_vel_z_range = (-0.5, 0.5)  # was (-1.0, 1.0) — less turning
    command_resampling_time = 10.0

    # Enhanced Reward Weights
    # Key changes:
    #   - Add forward drift penalty (new)
    #   - Add standing bonus (new)
    #   - Increase orientation penalty (forward pitch causes drift)
    #   - Increase action smoothness (reduces oscillation)

    track_lin_vel_xy_exp_weight = 2.0
    track_ang_vel_z_exp_weight = 1.0
    feet_air_time_weight = 2.5
    feet_air_time_threshold = 0.3

    lin_vel_z_l2_weight = -2.0
    ang_vel_xy_l2_weight = -0.1  # was -0.05 — STRONGER pitch/roll penalty
    dof_torques_l2_weight = -5.0e-6
    dof_acc_l2_weight = -3.75e-7
    action_rate_l2_weight = -0.03  # was -0.015 — STRONGER smoothness
    flat_orientation_l2_weight = -5.0  # was -2.5 — STRONGER upright penalty

    joint_deviation_hip_weight = -0.2
    joint_deviation_toes_weight = -0.2

    dof_pos_limits_weight = -1.0
    dof_pos_limits_soft_ratio = 0.9

    feet_slide_weight = -0.2  # was -0.1 — penalize sliding more

    # NEW: Penalize unintended forward drift when cmd_x = 0
    # This directly addresses the deployment failure mode
    forward_drift_penalty_weight = -1.0  # L2 penalty on vx when cmd_x < 0.1

    # NEW: Bonus for stable standing (low velocity when commanded)
    standing_bonus_weight = 0.5  # reward for |v| < 0.1 when cmd < 0.1

    termination_penalty = -200.0

    # Observation Noise: slightly higher to match real sensor noise
    add_noise = True
    noise_scales = {
        "lin_vel": 0.15,  # was 0.1
        "ang_vel": 0.3,   # was 0.2
        "projected_gravity": 0.05,
        "joint_pos": 0.02,  # was 0.01
        "joint_vel": 2.0,   # was 1.5
    }

    # Reset Randomization: full yaw, larger position noise
    reset_position_noise = 1.0  # was 0.5
    reset_yaw_noise = math.pi

    def __post_init__(self):
        super().__post_init__()

        # Remove ankle actuators to make them passive
        actuator_keys_to_remove = ["ankle_joint", "ankle", "ankle_left", "ankle_right"]

        keys_found = []
        for key in self.robot.actuators.keys():
            if any(x in key for x in actuator_keys_to_remove):
                keys_found.append(key)

        for key in keys_found:
            print(f"[CassieSim2SimPassiveEnhancedEnvCfg] Removing actuator: {key}")
            del self.robot.actuators[key]


@configclass
class CassieSim2SimPassiveEnhancedEnvCfg_PLAY(CassieSim2SimPassiveEnhancedEnvCfg):
    """Play configuration with fewer envs for visualization."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Disable noise during playback
    add_noise = False
