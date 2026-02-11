# Copyright (c) 2024, Boonyaporn Preechasuth
# Unitree Go2 config for sim2sim transfer (Isaac Lab → MuJoCo)

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg

# Import built-in Go2 config
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##############################################################################
# ROBOT CONFIGURATION - Based on built-in, with sim2sim transfer settings
##############################################################################

# Start with built-in config and modify for sim2sim compatibility
UNITREE_GO2_SIM2SIM_CFG = UNITREE_GO2_CFG.replace(
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.42),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            saturation_effort=30.0,
            effort_limit=30.0,
            velocity_limit=100.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)


##############################################################################
# SCENE CONFIGURATION
##############################################################################

@configclass
class Go2Sim2SimSceneCfg(InteractiveSceneCfg):
    """Scene config for Go2 sim2sim transfer."""

    # Flat ground (no rough terrain)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # Robot - using modified built-in config
    robot: ArticulationCfg = UNITREE_GO2_SIM2SIM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


##############################################################################
# OBSERVATIONS - 48 dimensions, no height scan
#
# obs[0:3]   = base_lin_vel * 2.0
# obs[3:6]   = base_ang_vel * 0.25
# obs[6:9]   = projected_gravity
# obs[9:12]  = commands
# obs[12:24] = joint_pos - default
# obs[24:36] = joint_vel * 0.05
# obs[36:48] = last_action
##############################################################################

@configclass
class Go2Sim2SimObservationsCfg:
    """Observation config (48 dimensions, no height scan)."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations - ORDER MATTERS!"""

        # [0:3] Linear velocity in body frame, scaled by 2.0
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)

        # [3:6] Angular velocity in body frame, scaled by 0.25
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)

        # [6:9] Projected gravity
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # [9:12] Velocity commands
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"}
        )

        # [12:24] Joint positions (relative to default)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        # [24:36] Joint velocities, scaled by 0.05
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)

        # [36:48] Last action
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##############################################################################
# ACTIONS
##############################################################################

@configclass
class Go2Sim2SimActionsCfg:
    """Action config for sim2sim transfer."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


##############################################################################
# COMMANDS
##############################################################################

@configclass
class Go2Sim2SimCommandsCfg:
    """Command config."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


##############################################################################
# REWARDS
##############################################################################

@configclass
class Go2Sim2SimRewardsCfg:
    """Reward config."""

    # Tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Regularization
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # Additional rewards
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Feet rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf"]),
            "threshold": 1.0,
        },
    )


##############################################################################
# TERMINATIONS
##############################################################################

@configclass
class Go2Sim2SimTerminationsCfg:
    """Termination config."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


##############################################################################
# EVENTS (Domain Randomization)
##############################################################################

@configclass
class Go2Sim2SimEventCfg:
    """Event config for domain randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


##############################################################################
# MAIN ENVIRONMENT CONFIG
##############################################################################

@configclass
class Go2Sim2SimEnvCfg(ManagerBasedRLEnvCfg):
    """
    Go2 environment config for sim2sim transfer (Isaac Lab to MuJoCo).

    Key settings:
    - Physics: 200 Hz (dt=0.005s)
    - Policy: 50 Hz (decimation=4)
    - Kp=20, Kd=0.5
    - action_scale=0.25
    - Torque limit: 30 N-m
    - Observation: 48 dims (no height scan)
    """

    # Scene
    scene: Go2Sim2SimSceneCfg = Go2Sim2SimSceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP
    observations: Go2Sim2SimObservationsCfg = Go2Sim2SimObservationsCfg()
    actions: Go2Sim2SimActionsCfg = Go2Sim2SimActionsCfg()
    commands: Go2Sim2SimCommandsCfg = Go2Sim2SimCommandsCfg()
    rewards: Go2Sim2SimRewardsCfg = Go2Sim2SimRewardsCfg()
    terminations: Go2Sim2SimTerminationsCfg = Go2Sim2SimTerminationsCfg()
    events: Go2Sim2SimEventCfg = Go2Sim2SimEventCfg()

    # No curriculum for flat terrain
    curriculum = None

    # Episode settings
    episode_length_s = 20.0

    # Decimation: 200Hz physics / 4 = 50Hz policy
    decimation = 4

    # Simulation
    sim = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=4,
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4


@configclass
class Go2Sim2SimEnvCfg_PLAY(Go2Sim2SimEnvCfg):
    """Play configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
