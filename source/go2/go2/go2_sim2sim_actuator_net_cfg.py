# Copyright (c) 2024, Boonyaporn Preechasuth
# Go2 sim2sim config with trained ActuatorNet replacing DCMotor PD controller.
#
# IDENTICAL to baseline Go2Sim2SimEnvCfg except the robot uses a trained
# neural network actuator instead of analytical PD (Kp/Kd).
# This enables fair comparison: same obs, rewards, DR, network — only the
# actuator model differs.

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from .go2_sim2sim_cfg import Go2Sim2SimEnvCfg, Go2Sim2SimSceneCfg, UNITREE_GO2_SIM2SIM_CFG

# Import actuator net config
import sys
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _project_root)
from actuator_net.isaac_actuator import ActuatorNetCfg

# ── Robot: same as baseline but with ActuatorNet instead of DCMotor ──────────

UNITREE_GO2_ACTUATOR_NET_CFG = UNITREE_GO2_SIM2SIM_CFG.replace(
    actuators={
        "base_legs": ActuatorNetCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            model_path=os.path.join(_project_root, "actuator_net/models/go2/actuator_net_jit.pt"),
            scaler_path=os.path.join(_project_root, "actuator_net/models/go2/feature_scaler.pkl"),
            num_model_joints=12,
            effort_limit=30.0,
            velocity_limit=100.0,
            fallback_stiffness=20.0,
            fallback_damping=0.5,
        ),
    },
)


@configclass
class Go2ActuatorNetSceneCfg(Go2Sim2SimSceneCfg):
    """Scene — identical to baseline except robot uses ActuatorNet."""

    robot: ArticulationCfg = UNITREE_GO2_ACTUATOR_NET_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ── Env configs (inherit everything else from baseline) ──────────────────────

@configclass
class Go2Sim2SimActuatorNetEnvCfg(Go2Sim2SimEnvCfg):
    """Identical to baseline — only the actuator model changes."""

    scene: Go2ActuatorNetSceneCfg = Go2ActuatorNetSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class Go2Sim2SimActuatorNetEnvCfg_PLAY(Go2Sim2SimActuatorNetEnvCfg):
    """Play configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
