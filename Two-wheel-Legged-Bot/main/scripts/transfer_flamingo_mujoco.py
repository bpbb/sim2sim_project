#!/usr/bin/env python3
"""Flamingo Isaac Lab to MuJoCo Transfer Script.

This script transfers policies trained in Isaac Lab to MuJoCo for the Flamingo robot.
It handles:
1.  Loading the policy checkpoint (.pt).
2.  Loading the MuJoCo XML model.
3.  Constructing the observation vector (history stacking + command).
4.  Running the control loop (position control for joints, velocity for wheels).

Usage:
    python scripts/transfer_flamingo_mujoco.py --policy logs/.../model_4999.pt --xml path/to/flamingo.xml
"""

import argparse
import os
import time
import collections

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

# Input Dimensions
NUM_OBS = 88         # Total Input Dimension (Stack * 3 + Command)
SINGLE_OBS_DIM = 28  # Dimension of a single frame observation
STACK_FRAMES = 3     # Number of history frames (t, t-1, t-2)
NUM_ACTIONS = 8      # Number of actions (2 hips, 2 shoulders, 2 legs, 2 wheels)

# Observation Indices (based on `env.yaml`)
# 1. hip_shoulder_joint_pos (4)
# 2. leg_joint_pos (2)
# 3. hip_shoulder_joint_vel (4)
# 4. joint_vel (leg) (2)
# 5. wheel_joint_vel (2)
# 6. base_ang_vel (3)
# 7. base_projected_gravity (3)
# 8. actions (8)
# Total = 4 + 2 + 4 + 2 + 2 + 3 + 3 + 8 = 28

# Default Joint Positions (for normalization/offset)
# Order: [FL_hip, FR_hip, FL_shoulder, FR_shoulder, FL_leg, FR_leg, FL_wheel, FR_wheel]
# Approximate zeros or standing pose.
DEFAULT_JOINT_POS = np.zeros(8, dtype=np.float32)

# Scales (from `env.yaml`)
LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.15 # scaled
WHEEL_VEL_SCALE = 0.15

# Gear Ratio for Legs
LEG_GEAR_RATIO = -1.5


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """RSL-RL compatible Actor-Critic network (Actor only for inference)."""

    def __init__(
        self,
        num_obs: int = 88,
        num_actions: int = 8,
        actor_hidden_dims: list = [512, 256, 128],
        activation: str = "elu",
    ):
        super().__init__()
        activation_fn = {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        actor_layers = []
        prev_dim = num_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.append(nn.Linear(prev_dim, hidden_dim))
            actor_layers.append(activation_fn)
            prev_dim = hidden_dim
        actor_layers.append(nn.Linear(prev_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.act(observations)


def load_policy(checkpoint_path: str, num_obs: int = 88, num_actions: int = 8):
    """Load policy from RSL-RL checkpoint."""
    print(f"Loading policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) # weights_only=False needed for older torch versions/structures
    policy = ActorCritic(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    model_state = checkpoint["model_state_dict"]
    actor_state = {k: v for k, v in model_state.items() if k.startswith("actor.")}
    policy.load_state_dict(actor_state, strict=False)
    policy.eval()
    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Compute gravity vector in body frame from quaternion (wxyz format)."""
    qw, qx, qy, qz = quaternion
    # Gravity in world is [0, 0, -1]. We want to project it to body frame.
    # Rotation matrix R maps body to world. v_world = R * v_body
    # v_body = R^T * v_world.
    # We want -R^T * [0, 0, 1]. This corresponds to the last row of R.
    return np.array([
        2 * (qx * qz - qw * qy),
        2 * (qy * qz + qw * qx),
        1 - 2 * (qx * qx + qy * qy)
    ]) * -1.0 # Gravity is down


def world_to_body_velocity(quat_wxyz: np.ndarray, lin_vel_w: np.ndarray, ang_vel_w: np.ndarray):
    """Transform world frame velocities to body frame."""
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat_wxyz)
    rot_mat = rot_mat.reshape(3, 3)
    lin_vel_b = rot_mat.T @ lin_vel_w # Rotate world vel to body frame
    ang_vel_b = rot_mat.T @ ang_vel_w
    return lin_vel_b, ang_vel_b


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Transfer Flamingo policy to MuJoCo")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--xml", type=str, required=True, help="Path to MuJoCo XML model file")
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Velocity command [vx, vy, wz] (default: 0 0 0)")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds")
    
    args = parser.parse_args()

    if not os.path.exists(args.xml):
        raise FileNotFoundError(f"MuJoCo XML file not found: {args.xml}")

    # 1. Load MuJoCo Model
    print(f"Loading MuJoCo model from: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # 2. Load Policy
    policy = load_policy(args.policy)

    # 3. Simulation Setup
    simulation_dt = 0.005 # 200 Hz
    model.opt.timestep = simulation_dt
    
    # Reset State
    mujoco.mj_resetData(model, data)
    # Set initial pose - slightly above ground
    data.qpos[2] = 0.45 # Z height
    mujoco.mj_forward(model, data)

    # Buffers
    obs_history = collections.deque(maxlen=STACK_FRAMES)
    # Initialize history with zeros
    for _ in range(STACK_FRAMES):
        obs_history.append(np.zeros(SINGLE_OBS_DIM, dtype=np.float32))

    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    
    # Action Scales
    # Order: [L_Hip, R_Hip, L_Sho, R_Sho, L_Leg, R_Leg, L_Wheel, R_Wheel] (Checking this assumption!)
    # Based on `ActionsCfg` in `velocity_env_cfg.py`:
    # hip_joint_pos (L, R) scale=1.0
    # shoulder_leg_joint_pos (L_Sho, R_Sho, L_Leg, R_Leg) scale=1.0
    # wheel_vel (L, R) scale=20.0
    
    # We need to map Policy Output (8 dims) to Actuators.
    # Policy Output Order: 
    # [L_Hip, R_Hip, L_Sho, R_Sho, L_Leg, R_Leg, L_Wheel, R_Wheel]
    
    # Action Scales Array
    action_scales = np.array([
        1.0, 1.0,   # Hips
        1.0, 1.0,   # Shoulders
        1.0, 1.0,   # Legs
        20.0, 20.0  # Wheels
    ], dtype=np.float32)


    # Command
    cmd_vx, cmd_vy, cmd_wz = args.cmd
    print(f"Command: vx={cmd_vx}, vy={cmd_vy}, wz={cmd_wz}")

    
    # Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        step_count = 0
        
        while viewer.is_running() and (time.time() - start_time) < args.duration:
            step_start = time.time()

            # ─── 1. Get Observations ──────────────────────────────────────────
            
            # Joint States
            # Assumption: MuJoCo joints are ordered similarly to Isaac, or we need names.
            # Using qpos/qvel directly for now.
            # qpos [7:] starts joints (after 3 pos + 4 quat)
            
            # TODO: Map joints by name for robustness if needed.
            # For now assuming standard order in XML.
            
            qpos = data.qpos[7:] 
            qvel = data.qvel[6:]

            # Construct Observation Vector (28 dims)
            current_obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
            
            # --- Fill Observation ---
            # 1. hip_shoulder_joint_pos (4)
            current_obs[0:4] = qpos[0:4] # Hips(2) + Shoulders(2)
            
            # 2. leg_joint_pos (2) - With Gear Ratio!
            current_obs[4:6] = qpos[4:6] * LEG_GEAR_RATIO
            
            # 3. hip_shoulder_joint_vel (4) - Scaled
            current_obs[6:10] = qvel[0:4] * DOF_VEL_SCALE
            
            # 4. joint_vel (leg) (2) - With Gear Ratio & Scaled
            current_obs[10:12] = (qvel[4:6] * LEG_GEAR_RATIO) * DOF_VEL_SCALE
            
            # 5. wheel_joint_vel (2) - Scaled
            current_obs[12:14] = qvel[6:8] * DOF_VEL_SCALE
            
            # 6. base_ang_vel (3) - Scaled, in Body Frame
            quat = data.qpos[3:7] # wxyz
            ang_vel_world = data.qvel[3:6]
            _, ang_vel_body = world_to_body_velocity(quat, np.zeros(3), ang_vel_world)
            current_obs[14:17] = ang_vel_body * ANG_VEL_SCALE
            
            # 7. base_projected_gravity (3) - Body Frame
            base_gravity = get_gravity_orientation(quat)
            current_obs[17:20] = base_gravity
            
            # 8. actions (8)
            current_obs[20:28] = last_action


            # Add to History
            obs_history.append(current_obs)

            # Construct Final Input (Stacked + Command)
            # Stacked: [t, t-1, t-2] -> 28 * 3 = 84
            # Command: [vx*2.0, vy*0.0, wz*0.25, z*1.0] -> 4 (Scale from env.yaml)
            
            stacked_obs = np.concatenate(list(obs_history))
            
            # Scale Commands
            scaled_cmd = np.array([
                cmd_vx * 2.0,
                cmd_vy * 0.0, # Usually 0 for 2-wheeled
                cmd_wz * 0.25,
                0.0 # Height/Z command placeholder? Used z=0 in original logic? Checking YAML: pos_z=(0.19.., 0.35..). 
                    # Warning: The env.yaml has `pos_z` in ranges, but `velocity_commands` obs term likely includes it if defined. 
                    # Usually `generated_scaled_commands` outputs the command values.
                    # Verify input dim 88. 84 (stack) + 4 (cmd). 
            ], dtype=np.float32)
            
            final_input = np.concatenate([stacked_obs, scaled_cmd])
            
            # ─── 2. Inference ────────────────────────────────────────────────
            with torch.no_grad():
                input_tensor = torch.from_numpy(final_input).unsqueeze(0).float()
                raw_action = policy(input_tensor).numpy().squeeze()
            
            # Clip Action
            raw_action = np.clip(raw_action, -1.0, 1.0) # Standard clip
            
            # ─── 3. Control ──────────────────────────────────────────────────
            
            # Scale Action
            scaled_action = raw_action * action_scales
            
            # Apply to Actuators
            # Assuming Position Control for Joints, Velocity/Torque for Wheels?
            # Config check:
            # - Joints (hips, shoulders, legs): DelayedPDActuator -> Position Target = Action * Scale + Default
            # - Wheels: DelayedPDActuator -> But stiffness=0, damping=0.7. This implies Torque/Velocity control.
            #   In Isaac: `wheel_vel` action maps to Velocity Target.
            
            # MuJoCo `ctrl`:
            # By default, needs to match actuator definitions in XML.
            # If XML uses `position` actuators for joints and `velocity` for wheels:
            
            # Compute Targets
            # Joints (Indices 0-5): Position = Current? No, usually Action is delta or target.
            # YAML: `use_default_offset=False`. So Action * Scale IS the target position? 
            # Wait, `JointPositionActionCfg`. If `use_default_offset=False`, then target = scale * action.
            # But usually we add default pose. The Config says False.
            
            # Wheels (Indices 6-7): Velocity Target = Scale * Action.
            
            data.ctrl[:] = scaled_action 
            # Note: This assumes the XML actuators are configured correctly (Pos for joints, Vel for wheels).

            # Step Physics
            mujoco.mj_step(model, data)
            
            last_action = raw_action.copy()
            step_count += 1
            
            # Viewer Sync
            viewer.sync()
            
            # Real-time
            elapsed = time.time() - step_start
            if elapsed < simulation_dt:
                time.sleep(simulation_dt - elapsed)

    print("Simulation finished.")

if __name__ == "__main__":
    main()
