#!/usr/bin/env python3
"""Improved Flamingo Isaac Lab to MuJoCo Transfer Script.

This script transfers policies trained in Isaac Lab to MuJoCo for the Flamingo robot.
Key fixes from the original script:
1. Correct joint order mapping between Isaac Lab and MuJoCo
2. Proper PD control implementation matching Isaac Lab actuators
3. Correct observation construction with proper scaling
4. Proper gear ratio handling for leg joints
5. Correct command scaling

Based on the trained env.yaml configuration:
- Observations: 88 dims (3×28 stacked + 4 command)
- Actions: 8 dims (2 hip + 2 shoulder + 2 leg + 2 wheel)

Usage:
    python scripts/transfer_flamingo_mujoco_improved.py --policy logs/.../model_4999.pt --xml assets/flamingo_torque.xml
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
# CONSTANTS (from env.yaml)
# ═══════════════════════════════════════════════════════════════════════════════

# Input/Output Dimensions
NUM_OBS = 88           # Total observation dimension (3×28 stacked + 4 command)
SINGLE_OBS_DIM = 28    # Single frame observation dimension
STACK_FRAMES = 3       # Number of history frames
NUM_ACTIONS = 8        # Number of actions

# Observation scaling (from env.yaml)
DOF_VEL_SCALE = 0.15   # Joint velocity scale
ANG_VEL_SCALE = 0.25   # Angular velocity scale
LEG_GEAR_RATIO = -1.5  # Gear ratio for leg joints

# Command scaling (from env.yaml velocity_commands)
CMD_LIN_VEL_X_SCALE = 2.0
CMD_LIN_VEL_Y_SCALE = 0.0  # Y velocity disabled for 2-wheel bot
CMD_ANG_VEL_Z_SCALE = 0.25
CMD_POS_Z_SCALE = 1.0

# Action scaling (from env.yaml actions)
ACTION_SCALES = np.array([
    1.0, 1.0,    # hip_joint_pos (left, right)
    1.0, 1.0,    # shoulder_joint_pos (left, right)
    1.0, 1.0,    # leg_joint_pos (left, right)
    20.0, 20.0   # wheel_vel (left, right)
], dtype=np.float32)

# PD controller gains (from env.yaml actuators)
PD_GAINS = {
    'hip': {'kp': 100.0, 'kd': 1.5},
    'shoulder': {'kp': 100.0, 'kd': 1.5},
    'leg': {'kp': 120.0, 'kd': 1.5},  # Note: gear_ratio applied separately
    'wheel': {'kp': 0.0, 'kd': 0.7},  # Velocity control (Kp=0)
}

# MuJoCo joint indices mapping
# qpos layout: [0:3]=pos, [3:7]=quat, [7:]=joints
# qvel layout: [0:3]=lin_vel, [3:6]=ang_vel, [6:]=joint_vel
MUJOCO_JOINT_ORDER = {
    'left_hip': {'qpos': 7, 'qvel': 6},
    'left_shoulder': {'qpos': 8, 'qvel': 7},
    'left_leg': {'qpos': 9, 'qvel': 8},
    'left_wheel': {'qpos': 10, 'qvel': 9},
    'right_hip': {'qpos': 11, 'qvel': 10},
    'right_shoulder': {'qpos': 12, 'qvel': 11},
    'right_leg': {'qpos': 13, 'qvel': 12},
    'right_wheel': {'qpos': 14, 'qvel': 13},
}

# Actuator indices in MuJoCo ctrl array (matches XML actuator order)
MUJOCO_ACTUATOR_ORDER = {
    'left_hip': 0,
    'right_hip': 1,
    'left_shoulder': 2,
    'right_shoulder': 3,
    'left_leg': 4,
    'right_leg': 5,
    'left_wheel': 6,
    'right_wheel': 7,
}

# Default joint positions (standing pose)
DEFAULT_JOINT_POS = np.zeros(8, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """Actor-Critic network matching the trained model architecture."""

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
    """Load policy from checkpoint."""
    print(f"Loading policy from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    policy = ActorCritic(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # Load only actor weights
    model_state = checkpoint["model_state_dict"]
    actor_state = {k: v for k, v in model_state.items() if k.startswith("actor.")}
    policy.load_state_dict(actor_state, strict=False)
    policy.eval()

    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Compute gravity vector in body frame from quaternion (wxyz format).

    Gravity in world frame is [0, 0, -1].
    This returns the gravity vector rotated into the body frame.
    """
    qw, qx, qy, qz = quat_wxyz

    # Compute R^T @ [0, 0, -1] = -R^T @ [0, 0, 1]
    gx = -2.0 * (qx * qz - qw * qy)
    gy = -2.0 * (qy * qz + qw * qx)
    gz = -(1.0 - 2.0 * (qx * qx + qy * qy))

    return np.array([gx, gy, gz], dtype=np.float32)


def world_to_body_velocity(quat_wxyz: np.ndarray, vel_world: np.ndarray) -> np.ndarray:
    """Transform velocity from world frame to body frame."""
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat_wxyz)
    rot_mat = rot_mat.reshape(3, 3)
    return rot_mat.T @ vel_world


def pd_control(target_pos: float, current_pos: float, current_vel: float,
               kp: float, kd: float) -> float:
    """Compute PD control torque."""
    return kp * (target_pos - current_pos) + kd * (0.0 - current_vel)


def velocity_control(target_vel: float, current_vel: float, kd: float) -> float:
    """Compute velocity control torque (Kp=0 PD)."""
    return kd * (target_vel - current_vel)


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def construct_observation(data: mujoco.MjData, last_action: np.ndarray) -> np.ndarray:
    """Construct single frame observation matching Isaac Lab format.

    Observation order (28 dims):
    1. hip_shoulder_joint_pos (4): left_hip, right_hip, left_shoulder, right_shoulder
    2. leg_joint_pos (2): left_leg, right_leg (× gear_ratio)
    3. hip_shoulder_joint_vel (4): velocities × 0.15
    4. leg_joint_vel (2): velocities × gear_ratio × 0.15
    5. wheel_joint_vel (2): velocities × 0.15
    6. base_ang_vel (3): body frame × 0.25
    7. base_projected_gravity (3): body frame
    8. last_action (8): raw action values
    """
    obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
    idx = 0

    # 1. hip_shoulder_joint_pos (4)
    # Order from env.yaml: ".*_hip_joint", ".*_shoulder_joint" with preserve_order=false
    # Note: right_shoulder has inverted axis (axis="0 0 -1"), negate position
    obs[idx] = data.qpos[MUJOCO_JOINT_ORDER['left_hip']['qpos']]
    obs[idx + 1] = data.qpos[MUJOCO_JOINT_ORDER['right_hip']['qpos']]
    obs[idx + 2] = data.qpos[MUJOCO_JOINT_ORDER['left_shoulder']['qpos']]
    obs[idx + 3] = -data.qpos[MUJOCO_JOINT_ORDER['right_shoulder']['qpos']]  # Axis inverted
    idx += 4

    # 2. leg_joint_pos (2) with gear ratio
    # Note: right_leg has inverted axis (axis="0 0 -1"), negate position
    obs[idx] = data.qpos[MUJOCO_JOINT_ORDER['left_leg']['qpos']] * LEG_GEAR_RATIO
    obs[idx + 1] = -data.qpos[MUJOCO_JOINT_ORDER['right_leg']['qpos']] * LEG_GEAR_RATIO  # Axis inverted
    idx += 2

    # 3. hip_shoulder_joint_vel (4) scaled
    # Note: right_shoulder has inverted axis
    obs[idx] = data.qvel[MUJOCO_JOINT_ORDER['left_hip']['qvel']] * DOF_VEL_SCALE
    obs[idx + 1] = data.qvel[MUJOCO_JOINT_ORDER['right_hip']['qvel']] * DOF_VEL_SCALE
    obs[idx + 2] = data.qvel[MUJOCO_JOINT_ORDER['left_shoulder']['qvel']] * DOF_VEL_SCALE
    obs[idx + 3] = -data.qvel[MUJOCO_JOINT_ORDER['right_shoulder']['qvel']] * DOF_VEL_SCALE  # Axis inverted
    idx += 4

    # 4. leg_joint_vel (2) with gear ratio and scaling
    # Note: right_leg has inverted axis
    obs[idx] = data.qvel[MUJOCO_JOINT_ORDER['left_leg']['qvel']] * LEG_GEAR_RATIO * DOF_VEL_SCALE
    obs[idx + 1] = -data.qvel[MUJOCO_JOINT_ORDER['right_leg']['qvel']] * LEG_GEAR_RATIO * DOF_VEL_SCALE  # Axis inverted
    idx += 2

    # 5. wheel_joint_vel (2) scaled
    # Note: Right wheel axis is inverted in MuJoCo, negate velocity for observation
    obs[idx] = data.qvel[MUJOCO_JOINT_ORDER['left_wheel']['qvel']] * DOF_VEL_SCALE
    obs[idx + 1] = -data.qvel[MUJOCO_JOINT_ORDER['right_wheel']['qvel']] * DOF_VEL_SCALE  # Negate for axis
    idx += 2

    # 6. base_ang_vel (3) in body frame, scaled
    quat_wxyz = data.qpos[3:7]  # MuJoCo uses wxyz
    ang_vel_world = data.qvel[3:6]
    ang_vel_body = world_to_body_velocity(quat_wxyz, ang_vel_world)
    obs[idx:idx + 3] = ang_vel_body * ANG_VEL_SCALE
    idx += 3

    # 7. base_projected_gravity (3)
    obs[idx:idx + 3] = get_projected_gravity(quat_wxyz)
    idx += 3

    # 8. last_action (8)
    obs[idx:idx + 8] = last_action
    idx += 8

    assert idx == SINGLE_OBS_DIM, f"Observation size mismatch: {idx} != {SINGLE_OBS_DIM}"
    return obs


def construct_command(cmd_vx: float, cmd_vy: float, cmd_wz: float, cmd_z: float = 0.0) -> np.ndarray:
    """Construct scaled command vector."""
    return np.array([
        cmd_vx * CMD_LIN_VEL_X_SCALE,
        cmd_vy * CMD_LIN_VEL_Y_SCALE,
        cmd_wz * CMD_ANG_VEL_Z_SCALE,
        cmd_z * CMD_POS_Z_SCALE,  # Height command (usually 0 for flat terrain)
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROL APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_actions(data: mujoco.MjData, actions: np.ndarray):
    """Apply policy actions to MuJoCo actuators with proper PD control.

    Action order (from env.yaml):
    [0:2] = hip_joint_pos (left, right) × 1.0
    [2:6] = shoulder_leg_joint_pos (left_shoulder, right_shoulder, left_leg, right_leg) × 1.0
    [6:8] = wheel_vel (left, right) × 20.0
    """
    # Scale actions
    scaled_actions = actions * ACTION_SCALES

    # Extract targets
    hip_targets = scaled_actions[0:2]       # left_hip, right_hip
    shoulder_targets = scaled_actions[2:4]  # left_shoulder, right_shoulder
    leg_targets = scaled_actions[4:6]       # left_leg, right_leg
    wheel_targets = scaled_actions[6:8]     # left_wheel, right_wheel (velocity)

    # Compute torques with PD control
    torques = np.zeros(8, dtype=np.float32)

    # Hip joints (position control)
    kp_hip, kd_hip = PD_GAINS['hip']['kp'], PD_GAINS['hip']['kd']
    torques[MUJOCO_ACTUATOR_ORDER['left_hip']] = pd_control(
        hip_targets[0],
        data.qpos[MUJOCO_JOINT_ORDER['left_hip']['qpos']],
        data.qvel[MUJOCO_JOINT_ORDER['left_hip']['qvel']],
        kp_hip, kd_hip
    )
    torques[MUJOCO_ACTUATOR_ORDER['right_hip']] = pd_control(
        hip_targets[1],
        data.qpos[MUJOCO_JOINT_ORDER['right_hip']['qpos']],
        data.qvel[MUJOCO_JOINT_ORDER['right_hip']['qvel']],
        kp_hip, kd_hip
    )

    # Shoulder joints (position control)
    # Note: right_shoulder has inverted axis (axis="0 0 -1")
    kp_sho, kd_sho = PD_GAINS['shoulder']['kp'], PD_GAINS['shoulder']['kd']
    torques[MUJOCO_ACTUATOR_ORDER['left_shoulder']] = pd_control(
        shoulder_targets[0],
        data.qpos[MUJOCO_JOINT_ORDER['left_shoulder']['qpos']],
        data.qvel[MUJOCO_JOINT_ORDER['left_shoulder']['qvel']],
        kp_sho, kd_sho
    )
    # For inverted axis, negate target (torque direction is also inverted by axis)
    torques[MUJOCO_ACTUATOR_ORDER['right_shoulder']] = pd_control(
        -shoulder_targets[1],  # Negate target for axis inversion
        data.qpos[MUJOCO_JOINT_ORDER['right_shoulder']['qpos']],
        data.qvel[MUJOCO_JOINT_ORDER['right_shoulder']['qvel']],
        kp_sho, kd_sho
    )

    # Leg joints (position control with gear ratio consideration)
    # Note: right_leg has inverted axis (axis="0 0 -1")
    kp_leg, kd_leg = PD_GAINS['leg']['kp'], PD_GAINS['leg']['kd']
    torques[MUJOCO_ACTUATOR_ORDER['left_leg']] = pd_control(
        leg_targets[0],
        data.qpos[MUJOCO_JOINT_ORDER['left_leg']['qpos']],
        data.qvel[MUJOCO_JOINT_ORDER['left_leg']['qvel']],
        kp_leg, kd_leg
    )
    # For inverted axis, negate target
    torques[MUJOCO_ACTUATOR_ORDER['right_leg']] = pd_control(
        -leg_targets[1],  # Negate target for axis inversion
        data.qpos[MUJOCO_JOINT_ORDER['right_leg']['qpos']],
        data.qvel[MUJOCO_JOINT_ORDER['right_leg']['qvel']],
        kp_leg, kd_leg
    )

    # Wheel joints (velocity control: Kp=0)
    # Note: Right wheel has inverted axis in MuJoCo XML (axis="0 0 -1")
    # Negate velocity command for right wheel to compensate
    kd_wheel = PD_GAINS['wheel']['kd']
    torques[MUJOCO_ACTUATOR_ORDER['left_wheel']] = velocity_control(
        wheel_targets[0],
        data.qvel[MUJOCO_JOINT_ORDER['left_wheel']['qvel']],
        kd_wheel
    )
    torques[MUJOCO_ACTUATOR_ORDER['right_wheel']] = velocity_control(
        -wheel_targets[1],  # Negate for right wheel axis inversion
        data.qvel[MUJOCO_JOINT_ORDER['right_wheel']['qvel']],
        kd_wheel
    )

    # Clip torques to actuator limits
    # Note: Use higher software limits than XML to allow full actuator response
    # (matching sim2sim_onnx behavior which uses -60/+60 for joints)
    joint_torque_limit = 60.0   # Software limit
    wheel_torque_limit = 36.0   # Software limit (matching sim2sim_onnx)

    torques[0:6] = np.clip(torques[0:6], -joint_torque_limit, joint_torque_limit)
    torques[6:8] = np.clip(torques[6:8], -wheel_torque_limit, wheel_torque_limit)

    # Apply to MuJoCo
    data.ctrl[:] = torques


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Improved Flamingo policy transfer to MuJoCo")
    parser.add_argument("--policy", type=str, required=True, help="Path to policy checkpoint (.pt)")
    parser.add_argument("--xml", type=str, required=True, help="Path to MuJoCo XML model file")
    parser.add_argument("--cmd_vx", type=float, default=0.0, help="Forward velocity command (m/s)")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="Lateral velocity command (m/s)")
    parser.add_argument("--cmd_wz", type=float, default=0.0, help="Angular velocity command (rad/s)")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration (s)")
    parser.add_argument("--decimation", type=int, default=4, help="Control decimation (default: 4)")

    args = parser.parse_args()

    if not os.path.exists(args.xml):
        raise FileNotFoundError(f"MuJoCo XML file not found: {args.xml}")
    if not os.path.exists(args.policy):
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy}")

    # Load MuJoCo model
    print(f"Loading MuJoCo model from: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Load policy
    policy = load_policy(args.policy)

    # Simulation settings
    simulation_dt = 0.005  # 200 Hz physics
    model.opt.timestep = simulation_dt
    control_dt = simulation_dt * args.decimation  # Control at 50 Hz (decimation=4)

    # Reset state
    mujoco.mj_resetData(model, data)
    # Set initial height (from env.yaml: init_state.pos = [0, 0, 0.5562])
    data.qpos[2] = 0.45  # Slightly lower for safety
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    mujoco.mj_forward(model, data)

    # Buffers
    obs_history = collections.deque(maxlen=STACK_FRAMES)
    for _ in range(STACK_FRAMES):
        obs_history.append(np.zeros(SINGLE_OBS_DIM, dtype=np.float32))

    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # Command
    command = construct_command(args.cmd_vx, args.cmd_vy, args.cmd_wz)
    print(f"Command: vx={args.cmd_vx:.2f}, vy={args.cmd_vy:.2f}, wz={args.cmd_wz:.2f}")
    print(f"Scaled command: {command}")

    # Viewer
    print("Starting simulation...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        step_count = 0
        control_step = 0

        while viewer.is_running() and (time.time() - start_time) < args.duration:
            step_start = time.time()

            # Run physics decimation steps
            for _ in range(args.decimation):
                mujoco.mj_step(model, data)

            # Get observation
            current_obs = construct_observation(data, last_action)
            obs_history.append(current_obs)

            # Stack observations (newest first: [t, t-1, t-2])
            # Note: deque stores [oldest, ..., newest], so reverse for correct order
            stacked_obs = np.concatenate(list(reversed(obs_history)))

            # Final input: stacked observations + command
            final_input = np.concatenate([stacked_obs, command])

            # Inference
            with torch.no_grad():
                input_tensor = torch.from_numpy(final_input).unsqueeze(0).float()
                raw_action = policy(input_tensor).numpy().squeeze()

            # Clip action
            raw_action = np.clip(raw_action, -1.0, 1.0)

            # Apply action with PD control
            apply_actions(data, raw_action)

            # Store for next observation
            last_action = raw_action.copy()

            control_step += 1
            step_count += args.decimation

            # Debug output every second
            if control_step % 50 == 0:
                height = data.qpos[2]
                vel_x = data.qvel[0]
                print(f"t={time.time()-start_time:.1f}s | height={height:.3f}m | vx={vel_x:.2f}m/s | action_mean={np.mean(np.abs(raw_action)):.3f}")

            # Sync viewer
            viewer.sync()

            # Real-time sync
            elapsed = time.time() - step_start
            target_dt = control_dt
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    print("Simulation finished.")


if __name__ == "__main__":
    main()