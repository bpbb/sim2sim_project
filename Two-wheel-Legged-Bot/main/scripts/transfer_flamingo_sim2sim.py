#!/usr/bin/env python3
"""Sim2Sim Transfer Script for Flamingo Robot.

Transfers the working Flamingo_Flat_Stand_Drive policy to MuJoCo.

Observation structure (88 dims total, verified from checkpoint actor.0.weight shape):
- Stacked frames (3 x 28 = 84 dims), newest first:
  Per frame (28 dims):
    [0:4]   hip_shoulder_joint_pos: left_hip, right_hip, left_shoulder, right_shoulder
    [4:6]   leg_joint_pos * gear_ratio(-1.5): left_leg, right_leg
    [6:10]  hip_shoulder_joint_vel * 0.15
    [10:12] leg_joint_vel * gear_ratio(-1.5) * 0.15
    [12:14] wheel_joint_vel * 0.15
    [14:17] base_ang_vel * 0.25 (body frame)
    [17:20] projected_gravity (body frame, unit gravity)
    [20:28] last_action (8 dims)
- Commands (4 dims): [vx*2.0, vy*0.0, wz*0.25, pos_z=0.0]

Action structure (8 dims):
  [0:2] hip targets (left_hip, right_hip) * 1.0
  [2:6] shoulder+leg targets (left_shoulder, right_shoulder, left_leg, right_leg) * 1.0
  [6:8] wheel velocity targets (left_wheel, right_wheel) * 20.0

Usage:
    python scripts/transfer_flamingo_sim2sim.py \
        --policy logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/model_4999.pt \
        --xml ../sim2sim_onnx/assets/flamingo_torque.xml \
        --cmd_vx 0.5 --pd_scale 1.0
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
from scipy.spatial.transform import Rotation as R


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (matching Flamingo_Flat_Stand_Drive training config)
# ═══════════════════════════════════════════════════════════════════════════════

# Dimensions
SINGLE_OBS_DIM = 28
STACK_FRAMES = 3       # num_policy_stacks=2 (CLI default) + 1
NUM_COMMANDS = 4       # [vx*2.0, vy*0.0, wz*0.25, pos_z]
NUM_OBS = SINGLE_OBS_DIM * STACK_FRAMES + NUM_COMMANDS  # 88
NUM_ACTIONS = 8

# Action scaling (from training config ActionsCfg)
ACTION_SCALE_JOINTS = 1.0
ACTION_SCALE_WHEELS = 20.0

# Observation scaling (from training config ObservationsCfg)
OBS_JOINT_VEL_SCALE = 0.15
OBS_ANG_VEL_SCALE = 0.25

# Gear ratio applied to leg observations (joint_pos_leg_gear, joint_vel_leg_gear)
OBS_LEG_GEAR_RATIO = -1.5

# Command scaling (from generated_scaled_commands, scale=(2.0, 0.0, 0.25))
CMD_SCALE = np.array([2.0, 0.0, 0.25, 1.0], dtype=np.float32)

# PD controller gains (from Isaac Lab robot config flamingo_rev03_1_1.py)
PD_GAINS = {
    'hip':      {'kp': 100.0, 'kd': 1.5},
    'shoulder': {'kp': 100.0, 'kd': 1.5},
    'leg':      {'kp': 120.0, 'kd': 1.5},
    'wheel':    {'kp': 0.0,   'kd': 0.7},
}

# Gear ratio for leg PD torque computation (GearDelayedPDActuator)
LEG_GEAR_RATIO = -1.5
LEG_GAMMA = 1.0

# Torque limits (from Isaac Lab robot config)
JOINT_TORQUE_LIMIT = 60.0   # hip, shoulder, leg effort_limit
WHEEL_TORQUE_LIMIT = 36.0   # wheel effort_limit

# MuJoCo joint indices (from XML body hierarchy)
# qpos: [0:3]=pos, [3:7]=quat(wxyz), [7:]=joints
# qvel: [0:3]=lin_vel, [3:6]=ang_vel, [6:]=joint_vel
JOINT_QPOS = {
    'left_hip': 7,  'left_shoulder': 8,  'left_leg': 9,  'left_wheel': 10,
    'right_hip': 11, 'right_shoulder': 12, 'right_leg': 13, 'right_wheel': 14,
}
JOINT_QVEL = {
    'left_hip': 6,  'left_shoulder': 7,  'left_leg': 8,  'left_wheel': 9,
    'right_hip': 10, 'right_shoulder': 11, 'right_leg': 12, 'right_wheel': 13,
}
# Actuator order in ctrl array (matches XML <actuator> order)
ACT = {
    'left_hip': 0, 'right_hip': 1,
    'left_shoulder': 2, 'right_shoulder': 3,
    'left_leg': 4, 'right_leg': 5,
    'left_wheel': 6, 'right_wheel': 7,
}


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, num_obs=88, num_actions=8,
                 actor_hidden_dims=[512, 256, 128], activation="elu"):
        super().__init__()
        act_fn = {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        layers = []
        prev = num_obs
        for h in actor_hidden_dims:
            layers.extend([nn.Linear(prev, h), act_fn])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, x):
        return self.actor(x)


def load_policy(checkpoint_path, num_obs=88, num_actions=8):
    print(f"Loading policy from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = ActorCritic(num_obs=num_obs, num_actions=num_actions)
    state = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
    policy.load_state_dict(state, strict=False)
    policy.eval()

    # Verify dimensions
    first_weight = state.get("actor.0.weight")
    if first_weight is not None:
        actual_in = first_weight.shape[1]
        if actual_in != num_obs:
            raise ValueError(f"Checkpoint expects {actual_in} obs but got {num_obs}")
    print(f"  Actor: {num_obs} -> [512,256,128] -> {num_actions}")
    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def quat_to_euler(quat_wxyz):
    """Quaternion (w,x,y,z) -> euler (roll, pitch, yaw)."""
    x, y, z, w = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def compute_projected_gravity(quat_wxyz):
    """Compute projected gravity vector in body frame (unit gravity).

    Isaac Lab: quat_rotate_inverse(root_quat_w, [0, 0, -1])
    For upright robot: returns [0, 0, -1]
    """
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r = R.from_quat(quat_xyzw)
    gravity_world = np.array([0.0, 0.0, -1.0])
    return r.apply(gravity_world, inverse=True).astype(np.float32)


def get_body_angular_velocity(data):
    """Get angular velocity in body frame.

    MuJoCo free joint convention:
      qvel[0:3] = linear velocity in WORLD frame (needs rotation to body)
      qvel[3:6] = angular velocity in BODY frame (use directly!)

    IMPORTANT: Use qvel[3:6] directly, NOT the gyro sensor.
    The gyro sensor has noise=0.2 but Isaac Lab training had NO sensor noise.
    """
    # qvel[3:6] is ALREADY in body frame for MuJoCo free joints - NO noise
    return data.qvel[3:6].astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVATION
# ═══════════════════════════════════════════════════════════════════════════════

def construct_observation(data, last_action, negate_wheels=True):
    """Build single-frame observation (28 dims).

    Matches Flamingo_Flat_Stand_Drive StackPolicyCfg:
      hip_shoulder_pos(4) + leg_pos*gear(2) + hip_shoulder_vel*0.15(4) +
      leg_vel*gear*0.15(2) + wheel_vel*0.15(2) + ang_vel*0.25(3) +
      projected_gravity(3) + last_action(8)
    """
    obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
    i = 0

    # [0:4] Hip + shoulder joint positions (raw)
    obs[i:i+4] = data.qpos[[
        JOINT_QPOS['left_hip'], JOINT_QPOS['right_hip'],
        JOINT_QPOS['left_shoulder'], JOINT_QPOS['right_shoulder'],
    ]]
    i += 4

    # [4:6] Leg joint positions * gear_ratio (-1.5)
    obs[i:i+2] = data.qpos[[
        JOINT_QPOS['left_leg'], JOINT_QPOS['right_leg'],
    ]] * OBS_LEG_GEAR_RATIO
    i += 2

    # [6:10] Hip + shoulder joint velocities * 0.15
    obs[i:i+4] = data.qvel[[
        JOINT_QVEL['left_hip'], JOINT_QVEL['right_hip'],
        JOINT_QVEL['left_shoulder'], JOINT_QVEL['right_shoulder'],
    ]] * OBS_JOINT_VEL_SCALE
    i += 4

    # [10:12] Leg joint velocities * gear_ratio * 0.15
    obs[i:i+2] = data.qvel[[
        JOINT_QVEL['left_leg'], JOINT_QVEL['right_leg'],
    ]] * OBS_LEG_GEAR_RATIO * OBS_JOINT_VEL_SCALE
    i += 2

    # [12:14] Wheel joint velocities * 0.15
    # Many Isaac Lab models use positive = forward. If negate_wheels is True,
    # we assume MJ needs inversion to match IL.
    wheel_vels = data.qvel[[JOINT_QVEL['left_wheel'], JOINT_QVEL['right_wheel']]]
    if negate_wheels:
        obs[i:i+2] = -wheel_vels * OBS_JOINT_VEL_SCALE
    else:
        obs[i:i+2] = wheel_vels * OBS_JOINT_VEL_SCALE
    i += 2

    # [14:17] Base angular velocity (body frame) * 0.25
    omega = get_body_angular_velocity(data)
    obs[i:i+3] = omega * OBS_ANG_VEL_SCALE
    i += 3

    # [17:20] Projected gravity (body frame)
    obs[i:i+3] = compute_projected_gravity(data.qpos[3:7])
    i += 3

    # [20:28] Last action (8 dims)
    obs[i:i+8] = last_action
    i += 8

    return obs


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_actions(model, data, actions, pd_scale=1.0, negate_wheels=True):
    """Apply policy actions via PD control.

    Actions [0:2] = hip position targets * 1.0
    Actions [2:6] = shoulder+leg position targets * 1.0
    Actions [6:8] = wheel velocity targets * 20.0

    Hip/shoulder: standard PD
    Legs: gear-transformed PD (GearDelayedPDActuator, gear_ratio=-1.5)
    Wheels: velocity PD (Kd only)

    pd_scale multiplies all PD gains to compensate for PhysX vs MuJoCo dynamics gap.
    """
    joint_targets = actions[0:6] * ACTION_SCALE_JOINTS
    wheel_targets = actions[6:8] * ACTION_SCALE_WHEELS

    torques = np.zeros(model.nu, dtype=np.float32)

    # Hip and shoulder: standard PD (no gear ratio)
    for idx, (name, jtype) in enumerate([
        ('left_hip', 'hip'), ('right_hip', 'hip'),
        ('left_shoulder', 'shoulder'), ('right_shoulder', 'shoulder'),
    ]):
        kp = PD_GAINS[jtype]['kp'] * pd_scale
        kd = PD_GAINS[jtype]['kd'] * pd_scale
        pos = data.qpos[JOINT_QPOS[name]]
        vel = data.qvel[JOINT_QVEL[name]]
        t = kp * (joint_targets[idx] - pos) + kd * (0.0 - vel)
        torques[ACT[name]] = np.clip(t, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)

    # Leg joints: gear-transformed PD (GearDelayedPDActuator, gear_ratio=-1.5)
    kp_leg = PD_GAINS['leg']['kp'] * pd_scale
    kd_leg = PD_GAINS['leg']['kd'] * pd_scale
    g = LEG_GEAR_RATIO
    gamma = LEG_GAMMA
    for idx, name in enumerate(['left_leg', 'right_leg']):
        pos = data.qpos[JOINT_QPOS[name]]
        vel = data.qvel[JOINT_QVEL[name]]
        pos_motor = pos * g
        vel_motor = vel * g
        t_motor = kp_leg * (joint_targets[4 + idx] - pos_motor) + kd_leg * (0.0 - vel_motor)
        t = t_motor * g * gamma
        torques[ACT[name]] = np.clip(t, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)

    # Velocity-controlled wheels: torque = Kd*(target_vel - vel)
    kd_w = PD_GAINS['wheel']['kd'] * pd_scale
    for idx, name in enumerate(['left_wheel', 'right_wheel']):
        vel = data.qvel[JOINT_QVEL[name]]
        target = wheel_targets[idx]
        if negate_wheels:
            target = -target
        t = kd_w * (target - vel)
        torques[ACT[name]] = np.clip(t, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)

    data.ctrl[:] = torques


def override_model_params(model):
    """Override MuJoCo model parameters to match Isaac Lab training config.

    CRITICAL: Disable ALL MuJoCo-side force limiting. We handle our own torque
    clipping in apply_actions(). The XML has restrictive limits (joints ±23 Nm,
    wheels ±5 Nm) but Isaac Lab training uses ±60 Nm / ±36 Nm.
    """
    print("Overriding MuJoCo model parameters:")

    # 1. Disable ctrl limiting on all actuators (we clip externally)
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        old_range = model.actuator_ctrlrange[i].copy()
        model.actuator_ctrllimited[i] = 0  # Disable ctrl clamping
        model.actuator_forcelimited[i] = 0  # Disable force clamping
        # Also set wide ranges as fallback
        model.actuator_ctrlrange[i] = [-200.0, 200.0]
        model.actuator_forcerange[i] = [-200.0, 200.0]
        print(f"    Actuator '{act_name}': ctrlrange {old_range} -> DISABLED "
              f"(ctrllimited=0, forcelimited=0)")

    # 2. Disable joint-level actuator force limiting
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        old_range = model.jnt_actfrcrange[j].copy()
        old_limited = model.jnt_actfrclimited[j]
        model.jnt_actfrclimited[j] = 0  # Disable actuator force limit on joint
        model.jnt_actfrcrange[j] = [-200.0, 200.0]  # Wide fallback
        if old_limited:
            print(f"    Joint '{jnt_name}': actfrcrange {old_range} -> DISABLED "
                  f"(actfrclimited=0)")

    # 3. Zero passive joint damping (Isaac Lab has no passive damping)
    # Isaac Lab PD damping is handled explicitly in our PD controller
    for i in range(model.nv):
        if i < 6:  # Free joint DOFs
            continue
        old_damp = model.dof_damping[i]
        model.dof_damping[i] = 0.0
        if old_damp != 0.0:
            dof_jnt = model.dof_jntid[i]
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, dof_jnt)
            print(f"    Joint '{jnt_name}' DOF {i}: passive damping {old_damp} -> 0.0")

    # 4. Disable sensor noise (Isaac Lab training has no sensor noise)
    for i in range(model.nsensor):
        old_noise = model.sensor_noise[i]
        if old_noise > 0:
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            model.sensor_noise[i] = 0.0
            print(f"    Sensor '{sensor_name}': noise {old_noise} -> 0.0")

    print("  All MuJoCo-side force limits DISABLED (we clip externally)")
    print("  All passive damping zeroed")
    print("  All sensor noise zeroed")


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINATION
# ═══════════════════════════════════════════════════════════════════════════════

TERM_MAX_PITCH = np.radians(60.0)
TERM_MAX_ROLL = np.radians(60.0)
TERM_MIN_HEIGHT = 0.10
TERM_MAX_HEIGHT = 1.0


def check_termination(data):
    euler = quat_to_euler(data.qpos[3:7])
    roll, pitch = euler[0], euler[1]
    height = data.qpos[2]

    if abs(pitch) > TERM_MAX_PITCH:
        return True, f"excessive pitch ({np.degrees(pitch):.1f}deg)"
    if abs(roll) > TERM_MAX_ROLL:
        return True, f"excessive roll ({np.degrees(roll):.1f}deg)"
    if height < TERM_MIN_HEIGHT:
        return True, f"height too low ({height:.3f}m)"
    if height > TERM_MAX_HEIGHT:
        return True, f"height too high ({height:.3f}m)"

    return False, None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Sim2Sim Flamingo transfer to MuJoCo")
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--cmd_vx", type=float, default=0.0)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_wz", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--init_height", type=float, default=0.30,
                        help="Init height (Isaac Lab: 0.5562m, MuJoCo max: 0.30m - GEOMETRY MISMATCH!)")
    parser.add_argument("--pd_scale", type=float, default=1.0,
                        help="Scale PD gains by this factor (try 2-6 for PhysX->MuJoCo gap)")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed diagnostics for first 10 steps")
    parser.add_argument("--no_negate", action="store_true",
                        help="Disable inversion of wheel axes (try if robot falls instantly)")
    args = parser.parse_args()

    negate_wheels = not args.no_negate

    for path, label in [(args.xml, "XML"), (args.policy, "Policy")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    # Load MuJoCo model
    print(f"Loading MuJoCo model: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Override model parameters to match Isaac Lab training
    override_model_params(model)

    # Load policy
    policy = load_policy(args.policy)

    # Simulation settings
    sim_dt = model.opt.timestep  # Use XML timestep (0.005s)
    ctrl_dt = sim_dt * args.decimation

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[2] = args.init_height
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # Initialize observation history with FIRST REAL observation
    initial_obs = construct_observation(data, last_action, negate_wheels=negate_wheels)
    obs_history = collections.deque(maxlen=STACK_FRAMES)
    for _ in range(STACK_FRAMES):
        obs_history.append(initial_obs.copy())

    # Commands: [vx, vy, wz, pos_z] scaled by CMD_SCALE
    # pos_z = 0 (flat env sets pos_z range to (0,0))
    raw_commands = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz, 0.0], dtype=np.float32)
    commands = raw_commands * CMD_SCALE

    print(f"\nConfiguration:")
    print(f"  Commands: vx={args.cmd_vx:.2f}, vy={args.cmd_vy:.2f}, wz={args.cmd_wz:.2f}")
    print(f"  Scaled:   [{commands[0]:.2f}, {commands[1]:.2f}, {commands[2]:.2f}, {commands[3]:.2f}]")
    print(f"  Sim dt={sim_dt}s, Ctrl dt={ctrl_dt}s (dec={args.decimation})")
    print(f"  Obs: {SINGLE_OBS_DIM}x{STACK_FRAMES}+{NUM_COMMANDS}={NUM_OBS}")
    print(f"  PD gain scale: {args.pd_scale}x")
    print(f"  Init height: {args.init_height}m")
    if args.pd_scale != 1.0:
        print(f"  Scaled PD gains: hip kp={100*args.pd_scale:.0f} kd={1.5*args.pd_scale:.1f}, "
              f"shoulder kp={100*args.pd_scale:.0f} kd={1.5*args.pd_scale:.1f}, "
              f"leg kp={120*args.pd_scale:.0f} kd={1.5*args.pd_scale:.1f}, "
              f"wheel kd={0.7*args.pd_scale:.2f}")
    print(f"  PD control: recomputed every physics step (matching Isaac Lab)")
    if args.debug:
        print("  DEBUG MODE: printing detailed diagnostics for first 10 steps")

    # Verify initial state
    euler = quat_to_euler(data.qpos[3:7])
    print(f"\nInitial state after mj_forward:")
    print(f"  height={data.qpos[2]:.4f}m, roll={np.degrees(euler[0]):.2f}deg, "
          f"pitch={np.degrees(euler[1]):.2f}deg")
    print(f"  proj_gravity={compute_projected_gravity(data.qpos[3:7])}")
    print(f"  ang_vel(qvel)={data.qvel[3:6]}")
    print()
    print("Starting simulation...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        step = 0
        terminated = False

        while viewer.is_running() and (time.time() - t0) < args.duration:
            t_step = time.time()

            # 1. Construct observation from current state
            obs = construct_observation(data, last_action, negate_wheels=negate_wheels)
            obs_history.append(obs)

            # 2. Stack (newest first, matching CO-RL StateHandler) + commands
            stacked = np.concatenate(list(reversed(obs_history)))
            final_obs = np.concatenate([stacked, commands])

            # 3. Policy inference
            with torch.no_grad():
                tensor = torch.from_numpy(final_obs).unsqueeze(0).float()
                raw_action = policy(tensor).numpy().squeeze()

            action = np.clip(raw_action, -1.0, 1.0)
            # Use RAW action for last_action history, as in Isaac Lab ActionManager
            last_action = raw_action.copy()

            # === DEBUG: detailed diagnostics for first N steps ===
            if args.debug and step < 10:
                print(f"\n{'='*70}")
                print(f"STEP {step}")
                print(f"{'='*70}")

                # State
                h = data.qpos[2]
                euler = quat_to_euler(data.qpos[3:7])
                proj_g = compute_projected_gravity(data.qpos[3:7])
                print(f"  State: h={h:.4f}m  quat={data.qpos[3:7]}")
                print(f"         roll={np.degrees(euler[0]):.2f} pitch={np.degrees(euler[1]):.2f} yaw={np.degrees(euler[2]):.2f}")
                print(f"         proj_gravity={proj_g}")
                print(f"         ang_vel(raw)={data.qvel[3:6]}")

                # Single-frame obs breakdown
                print(f"\n  Single-frame obs (28 dims):")
                print(f"    hip_sho_pos  [0:4]:   {obs[0:4]}")
                print(f"    leg_pos*gear [4:6]:   {obs[4:6]}")
                print(f"    hip_sho_vel  [6:10]:  {obs[6:10]}")
                print(f"    leg_vel*gear [10:12]: {obs[10:12]}")
                print(f"    wheel_vel    [12:14]: {obs[12:14]}")
                print(f"    ang_vel      [14:17]: {obs[14:17]}")
                print(f"    proj_grav    [17:20]: {obs[17:20]}")
                print(f"    last_act     [20:28]: {obs[20:28]}")

                # Stacked obs summary
                print(f"\n  Stacked obs ({len(final_obs)} dims):")
                for frame_i in range(STACK_FRAMES):
                    s = frame_i * SINGLE_OBS_DIM
                    frame = final_obs[s:s+SINGLE_OBS_DIM]
                    nonzero = np.count_nonzero(frame)
                    print(f"    Frame {frame_i} [{s}:{s+SINGLE_OBS_DIM}]: "
                          f"norm={np.linalg.norm(frame):.4f} nonzero={nonzero}/28")
                cmd_start = SINGLE_OBS_DIM * STACK_FRAMES
                print(f"    Commands [{cmd_start}:{cmd_start+NUM_COMMANDS}]: {final_obs[cmd_start:]}")

                # Policy output
                print(f"\n  Raw policy output: {raw_action}")
                print(f"  Clipped action:    {action}")
                print(f"  Hip targets     (x{ACTION_SCALE_JOINTS}): {action[0:2] * ACTION_SCALE_JOINTS}")
                print(f"  Sho+Leg targets (x{ACTION_SCALE_JOINTS}): {action[2:6] * ACTION_SCALE_JOINTS}")
                print(f"  Wheel targets   (x{ACTION_SCALE_WHEELS}): {action[6:8] * ACTION_SCALE_WHEELS}")

                # Preview torques (with pd_scale)
                print(f"\n  PD torques (pd_scale={args.pd_scale}x):")
                for idx, (name, jtype) in enumerate([
                    ('left_hip', 'hip'), ('right_hip', 'hip'),
                    ('left_shoulder', 'shoulder'), ('right_shoulder', 'shoulder'),
                ]):
                    kp = PD_GAINS[jtype]['kp'] * args.pd_scale
                    kd = PD_GAINS[jtype]['kd'] * args.pd_scale
                    pos = data.qpos[JOINT_QPOS[name]]
                    vel = data.qvel[JOINT_QVEL[name]]
                    target = action[idx] * ACTION_SCALE_JOINTS
                    t = kp * (target - pos) + kd * (0.0 - vel)
                    t_clipped = np.clip(t, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
                    print(f"    {name:>16s}: target={target:+.4f} pos={pos:+.4f} vel={vel:+.4f} -> t={t:+.1f} clip={t_clipped:+.1f}")

                gg = LEG_GEAR_RATIO
                for idx, name in enumerate(['left_leg', 'right_leg']):
                    pos = data.qpos[JOINT_QPOS[name]]
                    vel = data.qvel[JOINT_QVEL[name]]
                    target = action[4 + idx] * ACTION_SCALE_JOINTS
                    pos_motor = pos * gg
                    vel_motor = vel * gg
                    t_motor = (PD_GAINS['leg']['kp'] * args.pd_scale) * (target - pos_motor) + \
                              (PD_GAINS['leg']['kd'] * args.pd_scale) * (0.0 - vel_motor)
                    t = t_motor * gg * LEG_GAMMA
                    t_clipped = np.clip(t, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
                    print(f"    {name:>16s}: target={target:+.4f} pos={pos:+.4f}({pos_motor:+.4f}m) vel={vel:+.4f} -> gear_t={t:+.1f} clip={t_clipped:+.1f}")

                for idx, name in enumerate(['left_wheel', 'right_wheel']):
                    vel = data.qvel[JOINT_QVEL[name]]
                    target_vel = action[6 + idx] * ACTION_SCALE_WHEELS
                    kd_w = PD_GAINS['wheel']['kd'] * args.pd_scale
                    t = kd_w * (target_vel - vel)
                    t_clipped = np.clip(t, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
                    print(f"    {name:>16s}: tgt_vel={target_vel:+.2f} vel={vel:+.4f} -> t={t:+.1f} clip={t_clipped:+.1f}")

            # 4. Apply PD + step physics (PD recomputed EVERY physics step)
            for _ in range(args.decimation):
                apply_actions(model, data, action, pd_scale=args.pd_scale, negate_wheels=negate_wheels)
                mujoco.mj_step(model, data)

            # Verify actuator forces match what we set (first few steps)
            if args.debug and step < 3:
                print(f"\n  Actuator verification (after last substep):")
                print(f"    ctrl set:       {data.ctrl}")
                print(f"    actuator_force: {data.actuator_force}")
                diff = np.abs(data.ctrl - data.actuator_force)
                if np.max(diff) > 0.01:
                    print(f"    WARNING: ctrl != actuator_force! Max diff={np.max(diff):.4f}")
                    print(f"    MuJoCo may still be limiting forces!")
                else:
                    print(f"    OK: forces match (max diff={np.max(diff):.6f})")

            step += 1

            # 5. Check termination
            term, reason = check_termination(data)
            if term and not terminated:
                terminated = True
                print(f"\n*** TERMINATED at t={time.time()-t0:.2f}s (step {step}): {reason} ***\n")

            # Status every second
            if step % 50 == 0:
                h = data.qpos[2]
                vx = data.qvel[0]
                euler = quat_to_euler(data.qpos[3:7])
                leg_pos = [data.qpos[JOINT_QPOS['left_leg']], data.qpos[JOINT_QPOS['right_leg']]]
                max_ctrl = np.max(np.abs(data.ctrl))
                status = " [TERMINATED]" if terminated else ""
                print(f"t={time.time()-t0:.1f}s | h={h:.3f}m | vx={vx:.2f}m/s | "
                      f"roll={np.degrees(euler[0]):.1f} pitch={np.degrees(euler[1]):.1f} | "
                      f"legs=[{leg_pos[0]:.2f},{leg_pos[1]:.2f}] max_ctrl={max_ctrl:.1f}Nm"
                      f"{status}")

            viewer.sync()
            elapsed = time.time() - t_step
            if elapsed < ctrl_dt:
                time.sleep(ctrl_dt - elapsed)

    if terminated:
        print(f"Simulation ended (terminated at step {step}).")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
