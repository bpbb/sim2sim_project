#!/usr/bin/env python3
"""Debug script: test PD control WITHOUT any policy.

Holds the robot at target joint positions using PD control.
This isolates the MuJoCo model + PD from the policy/observation pipeline.

Tests:
  1. Can the robot stand at target height with default (zero) joint targets?
  2. What torques are needed? Are they within limits?
  3. Does passive damping help or hurt?
  4. Does the gear ratio for legs matter?

Usage:
    python scripts/debug_pd_control.py --xml sim2sim_onnx/assets/flamingo_torque.xml

    # Test with gear ratio for legs:
    python scripts/debug_pd_control.py --xml ... --gear_ratio

    # Test keeping XML passive damping (don't zero it out):
    python scripts/debug_pd_control.py --xml ... --keep_damping
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

PD_GAINS = {
    'hip':      {'kp': 100.0, 'kd': 1.5},
    'shoulder': {'kp': 100.0, 'kd': 1.5},
    'leg':      {'kp': 120.0, 'kd': 1.5},
    'wheel':    {'kp': 0.0,   'kd': 0.7},
}

LEG_GEAR_RATIO = -1.5
LEG_GAMMA = 1.0

JOINT_TORQUE_LIMIT = 60.0
WHEEL_TORQUE_LIMIT = 36.0

# MuJoCo indices
JOINT_QPOS = {
    'left_hip': 7, 'left_shoulder': 8, 'left_leg': 9, 'left_wheel': 10,
    'right_hip': 11, 'right_shoulder': 12, 'right_leg': 13, 'right_wheel': 14,
}
JOINT_QVEL = {
    'left_hip': 6, 'left_shoulder': 7, 'left_leg': 8, 'left_wheel': 9,
    'right_hip': 10, 'right_shoulder': 11, 'right_leg': 12, 'right_wheel': 13,
}
ACT = {
    'left_hip': 0, 'right_hip': 1,
    'left_shoulder': 2, 'right_shoulder': 3,
    'left_leg': 4, 'right_leg': 5,
    'left_wheel': 6, 'right_wheel': 7,
}

JOINT_ORDER = [
    ('left_hip', 'hip'), ('right_hip', 'hip'),
    ('left_shoulder', 'shoulder'), ('right_shoulder', 'shoulder'),
    ('left_leg', 'leg'), ('right_leg', 'leg'),
]


def apply_pd(model, data, joint_targets, use_gear_ratio=False):
    """Apply PD control to hold target joint positions.

    Args:
        joint_targets: dict mapping joint name -> target position (rad)
        use_gear_ratio: if True, apply gear ratio PD for legs
    """
    torques = np.zeros(model.nu, dtype=np.float32)

    for name, jtype in JOINT_ORDER:
        target = joint_targets.get(name, 0.0)
        kp = PD_GAINS[jtype]['kp']
        kd = PD_GAINS[jtype]['kd']
        pos = data.qpos[JOINT_QPOS[name]]
        vel = data.qvel[JOINT_QVEL[name]]

        if jtype == 'leg' and use_gear_ratio:
            g = LEG_GEAR_RATIO
            gamma = LEG_GAMMA
            pos_motor = pos * g
            vel_motor = vel * g
            t_motor = kp * (target - pos_motor) + kd * (0.0 - vel_motor)
            t = t_motor * g * gamma
        else:
            t = kp * (target - pos) + kd * (0.0 - vel)

        torques[ACT[name]] = np.clip(t, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)

    # Wheels: hold velocity at 0
    for name in ['left_wheel', 'right_wheel']:
        vel = data.qvel[JOINT_QVEL[name]]
        t = PD_GAINS['wheel']['kd'] * (0.0 - vel)
        torques[ACT[name]] = np.clip(t, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)

    data.ctrl[:] = torques
    return torques


def override_model_params(model, zero_damping=True):
    """Override MuJoCo model params."""
    # Override actuator limits
    for i in range(model.nu):
        if i < 6:
            model.actuator_ctrlrange[i] = [-JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT]
            model.actuator_forcerange[i] = [-JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT]
        else:
            model.actuator_ctrlrange[i] = [-WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT]
            model.actuator_forcerange[i] = [-WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT]

    # Override joint actuatorfrcrange
    wheel_joint_ids = set()
    for j in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jnt_name and 'wheel' in jnt_name:
            wheel_joint_ids.add(j)

    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if j in wheel_joint_ids:
            model.jnt_actfrcrange[j] = [-WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT]
        else:
            model.jnt_actfrcrange[j] = [-JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT]

    # Damping
    if zero_damping:
        for i in range(model.nv):
            if i < 6:
                continue
            model.dof_damping[i] = 0.0
        print("  Passive damping: ZEROED")
    else:
        print("  Passive damping: KEPT (XML defaults)")

    print(f"  Force limits: joints=±{JOINT_TORQUE_LIMIT}, wheels=±{WHEEL_TORQUE_LIMIT}")


def quat_to_euler(quat_wxyz):
    x, y, z, w = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def main():
    parser = argparse.ArgumentParser(description="Debug PD control (no policy)")
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--init_height", type=float, default=0.35)
    parser.add_argument("--gear_ratio", action="store_true",
                        help="Use gear ratio PD for legs")
    parser.add_argument("--keep_damping", action="store_true",
                        help="Keep XML passive damping (don't zero it)")
    parser.add_argument("--decimation", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.xml):
        raise FileNotFoundError(f"XML not found: {args.xml}")

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    print(f"=== DEBUG PD CONTROL (no policy) ===")
    print(f"  Gear ratio: {'YES (g={LEG_GEAR_RATIO})' if args.gear_ratio else 'NO (simple PD)'}")
    override_model_params(model, zero_damping=not args.keep_damping)

    sim_dt = 0.005
    model.opt.timestep = sim_dt
    ctrl_dt = sim_dt * args.decimation

    # Reset to standing
    mujoco.mj_resetData(model, data)
    data.qpos[2] = args.init_height
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # Target: all joints at 0 (default standing pose)
    joint_targets = {name: 0.0 for name, _ in JOINT_ORDER}

    print(f"  Init height: {args.init_height}m")
    print(f"  Joint targets: all zeros (default pose)")
    print(f"  Sim dt={sim_dt}s, Ctrl dt={ctrl_dt}s (dec={args.decimation})")
    print(f"  PD recomputed every physics step")
    print(f"\nStarting...\n")

    # Print header
    print(f"{'time':>6s} | {'height':>7s} | {'roll':>6s} {'pitch':>6s} | "
          f"{'L_hip':>6s} {'R_hip':>6s} {'L_sho':>6s} {'R_sho':>6s} {'L_leg':>6s} {'R_leg':>6s} | "
          f"{'t_Lhip':>7s} {'t_Rhip':>7s} {'t_Lsho':>7s} {'t_Rsho':>7s} {'t_Lleg':>7s} {'t_Rleg':>7s}")
    print("-" * 140)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        step = 0

        while viewer.is_running() and (time.time() - t0) < args.duration:
            t_step = time.time()

            # Apply PD + step physics
            for _ in range(args.decimation):
                torques = apply_pd(model, data, joint_targets,
                                   use_gear_ratio=args.gear_ratio)
                mujoco.mj_step(model, data)

            step += 1

            # Print every 0.5s
            if step % 25 == 0:
                h = data.qpos[2]
                euler = quat_to_euler(data.qpos[3:7])

                positions = []
                for name, _ in JOINT_ORDER:
                    positions.append(data.qpos[JOINT_QPOS[name]])

                torque_vals = []
                for name, _ in JOINT_ORDER:
                    torque_vals.append(torques[ACT[name]])

                t_elapsed = time.time() - t0
                print(f"{t_elapsed:5.1f}s | {h:6.3f}m | "
                      f"{np.degrees(euler[0]):5.1f}° {np.degrees(euler[1]):5.1f}° | "
                      + " ".join(f"{p:6.3f}" for p in positions) + " | "
                      + " ".join(f"{t:7.1f}" for t in torque_vals))

            viewer.sync()
            elapsed = time.time() - t_step
            if elapsed < ctrl_dt:
                time.sleep(ctrl_dt - elapsed)

    print("\nDone.")


if __name__ == "__main__":
    main()
