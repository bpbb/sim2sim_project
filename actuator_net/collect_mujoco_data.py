#!/usr/bin/env python3
"""Collect actuator data from MuJoCo for training Actuator Net.

Loads settings from configs/go2.yaml and runs the Go2 in MuJoCo with random
and structured motions.  Records MuJoCo's actual actuator forces
(qfrc_actuator) so the neural net learns the full actuator dynamics including
ctrlrange clipping, joint damping, armature, and friction.

Joint orders:
  qpos[7:] / qvel[6:] (joint order): FL_h FL_t FL_c  FR_h FR_t FR_c  RL_h RL_t RL_c  RR_h RR_t RR_c
  ctrl (actuator order):              FR_h FR_t FR_c  FL_h FL_t FL_c  RR_h RR_t RR_c  RL_h RL_t RL_c

Usage:
    python actuator_net/collect_mujoco_data.py
    python actuator_net/collect_mujoco_data.py --config actuator_net/configs/go2.yaml
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent

# Isaac Lab joint order mapping (same as deploy_mujoco.py)
MUJOCO_TO_ISAACLAB = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]

# MuJoCo joint names in qpos order
JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]


def apply_ctrl_swap(tau):
    """Convert joint-order torques to MuJoCo ctrl order (FR,FL,RR,RL)."""
    ctrl = np.zeros_like(tau)
    ctrl[0:3] = tau[3:6]    # FR
    ctrl[3:6] = tau[0:3]    # FL
    ctrl[6:9] = tau[9:12]   # RR
    ctrl[9:12] = tau[6:9]   # RL
    return ctrl


def collect_data(config):
    """Collect actuator data from MuJoCo simulation."""
    # ── Load MuJoCo model ─────────────────────────────────────────────────
    xml_path = str(PROJECT_ROOT / config["mujoco_model"])
    print(f"Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # ── Config values ─────────────────────────────────────────────────────
    kp = config["pd_gains"]["kp"]
    kd = config["pd_gains"]["kd"]
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    num_joints = config["num_joints"]
    ctrl_swap = config.get("ctrl_swap", True)

    dc = config["data_collection"]
    sim_dt = dc["simulation_dt"]
    num_episodes = dc["random_episodes"]
    steps_per_episode = dc["steps_per_episode"]
    num_sweeps = dc["joint_sweeps"]
    pos_range = dc["position_range"]
    sweep_range = dc["sweep_range"]

    model.opt.timestep = sim_dt

    print(f"PD gains: Kp={kp}, Kd={kd}")
    print(f"Default angles: {default_angles.tolist()}")
    print(f"Ctrl swap: {ctrl_swap}")
    print(f"Sim dt: {sim_dt}")

    samples = []

    # ── Helpers ────────────────────────────────────────────────────────────
    def reset_sim():
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0, 0, 0.35]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:] = default_angles
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

    def step_and_record(target_pos):
        """Apply PD torque, step MuJoCo, record actual actuator forces."""
        current_pos = data.qpos[7:].copy()
        current_vel = data.qvel[6:].copy()

        # PD torque in joint order
        tau = kp * (target_pos - current_pos) - kd * current_vel

        # Apply to ctrl (with swap for unitree_mujoco XML)
        if ctrl_swap:
            data.ctrl[:] = apply_ctrl_swap(tau)
        else:
            data.ctrl[:num_joints] = tau

        # Step simulation
        mujoco.mj_step(model, data)

        # Actual actuator force in joint order (skip 6 free-joint DOFs)
        actual_torque = data.qfrc_actuator[6:].copy()

        # Record per-joint samples with Isaac Lab joint index for one-hot
        for j in range(num_joints):
            samples.append((
                MUJOCO_TO_ISAACLAB[j],  # joint_idx in Isaac Lab order
                JOINT_NAMES[j],
                target_pos[j],
                current_pos[j],
                current_vel[j],
                actual_torque[j],
            ))

    # ── Phase 1: Random motion ────────────────────────────────────────────
    print(f"\nPhase 1: Random motion ({num_episodes} episodes x {steps_per_episode} steps)")
    for ep in range(num_episodes):
        reset_sim()
        offsets = np.random.uniform(pos_range[0], pos_range[1], num_joints)

        for step in range(steps_per_episode):
            if step % 50 == 0:
                offsets = np.random.uniform(pos_range[0], pos_range[1], num_joints)
            step_and_record(default_angles + offsets)

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{num_episodes}  samples: {len(samples)}")

    # ── Phase 2: Systematic joint sweeps ──────────────────────────────────
    print(f"\nPhase 2: Joint sweeps ({num_sweeps} per joint)")
    for j in range(num_joints):
        print(f"  Sweeping joint {j}: {JOINT_NAMES[j]}")
        for _ in range(num_sweeps):
            reset_sim()
            for val in np.linspace(sweep_range[0], sweep_range[1], 100):
                target = default_angles.copy()
                target[j] = val
                step_and_record(target)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nTotal samples: {len(samples)}")
    columns = ["joint_idx", "joint_name", "pos_desired", "pos_current", "velocity", "torque"]
    df = pd.DataFrame(samples, columns=columns)

    output_path = str(PROJECT_ROOT / "actuator_net" / config["paths"]["data"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"\nStatistics:\n{df.describe()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect MuJoCo actuator data")
    parser.add_argument(
        "--config", type=str, default="actuator_net/configs/go2.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(PROJECT_ROOT / config_path)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Config: {config_path}")
    print(f"Robot: {config['robot_name']}")
    collect_data(config)


if __name__ == "__main__":
    main()
