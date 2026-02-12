#!/usr/bin/env python3
"""Flamingo sim2sim transfer using position-velocity actuator model.

This version uses flamingo_pos_vel.xml which has built-in position and velocity
actuators, simplifying the control logic.

Usage:
    python scripts/transfer_flamingo_posvel.py --policy logs/.../model_4999.pt
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


# Constants
NUM_OBS = 88
SINGLE_OBS_DIM = 28
STACK_FRAMES = 3
NUM_ACTIONS = 8

# Scales from env.yaml
DOF_VEL_SCALE = 0.15
ANG_VEL_SCALE = 0.25
LEG_GEAR_RATIO = -1.5

# Action scales
ACTION_SCALES = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0], dtype=np.float32)

# Joint indices in MuJoCo
QPOS_IDX = {'l_hip': 7, 'l_sho': 8, 'l_leg': 9, 'l_whl': 10,
            'r_hip': 11, 'r_sho': 12, 'r_leg': 13, 'r_whl': 14}
QVEL_IDX = {'l_hip': 6, 'l_sho': 7, 'l_leg': 8, 'l_whl': 9,
            'r_hip': 10, 'r_sho': 11, 'r_leg': 12, 'r_whl': 13}


class ActorCritic(nn.Module):
    def __init__(self, num_obs=88, num_actions=8):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.actor(x)


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy = ActorCritic()
    state = {k: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("actor.")}
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy


def get_projected_gravity(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return np.array([
        -2.0 * (qx * qz - qw * qy),
        -2.0 * (qy * qz + qw * qx),
        -(1.0 - 2.0 * (qx * qx + qy * qy))
    ], dtype=np.float32)


def world_to_body(quat_wxyz, vec):
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat_wxyz)
    return rot.reshape(3, 3).T @ vec


def construct_obs(data, last_action):
    obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
    idx = 0

    # Joint positions (hip/shoulder: 4, leg: 2)
    # Note: right_shoulder and right_leg have inverted axes in MuJoCo
    obs[idx:idx+4] = [
        data.qpos[QPOS_IDX['l_hip']],
        data.qpos[QPOS_IDX['r_hip']],
        data.qpos[QPOS_IDX['l_sho']],
        -data.qpos[QPOS_IDX['r_sho']]  # Axis inverted
    ]
    idx += 4

    obs[idx:idx+2] = [
        data.qpos[QPOS_IDX['l_leg']] * LEG_GEAR_RATIO,
        -data.qpos[QPOS_IDX['r_leg']] * LEG_GEAR_RATIO  # Axis inverted
    ]
    idx += 2

    # Joint velocities (hip/shoulder: 4, leg: 2, wheel: 2)
    obs[idx:idx+4] = np.array([
        data.qvel[QVEL_IDX['l_hip']],
        data.qvel[QVEL_IDX['r_hip']],
        data.qvel[QVEL_IDX['l_sho']],
        -data.qvel[QVEL_IDX['r_sho']]  # Axis inverted
    ]) * DOF_VEL_SCALE
    idx += 4

    obs[idx:idx+2] = np.array([
        data.qvel[QVEL_IDX['l_leg']] * LEG_GEAR_RATIO,
        -data.qvel[QVEL_IDX['r_leg']] * LEG_GEAR_RATIO  # Axis inverted
    ]) * DOF_VEL_SCALE
    idx += 2

    obs[idx:idx+2] = np.array([
        data.qvel[QVEL_IDX['l_whl']],
        -data.qvel[QVEL_IDX['r_whl']]  # Axis inverted
    ]) * DOF_VEL_SCALE
    idx += 2

    # Angular velocity (body frame)
    quat = data.qpos[3:7]
    ang_vel_body = world_to_body(quat, data.qvel[3:6])
    obs[idx:idx+3] = ang_vel_body * ANG_VEL_SCALE
    idx += 3

    # Projected gravity
    obs[idx:idx+3] = get_projected_gravity(quat)
    idx += 3

    # Last action
    obs[idx:idx+8] = last_action
    idx += 8

    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--xml", type=str, default="assets/flamingo_pos_vel.xml")
    parser.add_argument("--cmd_vx", type=float, default=0.0)
    parser.add_argument("--cmd_wz", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    # Load model and policy
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    policy = load_policy(args.policy)

    model.opt.timestep = 0.005
    decimation = 4

    # Reset
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.35
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # Buffers
    obs_history = collections.deque(maxlen=STACK_FRAMES)
    for _ in range(STACK_FRAMES):
        obs_history.append(np.zeros(SINGLE_OBS_DIM, dtype=np.float32))

    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    command = np.array([args.cmd_vx * 2.0, 0.0, args.cmd_wz * 0.25, 0.0], dtype=np.float32)

    print(f"Command: vx={args.cmd_vx}, wz={args.cmd_wz}")
    print(f"Starting simulation...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        step = 0

        while viewer.is_running() and (time.time() - start) < args.duration:
            t0 = time.time()

            # Physics steps
            for _ in range(decimation):
                mujoco.mj_step(model, data)

            # Observation
            obs = construct_obs(data, last_action)
            obs_history.append(obs)
            stacked = np.concatenate(list(reversed(obs_history)))
            full_obs = np.concatenate([stacked, command])

            # Inference
            with torch.no_grad():
                action = policy(torch.tensor(full_obs).unsqueeze(0).float()).numpy().squeeze()
            action = np.clip(action, -1.0, 1.0)
            scaled = action * ACTION_SCALES

            # Apply to actuators (pos_vel.xml has position/velocity actuators)
            # Actuator order: l_hip, r_hip, l_sho, r_sho, l_leg, r_leg, l_whl, r_whl
            data.ctrl[0] = scaled[0]   # l_hip pos target
            data.ctrl[1] = scaled[1]   # r_hip pos target
            data.ctrl[2] = scaled[2]   # l_sho pos target
            data.ctrl[3] = -scaled[3]  # r_sho pos target (axis inverted)
            data.ctrl[4] = scaled[4]   # l_leg pos target
            data.ctrl[5] = -scaled[5]  # r_leg pos target (axis inverted)
            data.ctrl[6] = scaled[6]   # l_whl vel target
            data.ctrl[7] = -scaled[7]  # r_whl vel target (axis inverted)

            last_action = action.copy()
            step += 1

            if step % 50 == 0:
                h = data.qpos[2]
                vx = data.qvel[0]
                print(f"t={time.time()-start:.1f}s | h={h:.3f}m | vx={vx:.2f}m/s")

            viewer.sync()
            dt = time.time() - t0
            if dt < 0.005 * decimation:
                time.sleep(0.005 * decimation - dt)

    print("Done.")


if __name__ == "__main__":
    main()
