#!/usr/bin/env python3
import argparse
import os
import collections
import mujoco
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R

# Dimensions
SINGLE_OBS_DIM = 28
STACK_FRAMES = 3
NUM_COMMANDS = 4
NUM_OBS = 88
NUM_ACTIONS = 8
ACTION_SCALE_JOINTS = 1.0
ACTION_SCALE_WHEELS = 20.0
OBS_JOINT_VEL_SCALE = 0.15
OBS_ANG_VEL_SCALE = 0.25
OBS_LEG_GEAR_RATIO = -1.5
CMD_SCALE = np.array([2.0, 0.0, 0.25, 1.0], dtype=np.float32)
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
JOINT_QPOS = {
    'left_hip': 7,  'left_shoulder': 8,  'left_leg': 9,  'left_wheel': 10,
    'right_hip': 11, 'right_shoulder': 12, 'right_leg': 13, 'right_wheel': 14,
}
JOINT_QVEL = {
    'left_hip': 6,  'left_shoulder': 7,  'left_leg': 8,  'left_wheel': 9,
    'right_hip': 10, 'right_shoulder': 11, 'right_leg': 12, 'right_wheel': 13,
}
ACT = {
    'left_hip': 0, 'right_hip': 1,
    'left_shoulder': 2, 'right_shoulder': 3,
    'left_leg': 4, 'right_leg': 5,
    'left_wheel': 6, 'right_wheel': 7,
}

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
    def forward(self, x): return self.actor(x)

def load_policy(checkpoint_path, num_obs=88, num_actions=8):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = ActorCritic(num_obs=num_obs, num_actions=num_actions)
    state = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy

def quat_to_euler(quat_wxyz):
    x, y, z, w = quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)

def compute_projected_gravity(quat_wxyz):
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r = R.from_quat(quat_xyzw)
    gravity_world = np.array([0.0, 0.0, -1.0])
    return r.apply(gravity_world, inverse=True).astype(np.float32)

def construct_observation(data, last_action):
    obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
    obs[0:4] = data.qpos[[JOINT_QPOS['left_hip'], JOINT_QPOS['right_hip'], JOINT_QPOS['left_shoulder'], JOINT_QPOS['right_shoulder']]]
    obs[4:6] = data.qpos[[JOINT_QPOS['left_leg'], JOINT_QPOS['right_leg']]] * OBS_LEG_GEAR_RATIO
    obs[6:10] = data.qvel[[JOINT_QVEL['left_hip'], JOINT_QVEL['right_hip'], JOINT_QVEL['left_shoulder'], JOINT_QVEL['right_shoulder']]] * OBS_JOINT_VEL_SCALE
    obs[10:12] = data.qvel[[JOINT_QVEL['left_leg'], JOINT_QVEL['right_leg']]] * OBS_LEG_GEAR_RATIO * OBS_JOINT_VEL_SCALE
    obs[12:14] = -data.qvel[[JOINT_QVEL['left_wheel'], JOINT_QVEL['right_wheel']]] * OBS_JOINT_VEL_SCALE
    obs[14:17] = data.qvel[3:6] * OBS_ANG_VEL_SCALE
    obs[17:20] = compute_projected_gravity(data.qpos[3:7])
    obs[20:28] = last_action
    return obs

def apply_actions(model, data, actions, pd_scale=1.0):
    joint_targets = actions[0:6] * ACTION_SCALE_JOINTS
    wheel_targets = actions[6:8] * ACTION_SCALE_WHEELS
    torques = np.zeros(model.nu, dtype=np.float32)
    for idx, (name, jtype) in enumerate([('left_hip', 'hip'), ('right_hip', 'hip'), ('left_shoulder', 'shoulder'), ('right_shoulder', 'shoulder')]):
        kp, kd = PD_GAINS[jtype]['kp'] * pd_scale, PD_GAINS[jtype]['kd'] * pd_scale
        t = kp * (joint_targets[idx] - data.qpos[JOINT_QPOS[name]]) + kd * (0.0 - data.qvel[JOINT_QVEL[name]])
        torques[ACT[name]] = np.clip(t, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    kp_leg, kd_leg = PD_GAINS['leg']['kp'] * pd_scale, PD_GAINS['leg']['kd'] * pd_scale
    g = LEG_GEAR_RATIO
    for idx, name in enumerate(['left_leg', 'right_leg']):
        t_motor = kp_leg * (joint_targets[4 + idx] - data.qpos[JOINT_QPOS[name]] * g) + kd_leg * (0.0 - data.qvel[JOINT_QVEL[name]] * g)
        torques[ACT[name]] = np.clip(t_motor * g, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    kd_w = PD_GAINS['wheel']['kd'] * pd_scale
    for idx, name in enumerate(['left_wheel', 'right_wheel']):
        torques[ACT[name]] = np.clip(kd_w * (-wheel_targets[idx] - data.qvel[JOINT_QVEL[name]]), -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
    data.ctrl[:] = torques

def override_model_params(model):
    for i in range(model.nu):
        model.actuator_ctrllimited[i] = 0
        model.actuator_forcelimited[i] = 0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE: model.jnt_actfrclimited[j] = 0
    for i in range(model.nv):
        if i >= 6: model.dof_damping[i] = 0.0
    for i in range(model.nsensor): model.sensor_noise[i] = 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True)
    parser.add_argument("--xml", type=str, required=True)
    parser.add_argument("--cmd_vx", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--init_height", type=float, default=0.5562)
    parser.add_argument("--pd_scale", type=float, default=1.0)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    override_model_params(model)
    policy = load_policy(args.policy)
    mujoco.mj_resetData(model, data)
    data.qpos[2] = args.init_height
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    initial_obs = construct_observation(data, last_action)
    obs_history = collections.deque([initial_obs.copy() for _ in range(STACK_FRAMES)], maxlen=STACK_FRAMES)
    commands = np.array([args.cmd_vx, 0.0, 0.0, 0.0], dtype=np.float32) * CMD_SCALE
    for step in range(int(args.duration / (model.opt.timestep * 4))):
        obs = construct_observation(data, last_action)
        obs_history.append(obs)
        stacked = np.concatenate(list(reversed(obs_history)))
        final_obs = np.concatenate([stacked, commands])
        with torch.no_grad(): action = np.clip(policy(torch.from_numpy(final_obs).unsqueeze(0).float()).numpy().squeeze(), -1.0, 1.0)
        last_action = action.copy()
        for _ in range(4):
            apply_actions(model, data, action, pd_scale=args.pd_scale)
            mujoco.mj_step(model, data)
        if step % 25 == 0:
            print(f"t={step*0.02:.2f}s | h={data.qpos[2]:.3f}m | vx={data.qvel[0]:.2f}m/s | pitch={np.degrees(quat_to_euler(data.qpos[3:7])[1]):.1f}")

if __name__ == "__main__":
    main()
