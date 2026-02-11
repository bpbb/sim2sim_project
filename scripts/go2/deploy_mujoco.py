#!/usr/bin/env python3
"""
Deploy Isaac Lab trained Go2 policy in MuJoCo.

Supports two torque modes:
  1. PD control (default) — uses MuJoCo's native PD with ctrl swap
  2. ActuatorNet — uses trained neural network for torque prediction

Joint order mapping (CORRECTED):
- MuJoCo:    [FL_h, FL_t, FL_c, FR_h, FR_t, FR_c, RL_h, RL_t, RL_c, RR_h, RR_t, RR_c]
- Isaac Lab: [FL_h, FR_h, RL_h, RR_h, FL_t, FR_t, RL_t, RR_t, FL_c, FR_c, RL_c, RR_c]

Usage:
    # PD control (default)
    python scripts/go2/deploy_mujoco.py scripts/go2/go2_isaaclab.yaml

    # With ActuatorNet
    python scripts/go2/deploy_mujoco.py scripts/go2/go2_isaaclab.yaml --actuator_net
"""

import time
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import os

# Joint order mapping (CORRECTED)
# MuJoCo index -> Isaac Lab index
MUJOCO_TO_ISAACLAB = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
# Isaac Lab index -> MuJoCo index
ISAACLAB_TO_MUJOCO = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]


def quat_rotate_inverse(q, v):
    """Rotate vector by inverse of quaternion (world to body frame)."""
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([q_x, q_y, q_z]), v)
    return v - q_w * t + np.cross(np.array([q_x, q_y, q_z]), t)


def get_gravity_orientation(quaternion):
    """Get projected gravity vector from quaternion."""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control for joint torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


class ActuatorNetWrapper:
    """Wrapper for using trained ActuatorNet in MuJoCo."""

    def __init__(self, model_path, scaler_path, num_joints=12):
        try:
            import joblib
            self.scaler = joblib.load(scaler_path)
        except ImportError:
            import pickle
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.num_joints = num_joints
        self.joint_onehot = np.eye(num_joints)
        print(f"[ActuatorNet] Loaded model: {model_path}")
        print(f"[ActuatorNet] Loaded scaler: {scaler_path}")

    def compute_torques(self, pos_desired, pos_current, velocity):
        """Predict torques: (pos_desired, pos_current, vel, joint_id) -> torque."""
        raw = np.stack([pos_desired, pos_current, velocity], axis=1)  # [12, 3]
        features = np.hstack([raw, self.joint_onehot])  # [12, 15]
        features_scaled = self.scaler.transform(features)
        with torch.no_grad():
            torques = self.model(torch.FloatTensor(features_scaled)).numpy()
        return torques.squeeze()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name or path")
    parser.add_argument("--actuator_net", action="store_true",
                        help="Use ActuatorNet for torque computation instead of PD")
    parser.add_argument("--actuator_net_path", type=str,
                        default=None, help="Path to actuator net model (default: actuator_net/models/go2/)")
    parser.add_argument("--policy", type=str, default=None,
                        help="Override policy path from yaml")
    args = parser.parse_args()

    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))

    # Handle config file path
    if os.path.isabs(args.config_file):
        config_path = args.config_file
    elif os.path.exists(args.config_file):
        config_path = args.config_file
    else:
        config_path = os.path.join(script_dir, args.config_file)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        # Handle path placeholders
        policy_path = args.policy or config["policy_path"]
        if "{SCRIPT_DIR}" in policy_path:
            policy_path = policy_path.replace("{SCRIPT_DIR}", script_dir)

        xml_path = config["xml_path"]
        if "{SCRIPT_DIR}" in xml_path:
            xml_path = xml_path.replace("{SCRIPT_DIR}", script_dir)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        lin_vel_scale = config.get("lin_vel_scale", 2.0)
        
        # Initial pose from config (optional)
        base_init_pos = config.get("base_init_pos", [0, 0, 0.45])
        base_init_quat = config.get("base_init_quat", [1, 0, 0, 0])

    # Reorder default angles to Isaac Lab order
    default_angles_isaaclab = default_angles[MUJOCO_TO_ISAACLAB]

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Set initial pose
    d.qpos[0:3] = base_init_pos
    d.qpos[3:7] = base_init_quat
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    # Load actuator net if requested
    actuator_net = None
    if args.actuator_net:
        if args.actuator_net_path:
            anet_model = args.actuator_net_path
            anet_scaler = args.actuator_net_path.replace("actuator_net_jit.pt", "feature_scaler.pkl")
        else:
            anet_model = os.path.join(project_root, "actuator_net/models/go2/actuator_net_jit.pt")
            anet_scaler = os.path.join(project_root, "actuator_net/models/go2/feature_scaler.pkl")
        actuator_net = ActuatorNetWrapper(anet_model, anet_scaler)

    torque_mode = "ActuatorNet" if actuator_net else "PD"
    print(f"Torque mode: {torque_mode}")
    print(f"Joint mapping: MuJoCo -> Isaac Lab: {MUJOCO_TO_ISAACLAB}")
    print(f"Joint mapping: Isaac Lab -> MuJoCo: {ISAACLAB_TO_MUJOCO}")
    print(f"Policy path: {policy_path}")
    print(f"XML path: {xml_path}")

    # Load policy
    policy = torch.jit.load(policy_path)
    policy.eval()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            if actuator_net:
                # ActuatorNet mode: predict torques directly
                tau = actuator_net.compute_torques(
                    target_dof_pos, d.qpos[7:], d.qvel[6:]
                )
                # Ctrl swap: actuator order (FR,FL,RR,RL) ≠ joint order (FL,FR,RL,RR)
                d.ctrl[0:3] = tau[3:6]   # FR
                d.ctrl[3:6] = tau[0:3]   # FL
                d.ctrl[6:9] = tau[9:12]  # RR
                d.ctrl[9:12] = tau[6:9]  # RL
            else:
                # PD control mode with ctrl swap
                tau = pd_control(target_dof_pos, d.qpos[7:], kps,
                                 np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[0:3] = tau[3:6]   # FR
                d.ctrl[3:6] = tau[0:3]   # FL
                d.ctrl[6:9] = tau[9:12]  # RR
                d.ctrl[9:12] = tau[6:9]  # RL

            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]  # MuJoCo order
                dqj = d.qvel[6:]  # MuJoCo order
                quat = d.qpos[3:7]

                # World to body frame
                world_lin_vel = d.qvel[0:3]
                world_ang_vel = d.qvel[3:6]
                base_lin_vel = quat_rotate_inverse(quat, world_lin_vel)
                base_ang_vel = quat_rotate_inverse(quat, world_ang_vel)

                gravity_orientation = get_gravity_orientation(quat)

                # Reorder joint data: MuJoCo -> Isaac Lab
                qj_isaaclab = qj[MUJOCO_TO_ISAACLAB]
                dqj_isaaclab = dqj[MUJOCO_TO_ISAACLAB]

                # Build observation (Isaac Lab order)
                obs[0:3] = base_lin_vel * lin_vel_scale
                obs[3:6] = base_ang_vel * ang_vel_scale
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = (qj_isaaclab - default_angles_isaaclab) * dof_pos_scale
                obs[24:36] = dqj_isaaclab * dof_vel_scale
                obs[36:48] = action  # previous action (already in Isaac Lab order)

                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    action = policy(obs_tensor).detach().numpy().squeeze()  # Isaac Lab order

                # Reorder action: Isaac Lab -> MuJoCo
                action_mujoco = action[ISAACLAB_TO_MUJOCO]
                target_dof_pos = action_mujoco * action_scale + default_angles

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
