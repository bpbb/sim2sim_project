#!/usr/bin/env python3
"""
Deploy RMA (Rapid Motor Adaptation) Policy in MuJoCo.

This script runs the trained RMA policy and adaptation module in MuJoCo.
It maintains an observation history buffer, estimates environment parameters
(or latents) using the adaptation module, and feeds them to the policy.

Usage:
    python scripts/go2/deploy_rma_mujoco.py \
        --config scripts/go2/go2_isaaclab.yaml \
        --policy logs/rsl_rl/go2/rma_phase1/policy.pt \
        --adapter rma/models/go2/adaptation_jit.pt \
        --info rma/models/go2/adaptation_info.pt

"""

import argparse
import os
import time
import numpy as np
import torch
import yaml
import mujoco
import mujoco.viewer

# Helper functions from deploy_mujoco.py
def quat_rotate_inverse(q, v):
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([q_x, q_y, q_z]), v)
    return v - q_w * t + np.cross(np.array([q_x, q_y, q_z]), t)

def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

# Joint mapping - SAME AS deploy_mujoco.py
MUJOCO_TO_ISAACLAB = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ISAACLAB_TO_MUJOCO = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

def main():
    parser = argparse.ArgumentParser(description="Deploy RMA Policy in MuJoCo")
    parser.add_argument("--config", type=str, default="scripts/go2/go2_isaaclab.yaml", help="Path to config file")
    parser.add_argument("--policy", type=str, required=True, help="Path to Phase 1 policy (TorchScript)")
    parser.add_argument("--adapter", type=str, required=True, help="Path to Phase 2 adaptation module (TorchScript)")
    parser.add_argument("--info", type=str, default=None, help="Path to adaptation info (optional, for auto-detecting history length)")
    parser.add_argument("--sim_duration", type=float, default=60.0, help="Simulation duration")
    parser.add_argument("--headless", action="store_true", help="Run without visualization")
    args = parser.parse_args()

    # Load Config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Handle paths
    if "{SCRIPT_DIR}" in config["xml_path"]:
        script_dir = os.path.dirname(os.path.abspath(args.config))
        config["xml_path"] = config["xml_path"].replace("{SCRIPT_DIR}", script_dir)

    # Simulation params
    sim_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    
    # Scaling
    lin_vel_scale = config.get("lin_vel_scale", 2.0)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    
    # Load Models
    print(f"Loading Policy: {args.policy}")
    policy = torch.jit.load(args.policy)
    policy.eval()

    print(f"Loading Adapter: {args.adapter}")
    adapter = torch.jit.load(args.adapter)
    adapter.eval()

    # Determine history length and input type (MLP vs TCN)
    history_length = 50  # Default
    obs_dim = 48
    use_tcn = False

    if args.info and os.path.exists(args.info):
        print(f"Loading Info: {args.info}")
        info = torch.load(args.info)
        history_length = info.get("history_length", 50)
        obs_dim = info.get("obs_dim", 48)
        use_tcn = (info.get("input_type", "mlp") == "tcn")
        print(f"  History Length: {history_length}")
        print(f"  Obs Dim: {obs_dim}")
        print(f"  Model Type: {'TCN' if use_tcn else 'MLP'}")

    # MuJoCo Setup
    xml_path = config["xml_path"]
    if not os.path.isabs(xml_path):
        xml_path = os.path.abspath(xml_path)
    
    print(f"Loading MuJoCo model: {xml_path}")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = sim_dt

    # Reset
    d.qpos[0:3] = [0, 0, 0.35]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    # State
    obs_history = np.zeros((history_length, obs_dim), dtype=np.float32)
    obs = np.zeros(obs_dim, dtype=np.float32)
    action = np.zeros(12, dtype=np.float32) # Last action
    target_dof_pos = default_angles.copy()
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32) # Forward command

    default_angles_isaac = default_angles[MUJOCO_TO_ISAACLAB]
    
    # Simulation Function
    def run_simulation_step(viewer_sync=None):
        nonlocal counter, obs, action, target_dof_pos, obs_history
        
        # PD Control
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[0:3] = tau[3:6]
        d.ctrl[3:6] = tau[0:3]
        d.ctrl[6:9] = tau[9:12]
        d.ctrl[9:12] = tau[6:9]

        mujoco.mj_step(m, d)
        counter += 1

        # Policy Step
        if counter % control_decimation == 0:
            # 1. Build Observation (Isaac Lab Order)
            qj = d.qpos[7:]
            dqj = d.qvel[6:]
            quat = d.qpos[3:7]
            
            base_lin_vel = quat_rotate_inverse(quat, d.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(quat, d.qvel[3:6])
            gravity = get_gravity_orientation(quat)
            
            qj_isaac = qj[MUJOCO_TO_ISAACLAB]
            dqj_isaac = dqj[MUJOCO_TO_ISAACLAB]

            obs[0:3] = base_lin_vel * lin_vel_scale
            obs[3:6] = base_ang_vel * ang_vel_scale
            obs[6:9] = gravity
            obs[9:12] = cmd * cmd_scale
            obs[12:24] = (qj_isaac - default_angles_isaac) * dof_pos_scale
            obs[24:36] = dqj_isaac * dof_vel_scale
            obs[36:48] = action

            # 2. Update History
            obs_history[:-1] = obs_history[1:]
            obs_history[-1] = obs.copy()

            # 3. Adaptation Step
            with torch.no_grad():
                if use_tcn:
                    hist_tensor = torch.from_numpy(obs_history).unsqueeze(0).float()
                    latent = adapter(hist_tensor)
                else:
                    hist_flat = torch.from_numpy(obs_history.flatten()).unsqueeze(0).float()
                    latent = adapter(hist_flat)
            
            # 4. Policy Step
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
            policy_input = torch.cat([obs_tensor, latent], dim=1)
            
            with torch.no_grad():
                action = policy(policy_input).numpy().squeeze()
            
            # 5. Convert Action to Targets
            action_mujoco = action[ISAACLAB_TO_MUJOCO]
            target_dof_pos = action_mujoco * action_scale + default_angles

        if viewer_sync:
            viewer_sync()
            
    # Run
    start_time = time.time()
    counter = 0
    print(f"Starting simulation (Headless: {args.headless})...")

    if args.headless:
        while time.time() - start_time < args.sim_duration:
            run_simulation_step()
            # Sim is faster than real time, sleep if needed? 
            # Usually strict real-time not needed for headless check, but helps logic
            # Just run as fast as possible or add simple sleep
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running() and time.time() - start_time < args.sim_duration:
                step_start = time.time()
                run_simulation_step(viewer.sync)
                
                time_until_next = m.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)

    print("Experiment Complete.")

if __name__ == "__main__":
    main()
