#!/usr/bin/env python3
"""Collect RMA Phase 2 training data from MuJoCo.

Runs the trained policy in MuJoCo with randomized physics parameters and
collects (observation_history, privileged_info) pairs for training the
adaptation module. This is the preferred approach for sim2sim because:
  1. The adaptation module will be deployed in MuJoCo
  2. Observations match MuJoCo deployment distribution
  3. Full control over physics randomization
  4. No Isaac Lab dependency

Privileged info (4 dims):
  [0] friction_scale  — ground friction scale (1.0 = nominal)
  [1] mass_offset     — mass added to base (kg)
  [2] kp_scale        — PD stiffness scale (1.0 = nominal Kp=20)
  [3] kd_scale        — PD damping scale (1.0 = nominal Kd=0.5)

Usage:
    python rma/collect_mujoco_rma_data.py \
        --policy logs/go2/policies/baseline.pt \
        --config scripts/go2/go2_isaaclab.yaml \
        --output rma/data/go2_rma_mujoco.npz \
        --num_episodes 200 --steps_per_episode 500

    # With RMA config (uses default ranges):
    python rma/collect_mujoco_rma_data.py \
        --policy logs/go2/policies/moderate.pt \
        --rma_config rma/configs/go2.yaml
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).parent.parent

# Joint order mapping (same as deploy_mujoco.py)
MUJOCO_TO_ISAACLAB = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ISAACLAB_TO_MUJOCO = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]


def quat_rotate_inverse(q, v):
    """Rotate vector by inverse of quaternion (world to body frame)."""
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([q_x, q_y, q_z]), v)
    return v - q_w * t + np.cross(np.array([q_x, q_y, q_z]), t)


def get_gravity_orientation(quaternion):
    """Get projected gravity vector from quaternion."""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    grav = np.zeros(3)
    grav[0] = 2 * (-qz * qx + qw * qy)
    grav[1] = -2 * (qz * qy + qw * qx)
    grav[2] = 1 - 2 * (qw * qw + qz * qz)
    return grav


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control for joint torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


def apply_ctrl_swap(tau):
    """Convert joint-order torques to MuJoCo ctrl order (FR,FL,RR,RL)."""
    ctrl = np.zeros_like(tau)
    ctrl[0:3] = tau[3:6]    # FR
    ctrl[3:6] = tau[0:3]    # FL
    ctrl[6:9] = tau[9:12]   # RR
    ctrl[9:12] = tau[6:9]   # RL
    return ctrl


def collect_data(
    policy_path: str,
    xml_path: str,
    output_path: str,
    num_episodes: int,
    steps_per_episode: int,
    history_length: int,
    collect_interval: int,
    # Physics params
    default_angles: np.ndarray,
    nominal_kp: float,
    nominal_kd: float,
    action_scale: float,
    sim_dt: float,
    control_decimation: int,
    # Randomization ranges (match moderate DR)
    friction_range: tuple,
    mass_offset_range: tuple,
    kp_scale_range: tuple,
    kd_scale_range: tuple,
    # Command distribution
    cmd_range: tuple,
):
    """Collect RMA data from MuJoCo rollouts with randomized physics."""

    print("=" * 60)
    print("RMA Phase 2 Data Collection (MuJoCo)")
    print("=" * 60)
    print(f"Policy: {policy_path}")
    print(f"MuJoCo model: {xml_path}")
    print(f"Episodes: {num_episodes}, Steps/ep: {steps_per_episode}")
    print(f"History length: {history_length}")
    print(f"Friction range: {friction_range}")
    print(f"Mass offset range: {mass_offset_range}")
    print(f"Kp scale range: {kp_scale_range}")
    print(f"Kd scale range: {kd_scale_range}")
    print()

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt

    # Save nominal values for restoration
    nominal_geom_friction = model.geom_friction.copy()
    nominal_body_mass = model.body_mass.copy()

    # Reorder default angles to Isaac Lab order
    default_angles_isaaclab = default_angles[MUJOCO_TO_ISAACLAB]

    # Load policy
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()
    print(f"Policy loaded: {policy_path}")

    # Storage
    all_obs_history = []
    all_privileged = []

    for ep in range(num_episodes):
        # ── Randomize physics for this episode ────────────────────────
        friction_scale = np.random.uniform(*friction_range)
        mass_offset = np.random.uniform(*mass_offset_range)
        kp_scale = np.random.uniform(*kp_scale_range)
        kd_scale = np.random.uniform(*kd_scale_range)

        # Apply friction (scale all geom frictions)
        model.geom_friction[:] = nominal_geom_friction * friction_scale

        # Apply mass offset to base body (body index 1 in typical MuJoCo model)
        model.body_mass[:] = nominal_body_mass.copy()
        model.body_mass[1] += mass_offset  # base body

        # Compute effective PD gains
        kps = np.full(12, nominal_kp * kp_scale, dtype=np.float32)
        kds = np.full(12, nominal_kd * kd_scale, dtype=np.float32)

        # Privileged info for this episode
        privileged = np.array([
            friction_scale,
            mass_offset,
            kp_scale,
            kd_scale,
        ], dtype=np.float32)

        # ── Reset simulation ──────────────────────────────────────────
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = [0, 0, 0.35]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:] = default_angles
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)

        # Random command for this episode
        cmd = np.array([
            np.random.uniform(*cmd_range),
            np.random.uniform(*cmd_range),
            np.random.uniform(-1.0, 1.0),
        ], dtype=np.float32)
        cmd_scaled = cmd * np.array([2.0, 2.0, 0.25], dtype=np.float32)

        # Episode state
        action = np.zeros(12, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        obs = np.zeros(48, dtype=np.float32)
        obs_history = np.zeros((history_length, 48), dtype=np.float32)
        counter = 0

        for step in range(steps_per_episode):
            # Apply PD torques with ctrl swap
            tau = pd_control(target_dof_pos, data.qpos[7:], kps,
                             np.zeros_like(kds), data.qvel[6:], kds)
            data.ctrl[0:3] = tau[3:6]   # FR
            data.ctrl[3:6] = tau[0:3]   # FL
            data.ctrl[6:9] = tau[9:12]  # RR
            data.ctrl[9:12] = tau[6:9]  # RL

            mujoco.mj_step(model, data)
            counter += 1

            # Policy step (at control frequency)
            if counter % control_decimation == 0:
                qj = data.qpos[7:]
                dqj = data.qvel[6:]
                quat = data.qpos[3:7]

                # Body frame velocities
                base_lin_vel = quat_rotate_inverse(quat, data.qvel[0:3])
                base_ang_vel = quat_rotate_inverse(quat, data.qvel[3:6])
                gravity_orientation = get_gravity_orientation(quat)

                # Reorder to Isaac Lab order
                qj_isaaclab = qj[MUJOCO_TO_ISAACLAB]
                dqj_isaaclab = dqj[MUJOCO_TO_ISAACLAB]

                # Build observation (same as deploy_mujoco.py)
                obs[0:3] = base_lin_vel * 2.0
                obs[3:6] = base_ang_vel * 0.25
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd_scaled
                obs[12:24] = qj_isaaclab - default_angles_isaaclab
                obs[24:36] = dqj_isaaclab * 0.05
                obs[36:48] = action

                # Update history (shift left, add new)
                obs_history[:-1] = obs_history[1:]
                obs_history[-1] = obs.copy()

                # Get policy action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                
                # IMPORTANT: For RMA Phase 1, the policy expects [obs, privileged].
                # privileged array: [friction, mass, kp, kd]
                priv_tensor = torch.from_numpy(privileged).unsqueeze(0).float()
                
                # Input to policy: 52 dims
                policy_input = torch.cat([obs_tensor, priv_tensor], dim=1)
                
                with torch.no_grad():
                    action = policy(policy_input).detach().numpy().squeeze()

                # Convert action to MuJoCo target positions
                action_mujoco = action[ISAACLAB_TO_MUJOCO]
                target_dof_pos = action_mujoco * action_scale + default_angles

                # Check for fall
                if data.qpos[2] < 0.15:
                    break

            # Collect data at interval (after history is filled)
            policy_step = counter // control_decimation
            if (policy_step >= history_length
                    and counter % control_decimation == 0
                    and policy_step % collect_interval == 0):
                all_obs_history.append(obs_history.copy())
                all_privileged.append(privileged.copy())

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{num_episodes}, "
                  f"samples: {len(all_obs_history)}, "
                  f"friction={friction_scale:.2f}, mass={mass_offset:.2f}, "
                  f"kp_s={kp_scale:.2f}, kd_s={kd_scale:.2f}")

    # Restore nominal values
    model.geom_friction[:] = nominal_geom_friction
    model.body_mass[:] = nominal_body_mass

    # Save data
    obs_history_all = np.array(all_obs_history, dtype=np.float32)
    privileged_all = np.array(all_privileged, dtype=np.float32)

    print(f"\nCollected data shapes:")
    print(f"  Observation history: {obs_history_all.shape}")
    print(f"  Privileged obs: {privileged_all.shape}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        obs_history=obs_history_all,
        privileged=privileged_all,
    )
    print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    print("\nNext step: Train the adaptation module with:")
    print(f"  python rma/train_adaptation.py --data {output_path}")

    return obs_history_all, privileged_all


def main():
    parser = argparse.ArgumentParser(description="Collect RMA data from MuJoCo")
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to trained policy (TorchScript .pt)")
    parser.add_argument("--config", type=str, default="scripts/go2/go2_isaaclab.yaml",
                        help="Go2 deployment config YAML")
    parser.add_argument("--rma_config", type=str, default="rma/configs/go2.yaml",
                        help="RMA config YAML (for randomization ranges)")
    parser.add_argument("--output", type=str, default="rma/data/go2_rma_mujoco.npz",
                        help="Output .npz file path")
    parser.add_argument("--num_episodes", type=int, default=200,
                        help="Number of episodes to collect")
    parser.add_argument("--steps_per_episode", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--history_length", type=int, default=50,
                        help="Observation history length")
    parser.add_argument("--collect_interval", type=int, default=5,
                        help="Collect sample every N policy steps")
    args = parser.parse_args()

    # Resolve paths
    def resolve(p):
        if os.path.isabs(p):
            return p
        return str(PROJECT_ROOT / p)

    # Load deployment config
    config_path = resolve(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    xml_path = config["xml_path"]
    if "{SCRIPT_DIR}" in xml_path:
        script_dir = os.path.dirname(os.path.abspath(config_path))
        xml_path = xml_path.replace("{SCRIPT_DIR}", script_dir)

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    nominal_kp = config["kps"][0]
    nominal_kd = config["kds"][0]
    action_scale = config["action_scale"]
    sim_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    # Load RMA config for randomization ranges
    rma_config_path = resolve(args.rma_config)
    with open(rma_config_path) as f:
        rma_config = yaml.safe_load(f)

    priv = rma_config["privileged_info"]["components"]
    friction_range = tuple(next(c for c in priv if c["name"] == "ground_friction_scale")["range"])
    mass_range = tuple(next(c for c in priv if c["name"] == "base_mass_offset")["range"])
    kp_range = tuple(next(c for c in priv if c["name"] == "kp_scale")["range"])
    kd_range = tuple(next(c for c in priv if c["name"] == "kd_scale")["range"])

    collect_data(
        policy_path=resolve(args.policy),
        xml_path=xml_path,
        output_path=resolve(args.output),
        num_episodes=args.num_episodes,
        steps_per_episode=args.steps_per_episode,
        history_length=args.history_length,
        collect_interval=args.collect_interval,
        default_angles=default_angles,
        nominal_kp=nominal_kp,
        nominal_kd=nominal_kd,
        action_scale=action_scale,
        sim_dt=sim_dt,
        control_decimation=control_decimation,
        friction_range=friction_range,
        mass_offset_range=mass_range,
        kp_scale_range=kp_range,
        kd_scale_range=kd_range,
        cmd_range=(-1.0, 1.0),
    )


if __name__ == "__main__":
    main()
