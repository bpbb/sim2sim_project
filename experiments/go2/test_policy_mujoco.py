#!/usr/bin/env python3
"""Test if Isaac Lab policy works in MuJoCo before fine-tuning.

This is a sanity check to ensure:
1. Policy loads correctly
2. Policy outputs are valid (not NaN/Inf)
3. MuJoCo simulation is stable
4. Basic deployment works

Usage:
    python test_policy_mujoco.py \\
        --policy logs/go2/policies/moderate.pt \\
        --config scripts/go2/go2_isaaclab.yaml
"""

import argparse
import yaml
import numpy as np
import torch
import mujoco
import mujoco.viewer
from pathlib import Path
import sys

# Add eval_mujoco to path
sys.path.insert(0, str(Path(__file__).parent))
from eval_mujoco import quat_rotate_inverse, MUJOCO_TO_ISAACLAB, ISAACLAB_TO_MUJOCO


def test_policy(args):
    """Test policy in MuJoCo with safety checks."""

    print(f"\n{'='*70}")
    print(f"  Testing Policy in MuJoCo")
    print(f"{'='*70}\n")

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load policy
    print(f"✓ Loading policy: {args.policy}")
    policy = torch.jit.load(args.policy)
    policy.eval()

    # Check if policy is trainable
    try:
        params = list(policy.parameters())
        print(f"✓ Policy has {len(params)} parameter groups")
        print(f"  - Trainable: {any(p.requires_grad for p in params)}")
    except:
        print(f"✗ Policy is frozen TorchScript (not trainable)")
        print(f"  → Fine-tuning requires converting to Python nn.Module first")

    # Load MuJoCo model
    print(f"\n✓ Loading MuJoCo model: {config['xml_path']}")
    model = mujoco.MjModel.from_xml_path(config['xml_path'])
    data = mujoco.MjData(model)

    # Config
    default_angles = np.array(config['default_angles'], dtype=np.float32)
    default_angles_isaaclab = default_angles[MUJOCO_TO_ISAACLAB]
    kps = np.array(config['kps'], dtype=np.float32)
    kds = np.array(config['kds'], dtype=np.float32)
    action_scale = config['action_scale']
    decimation = config['control_decimation']

    # Initialize safely
    print(f"\n✓ Initializing simulation state")
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.35]  # Fixed safe position
    data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
    data.qpos[7:19] = default_angles  # Default joint angles
    data.qvel[:] = 0

    # Test policy output
    print(f"\n✓ Testing policy forward pass...")

    # Get observation
    lin_vel_world = data.qvel[0:3].copy()
    ang_vel_world = data.qvel[3:6].copy()
    quat = data.qpos[3:7].copy()

    lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)
    ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)
    gravity_proj = quat_rotate_inverse(quat, np.array([0, 0, -1]))

    joint_pos_mj = data.qpos[7:19].copy()
    joint_vel_mj = data.qvel[6:18].copy()
    joint_pos = joint_pos_mj[MUJOCO_TO_ISAACLAB] - default_angles_isaaclab
    joint_vel = joint_vel_mj[MUJOCO_TO_ISAACLAB]

    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)  # Forward command
    cmd_scaled = cmd * np.array([2.0, 2.0, 0.25])
    actions_prev = np.zeros(12, dtype=np.float32)

    obs = np.concatenate([
        lin_vel_body * 2.0,
        ang_vel_body * 0.25,
        gravity_proj,
        cmd_scaled,
        joint_pos * 1.0,
        joint_vel * 0.05,
        actions_prev,
    ]).astype(np.float32)

    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Forward pass
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).unsqueeze(0).float()
        action_t = policy(obs_t)
        action = action_t.squeeze().numpy()

    print(f"  - Action shape: {action.shape}")
    print(f"  - Action range: [{action.min():.3f}, {action.max():.3f}]")

    # Check for NaN/Inf
    if np.any(np.isnan(action)) or np.any(np.isinf(action)):
        print(f"\n✗ ERROR: Policy outputs NaN/Inf!")
        print(f"  → Policy might be corrupted or incompatible")
        return False

    print(f"\n✓ Policy output is valid!")

    # Test PD control
    print(f"\n✓ Testing PD control...")
    target_pos_isaaclab = default_angles_isaaclab + np.clip(action, -1, 1) * action_scale
    target_pos_mj = target_pos_isaaclab[ISAACLAB_TO_MUJOCO]

    q = data.qpos[7:19].copy()
    qd = data.qvel[6:18].copy()
    tau = kps * (target_pos_mj - q) - kds * qd

    print(f"  - Torque range: [{tau.min():.2f}, {tau.max():.2f}] Nm")

    if np.any(np.abs(tau) > 100):
        print(f"  ⚠ Warning: Very high torques detected!")
        print(f"    → This may cause instability")

    if np.any(np.isnan(tau)) or np.any(np.isinf(tau)):
        print(f"\n✗ ERROR: PD control produces NaN/Inf torques!")
        return False

    print(f"✓ PD control is valid!")

    # Test simulation step
    print(f"\n✓ Testing simulation step...")
    try:
        # Swap FL <-> FR and RL <-> RR
        tau_swapped = tau.copy()
        tau_swapped[0:3], tau_swapped[3:6] = tau[3:6].copy(), tau[0:3].copy()
        tau_swapped[6:9], tau_swapped[9:12] = tau[9:12].copy(), tau[6:9].copy()

        data.ctrl[:12] = tau_swapped
        mujoco.mj_step(model, data)

        print(f"  - Simulation step successful!")
        print(f"  - Robot height: {data.qpos[2]:.3f} m")

    except Exception as e:
        print(f"\n✗ ERROR during simulation step:")
        print(f"  {e}")
        return False

    # Run a few more steps
    print(f"\n✓ Running 100 simulation steps...")
    for i in range(100):
        # Update obs
        lin_vel_world = data.qvel[0:3].copy()
        ang_vel_world = data.qvel[3:6].copy()
        quat = data.qpos[3:7].copy()

        lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)
        ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)
        gravity_proj = quat_rotate_inverse(quat, np.array([0, 0, -1]))

        joint_pos_mj = data.qpos[7:19].copy()
        joint_vel_mj = data.qvel[6:18].copy()
        joint_pos = joint_pos_mj[MUJOCO_TO_ISAACLAB] - default_angles_isaaclab
        joint_vel = joint_vel_mj[MUJOCO_TO_ISAACLAB]

        obs = np.concatenate([
            lin_vel_body * 2.0,
            ang_vel_body * 0.25,
            gravity_proj,
            cmd_scaled,
            joint_pos * 1.0,
            joint_vel * 0.05,
            action,
        ]).astype(np.float32)

        # Get action
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).float()
            action_t = policy(obs_t)
            action = action_t.squeeze().numpy()
            action = np.clip(action, -1, 1)

        # PD control
        for _ in range(decimation):
            target_pos_isaaclab = default_angles_isaaclab + action * action_scale
            target_pos_mj = target_pos_isaaclab[ISAACLAB_TO_MUJOCO]

            q = data.qpos[7:19].copy()
            qd = data.qvel[6:18].copy()
            tau = kps * (target_pos_mj - q) - kds * qd

            tau_swapped = tau.copy()
            tau_swapped[0:3], tau_swapped[3:6] = tau[3:6].copy(), tau[0:3].copy()
            tau_swapped[6:9], tau_swapped[9:12] = tau[9:12].copy(), tau[6:9].copy()

            data.ctrl[:12] = tau_swapped
            mujoco.mj_step(model, data)

        if i % 20 == 0:
            print(f"  Step {i:3d}: height={data.qpos[2]:.3f}m, "
                  f"vel=[{lin_vel_body[0]:.2f}, {lin_vel_body[1]:.2f}] m/s")

        # Check for failure
        if data.qpos[2] < 0.15:
            print(f"\n✗ Robot fell at step {i}")
            return False

    print(f"\n{'='*70}")
    print(f"✅ SUCCESS: Policy works in MuJoCo!")
    print(f"{'='*70}\n")
    print(f"Next steps:")
    print(f"  1. Policy deployment works ✓")
    print(f"  2. Ready for fine-tuning (but policy is frozen TorchScript)")
    print(f"  3. Alternative: Train new policy directly in MuJoCo")
    print(f"\n")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    success = test_policy(args)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
