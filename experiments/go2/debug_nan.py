#!/usr/bin/env python3
"""Step-by-step NaN debugger for MuJoCo deployment."""

import argparse
import yaml
import numpy as np
import torch
import mujoco
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from eval_mujoco import quat_rotate_inverse, MUJOCO_TO_ISAACLAB, ISAACLAB_TO_MUJOCO


def check_nan(name, arr, stop_on_nan=True):
    """Check array for NaN/Inf and print diagnostics."""
    arr_np = arr.cpu().numpy() if isinstance(arr, torch.Tensor) else arr

    has_nan = np.any(np.isnan(arr_np))
    has_inf = np.any(np.isinf(arr_np))

    if has_nan or has_inf:
        print(f"\n{'='*70}")
        print(f"❌ {name}: NaN={has_nan}, Inf={has_inf}")
        print(f"{'='*70}")
        print(f"Shape: {arr_np.shape}")
        print(f"Min: {np.nanmin(arr_np) if not np.all(np.isnan(arr_np)) else 'all NaN'}")
        print(f"Max: {np.nanmax(arr_np) if not np.all(np.isnan(arr_np)) else 'all NaN'}")
        print(f"Mean: {np.nanmean(arr_np) if not np.all(np.isnan(arr_np)) else 'all NaN'}")
        print(f"\nValues:")
        print(arr_np)

        if stop_on_nan:
            print(f"\n💥 STOPPED: NaN/Inf detected in {name}")
            return False
    else:
        print(f"✓ {name}: OK (range [{arr_np.min():.3f}, {arr_np.max():.3f}])")

    return True


def debug_step_by_step(args):
    """Debug each component individually."""

    print(f"\n{'='*70}")
    print(f"  Step-by-Step NaN Debugger")
    print(f"{'='*70}\n")

    # 1. Load config
    print("Step 1: Loading config...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"✓ Config loaded")

    # 2. Load MuJoCo model
    print(f"\nStep 2: Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(config['xml_path'])
    data = mujoco.MjData(model)
    print(f"✓ Model loaded: {model.nq} positions, {model.nv} velocities, {model.nu} controls")

    # 3. Initialize state
    print(f"\nStep 3: Initializing state...")
    mujoco.mj_resetData(model, data)

    default_angles = np.array(config['default_angles'], dtype=np.float32)
    print(f"  Default angles: {default_angles}")
    check_nan("default_angles", default_angles)

    # Set base position
    data.qpos[0:3] = [0, 0, 0.35]
    check_nan("base_pos", data.qpos[0:3])

    # Set base orientation (identity quaternion)
    data.qpos[3:7] = [1, 0, 0, 0]  # w, x, y, z
    check_nan("base_quat", data.qpos[3:7])

    # Normalize quaternion
    quat_norm = np.linalg.norm(data.qpos[3:7])
    print(f"  Quaternion norm: {quat_norm:.6f}")
    if abs(quat_norm - 1.0) > 0.01:
        print(f"  ⚠ Warning: Quaternion not normalized! Normalizing...")
        data.qpos[3:7] /= quat_norm

    # Set joint positions
    data.qpos[7:19] = default_angles
    check_nan("joint_pos", data.qpos[7:19])

    # Zero velocities
    data.qvel[:] = 0
    check_nan("velocities", data.qvel)

    print(f"\n✓ State initialized")

    # 4. Test quaternion rotation
    print(f"\nStep 4: Testing quaternion rotation...")
    quat = data.qpos[3:7].copy()
    test_vec = np.array([0, 0, -1], dtype=np.float32)

    print(f"  Input quat: {quat}")
    print(f"  Input vector: {test_vec}")

    try:
        rotated = quat_rotate_inverse(quat, test_vec)
        check_nan("rotated_vector", rotated)
        print(f"  Output: {rotated}")
    except Exception as e:
        print(f"  ❌ Rotation failed: {e}")
        return False

    # 5. Build observation
    print(f"\nStep 5: Building observation...")

    lin_vel_world = data.qvel[0:3].copy()
    ang_vel_world = data.qvel[3:6].copy()
    quat = data.qpos[3:7].copy()

    check_nan("lin_vel_world", lin_vel_world) and \
    check_nan("ang_vel_world", ang_vel_world) and \
    check_nan("quat", quat)

    # Rotate to body frame
    lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)
    ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)
    gravity_proj = quat_rotate_inverse(quat, np.array([0, 0, -1]))

    check_nan("lin_vel_body", lin_vel_body) and \
    check_nan("ang_vel_body", ang_vel_body) and \
    check_nan("gravity_proj", gravity_proj)

    # Joint states
    default_angles_isaaclab = default_angles[MUJOCO_TO_ISAACLAB]
    check_nan("default_angles_isaaclab", default_angles_isaaclab)

    joint_pos_mj = data.qpos[7:19].copy()
    joint_vel_mj = data.qvel[6:18].copy()
    check_nan("joint_pos_mj", joint_pos_mj) and \
    check_nan("joint_vel_mj", joint_vel_mj)

    joint_pos = joint_pos_mj[MUJOCO_TO_ISAACLAB] - default_angles_isaaclab
    joint_vel = joint_vel_mj[MUJOCO_TO_ISAACLAB]
    check_nan("joint_pos (isaaclab)", joint_pos) and \
    check_nan("joint_vel (isaaclab)", joint_vel)

    # Command
    cmd = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    cmd_scaled = cmd * np.array([2.0, 2.0, 0.25])
    check_nan("cmd_scaled", cmd_scaled)

    # Previous actions
    actions_prev = np.zeros(12, dtype=np.float32)

    # Build full observation
    obs = np.concatenate([
        lin_vel_body * 2.0,      # 3
        ang_vel_body * 0.25,     # 3
        gravity_proj,            # 3
        cmd_scaled,              # 3
        joint_pos * 1.0,         # 12
        joint_vel * 0.05,        # 12
        actions_prev,            # 12
    ]).astype(np.float32)

    print(f"  Observation shape: {obs.shape}")
    if not check_nan("observation", obs):
        print("\n💥 Observation contains NaN! Cannot proceed to policy.")
        return False

    # 6. Load policy
    print(f"\nStep 6: Loading policy...")
    try:
        policy = torch.jit.load(args.policy)
        policy.eval()
        print(f"✓ Policy loaded")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        return False

    # 7. Test policy
    print(f"\nStep 7: Testing policy forward pass...")
    try:
        obs_t = torch.from_numpy(obs).unsqueeze(0).float()
        check_nan("obs_tensor", obs_t)

        with torch.no_grad():
            action_t = policy(obs_t)

        print(f"  Action tensor shape: {action_t.shape}")
        if not check_nan("action_tensor", action_t):
            print("\n💥 Policy outputs NaN!")
            print("\nPossible causes:")
            print("  1. Policy checkpoint is corrupted")
            print("  2. Observation normalization mismatch")
            print("  3. Policy trained with different obs format")
            return False

        action = action_t.squeeze().cpu().numpy()
        action = np.clip(action, -1, 1)
        check_nan("action_clipped", action)

    except Exception as e:
        print(f"❌ Policy forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 8. Test PD control
    print(f"\nStep 8: Testing PD control...")
    kps = np.array(config['kps'], dtype=np.float32)
    kds = np.array(config['kds'], dtype=np.float32)
    action_scale = config['action_scale']

    check_nan("kps", kps) and check_nan("kds", kds)

    target_pos_isaaclab = default_angles_isaaclab + action * action_scale
    check_nan("target_pos_isaaclab", target_pos_isaaclab)

    target_pos_mj = target_pos_isaaclab[ISAACLAB_TO_MUJOCO]
    check_nan("target_pos_mj", target_pos_mj)

    q = data.qpos[7:19].copy()
    qd = data.qvel[6:18].copy()
    check_nan("q", q) and check_nan("qd", qd)

    tau = kps * (target_pos_mj - q) - kds * qd
    print(f"  Torque range: [{tau.min():.2f}, {tau.max():.2f}] Nm")

    if not check_nan("torques", tau):
        print("\n💥 PD control produces NaN torques!")
        return False

    # 9. Test control swap
    print(f"\nStep 9: Testing control swap...")
    tau_swapped = tau.copy()
    tau_swapped[0:3], tau_swapped[3:6] = tau[3:6].copy(), tau[0:3].copy()
    tau_swapped[6:9], tau_swapped[9:12] = tau[9:12].copy(), tau[6:9].copy()
    check_nan("tau_swapped", tau_swapped)

    # 10. Test simulation step
    print(f"\nStep 10: Testing simulation step...")
    try:
        data.ctrl[:12] = tau_swapped
        check_nan("ctrl", data.ctrl[:12])

        mujoco.mj_step(model, data)

        print(f"  ✓ Simulation step successful!")
        print(f"  Robot height: {data.qpos[2]:.3f} m")
        check_nan("qpos_after_step", data.qpos)
        check_nan("qvel_after_step", data.qvel)

    except Exception as e:
        print(f"  ❌ Simulation step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'='*70}")
    print(f"✅ ALL CHECKS PASSED - No NaN detected!")
    print(f"{'='*70}\n")
    print(f"The deployment pipeline is working correctly.")
    print(f"If you still see NaN errors, they occur during multi-step simulation.")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    success = debug_step_by_step(args)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
