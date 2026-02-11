#!/usr/bin/env python3
"""Fine-tune Isaac Lab policy in MuJoCo for better deployment performance.

This script implements domain adaptation by:
1. Loading an Isaac Lab-trained policy
2. Training it further using MuJoCo rollouts
3. Optimizing for MuJoCo-specific dynamics

Usage:
    python finetune_mujoco.py \\
        --policy logs/go2/policies/moderate.pt \\
        --config scripts/go2/go2_isaaclab.yaml \\
        --output experiments/go2/adapted_policies/moderate_adapted.pt \\
        --steps 50000
"""

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import mujoco
import mujoco.viewer
from collections import deque
import time

# Add eval_mujoco to path for utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_mujoco import quat_rotate_inverse, MUJOCO_TO_ISAACLAB, ISAACLAB_TO_MUJOCO


class MuJoCoVecEnv:
    """Vectorized MuJoCo environment for parallel training."""

    def __init__(self, xml_path: str, config: dict, num_envs: int = 64):
        self.num_envs = num_envs
        self.config = config

        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = config['simulation_dt']

        # Create data instances for each env
        self.data_list = [mujoco.MjData(self.model) for _ in range(num_envs)]

        # Config
        self.default_angles = np.array(config['default_angles'], dtype=np.float32)
        self.default_angles_isaaclab = self.default_angles[MUJOCO_TO_ISAACLAB]
        self.kps = np.array(config['kps'], dtype=np.float32)
        self.kds = np.array(config['kds'], dtype=np.float32)
        self.action_scale = config['action_scale']
        self.lin_vel_scale = config['lin_vel_scale']
        self.ang_vel_scale = config['ang_vel_scale']
        self.dof_pos_scale = config['dof_pos_scale']
        self.dof_vel_scale = config['dof_vel_scale']
        self.decimation = config['control_decimation']

        # Command sampling ranges
        self.cmd_ranges = {
            'vx': (-1.0, 1.5),
            'vy': (-0.5, 0.5),
            'wz': (-1.0, 1.0)
        }

        # Episode tracking
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.max_episode_length = 1000  # 20 seconds at 50Hz

        # Reward tracking
        self.rew_buf = np.zeros(num_envs, dtype=np.float32)

    def reset(self, env_ids=None):
        """Reset specified environments (or all if None)."""
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        obs_list = []
        for i in env_ids:
            data = self.data_list[i]

            # Reset state
            mujoco.mj_resetData(self.model, data)

            # Random initial position
            data.qpos[0:3] = [
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                0.35 + np.random.uniform(-0.05, 0.05)
            ]

            # Random initial orientation (small perturbations)
            angle = np.random.uniform(-0.1, 0.1)
            axis = np.array([0, 0, 1])
            data.qpos[3] = np.cos(angle / 2)
            data.qpos[4:7] = np.sin(angle / 2) * axis

            # Set joint positions with noise
            noise = np.random.uniform(-0.1, 0.1, 12)
            data.qpos[7:19] = self.default_angles + noise

            # Zero velocities
            data.qvel[:] = 0

            # Sample random command
            if not hasattr(self, 'commands'):
                self.commands = np.zeros((self.num_envs, 3), dtype=np.float32)

            self.commands[i] = [
                np.random.uniform(*self.cmd_ranges['vx']),
                np.random.uniform(*self.cmd_ranges['vy']),
                np.random.uniform(*self.cmd_ranges['wz'])
            ]

            # Reset episode length
            self.episode_lengths[i] = 0

            # Get observation
            obs = self._get_obs(i)
            obs_list.append(obs)

        return np.array(obs_list)

    def _get_obs(self, env_id):
        """Get observation for one environment (48-dim)."""
        data = self.data_list[env_id]

        # Base linear velocity (world frame)
        lin_vel_world = data.qvel[0:3].copy()

        # Base angular velocity (world frame)
        ang_vel_world = data.qvel[3:6].copy()

        # Base orientation (quaternion w,x,y,z)
        quat = data.qpos[3:7].copy()

        # Rotate velocities to body frame
        lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)
        ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)

        # Projected gravity
        gravity = np.array([0, 0, -1], dtype=np.float32)
        projected_gravity = quat_rotate_inverse(quat, gravity)

        # Joint positions and velocities (MuJoCo order)
        joint_pos_mj = data.qpos[7:19].copy()
        joint_vel_mj = data.qvel[6:18].copy()

        # Convert to Isaac Lab order
        joint_pos = joint_pos_mj[MUJOCO_TO_ISAACLAB] - self.default_angles_isaaclab
        joint_vel = joint_vel_mj[MUJOCO_TO_ISAACLAB]

        # Actions (previous)
        if not hasattr(self, 'last_actions'):
            self.last_actions = np.zeros((self.num_envs, 12), dtype=np.float32)
        actions = self.last_actions[env_id]

        # Commands (scaled)
        cmd = self.commands[env_id]
        cmd_scaled = cmd * np.array([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale])

        # Build observation (48-dim)
        obs = np.concatenate([
            lin_vel_body * self.lin_vel_scale,           # 3
            ang_vel_body * self.ang_vel_scale,           # 3
            projected_gravity,                            # 3
            cmd_scaled,                                   # 3
            joint_pos * self.dof_pos_scale,              # 12
            joint_vel * self.dof_vel_scale,              # 12
            actions,                                      # 12
        ]).astype(np.float32)

        # Replace NaN/Inf with zeros (safety)
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.zeros(48, dtype=np.float32)

        return obs

    def step(self, actions):
        """Step all environments with given actions."""
        # Store actions
        self.last_actions = actions.copy()

        obs_list = []
        rew_list = []
        done_list = []

        for i in range(self.num_envs):
            data = self.data_list[i]
            action = actions[i]

            # Clip actions
            action = np.clip(action, -1.0, 1.0)

            # Compute target joint positions
            target_pos_isaaclab = self.default_angles_isaaclab + action * self.action_scale
            target_pos_mj = target_pos_isaaclab[ISAACLAB_TO_MUJOCO]

            # Apply PD control for decimation steps
            for _ in range(self.decimation):
                # Get current joint state
                q = data.qpos[7:19].copy()
                qd = data.qvel[6:18].copy()

                # PD control
                tau = self.kps * (target_pos_mj - q) - self.kds * qd

                # Check for NaN/Inf in torques
                if np.any(np.isnan(tau)) or np.any(np.isinf(tau)):
                    # Zero torques to prevent instability
                    tau = np.zeros_like(tau)

                # Swap FL <-> FR and RL <-> RR for MuJoCo
                tau_swapped = tau.copy()
                tau_swapped[0:3], tau_swapped[3:6] = tau[3:6].copy(), tau[0:3].copy()  # FL <-> FR
                tau_swapped[6:9], tau_swapped[9:12] = tau[9:12].copy(), tau[6:9].copy()  # RL <-> RR

                # Clip torques to reasonable range
                tau_swapped = np.clip(tau_swapped, -100, 100)

                # Apply control
                data.ctrl[:12] = tau_swapped

                # Step physics
                mujoco.mj_step(self.model, data)

            # Get reward
            reward = self._compute_reward(i)
            rew_list.append(reward)

            # Check termination
            self.episode_lengths[i] += 1
            done = self._check_termination(i)
            done_list.append(done)

            # Get next observation
            if done:
                # Store final obs
                obs = self._get_obs(i)
                obs_list.append(obs)

                # Reset this env
                new_obs = self.reset(env_ids=[i])[0]
                obs_list[-1] = new_obs  # Replace with reset obs
            else:
                obs = self._get_obs(i)
                obs_list.append(obs)

        return np.array(obs_list), np.array(rew_list), np.array(done_list), {}

    def _compute_reward(self, env_id):
        """Compute reward for tracking commands."""
        data = self.data_list[env_id]

        # Get body velocity
        lin_vel_world = data.qvel[0:3]
        ang_vel_world = data.qvel[3:6]
        quat = data.qpos[3:7]
        lin_vel_body = quat_rotate_inverse(quat, lin_vel_world)
        ang_vel_body = quat_rotate_inverse(quat, ang_vel_world)

        # Command tracking
        cmd = self.commands[env_id]
        vel_error = np.sum((lin_vel_body[:2] - cmd[:2])**2)
        ang_error = (ang_vel_body[2] - cmd[2])**2

        # Rewards
        rew_tracking = -vel_error - ang_error

        # Orientation (penalize tilt)
        gravity_proj = quat_rotate_inverse(quat, np.array([0, 0, -1]))
        rew_orientation = -np.sum((gravity_proj - np.array([0, 0, -1]))**2)

        # Action smoothness
        if env_id < len(self.last_actions):
            action_rate = np.sum(np.square(self.last_actions[env_id]))
            rew_action = -0.01 * action_rate
        else:
            rew_action = 0.0

        # Total reward
        reward = rew_tracking * 2.0 + rew_orientation * 0.5 + rew_action

        return float(reward)

    def _check_termination(self, env_id):
        """Check if episode should terminate."""
        data = self.data_list[env_id]

        # NaN check (simulation instability)
        if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
            return True
        if np.any(np.isinf(data.qpos)) or np.any(np.isinf(data.qvel)):
            return True

        # Height check
        if data.qpos[2] < 0.15:
            return True

        # Max episode length
        if self.episode_lengths[env_id] >= self.max_episode_length:
            return True

        return False


class PPOBuffer:
    """Rollout buffer for PPO."""

    def __init__(self, num_envs, horizon, obs_dim, act_dim):
        self.num_envs = num_envs
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Storage
        self.obs_buf = np.zeros((horizon, num_envs, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((horizon, num_envs, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((horizon, num_envs), dtype=np.float32)
        self.val_buf = np.zeros((horizon, num_envs), dtype=np.float32)
        self.logp_buf = np.zeros((horizon, num_envs), dtype=np.float32)
        self.done_buf = np.zeros((horizon, num_envs), dtype=np.float32)

        self.ptr = 0

    def store(self, obs, act, rew, val, logp, done):
        """Store one step."""
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def get(self):
        """Get all data and compute advantages."""
        assert self.ptr == self.horizon

        # Flatten
        obs = self.obs_buf.reshape(-1, self.obs_dim)
        act = self.act_buf.reshape(-1, self.act_dim)
        logp = self.logp_buf.reshape(-1)

        # Compute advantages (simple TD error for now)
        # In full PPO, you'd use GAE
        adv = np.zeros_like(self.rew_buf)
        for t in range(self.horizon):
            if t == self.horizon - 1:
                adv[t] = self.rew_buf[t] - self.val_buf[t]
            else:
                adv[t] = self.rew_buf[t] + 0.99 * self.val_buf[t+1] * (1 - self.done_buf[t]) - self.val_buf[t]

        adv = adv.reshape(-1)
        ret = (adv + self.val_buf.reshape(-1))

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Reset
        self.ptr = 0

        return obs, act, logp, adv, ret


def finetune_policy(args):
    """Fine-tune policy in MuJoCo."""

    print(f"\n{'='*70}")
    print(f"  MuJoCo Policy Fine-Tuning")
    print(f"{'='*70}\n")

    # Note: MuJoCo warnings are handled gracefully by resetting unstable environments

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load policy
    print(f"Loading policy: {args.policy}")
    policy_module = torch.jit.load(args.policy)
    policy_module.eval()

    # Extract actor and critic
    # The TorchScript module should have actor and value_net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy_module = policy_module.to(device)

    # Create environment
    print(f"Creating {args.num_envs} parallel MuJoCo environments...")
    xml_path = config['xml_path']
    env = MuJoCoVecEnv(xml_path, config, num_envs=args.num_envs)

    # Create buffer
    horizon = args.horizon
    buffer = PPOBuffer(args.num_envs, horizon, obs_dim=48, act_dim=12)

    # Optimizer
    optimizer = optim.Adam(policy_module.parameters(), lr=args.lr)

    # Training loop
    print(f"\nFine-tuning for {args.steps} total steps...")
    print(f"Horizon: {horizon}, Num envs: {args.num_envs}, Updates: {args.steps // (horizon * args.num_envs)}\n")

    obs = env.reset()
    total_steps = 0
    update_count = 0

    while total_steps < args.steps:
        # Collect rollout
        for step in range(horizon):
            obs_t = torch.from_numpy(obs).float().to(device)

            with torch.no_grad():
                # Get action and value from policy
                action_mean = policy_module(obs_t)
                # Add exploration noise
                action = action_mean + torch.randn_like(action_mean) * 0.1
                action = torch.clamp(action, -1.0, 1.0)

                # Compute log prob (assuming Gaussian)
                logp = -0.5 * torch.sum((action - action_mean)**2 / (0.1**2), dim=1)

                # Value estimate (use a simple baseline for now)
                value = torch.zeros(args.num_envs).to(device)

            # Step environment
            next_obs, reward, done, _ = env.step(action.cpu().numpy())

            # Store
            buffer.store(
                obs,
                action.cpu().numpy(),
                reward,
                value.cpu().numpy(),
                logp.cpu().numpy(),
                done
            )

            obs = next_obs
            total_steps += args.num_envs

        # Update policy
        obs_buf, act_buf, logp_old, adv, ret = buffer.get()

        # Convert to tensors
        obs_t = torch.from_numpy(obs_buf).float().to(device)
        act_t = torch.from_numpy(act_buf).float().to(device)
        logp_old_t = torch.from_numpy(logp_old).float().to(device)
        adv_t = torch.from_numpy(adv).float().to(device)
        ret_t = torch.from_numpy(ret).float().to(device)

        # PPO update
        for epoch in range(args.ppo_epochs):
            # Forward pass
            action_mean = policy_module(obs_t)

            # Log prob
            logp = -0.5 * torch.sum((act_t - action_mean)**2 / (0.1**2), dim=1)

            # PPO ratio
            ratio = torch.exp(logp - logp_old_t)
            clip_adv = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv_t
            loss_policy = -torch.min(ratio * adv_t, clip_adv).mean()

            # Optimize
            optimizer.zero_grad()
            loss_policy.backward()
            torch.nn.utils.clip_grad_norm_(policy_module.parameters(), 0.5)
            optimizer.step()

        # Log
        update_count += 1
        if update_count % 10 == 0:
            mean_reward = buffer.rew_buf.mean()
            print(f"Update {update_count:4d} | Steps {total_steps:7d} | Reward: {mean_reward:7.3f}")

    # Save adapted policy
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(policy_module, str(output_path))
    print(f"\n✓ Saved adapted policy to: {output_path}")

    print(f"\nFine-tuning complete! Run deployment to see improvements.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Isaac Lab policy in MuJoCo")
    parser.add_argument('--policy', type=str, required=True, help='Path to Isaac Lab policy (.pt)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output', type=str, required=True, help='Path to save adapted policy')
    parser.add_argument('--steps', type=int, default=50000, help='Total training steps')
    parser.add_argument('--num_envs', type=int, default=64, help='Number of parallel environments')
    parser.add_argument('--horizon', type=int, default=100, help='Rollout horizon')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--ppo_epochs', type=int, default=5, help='PPO update epochs')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon')

    args = parser.parse_args()
    finetune_policy(args)


if __name__ == '__main__':
    main()
