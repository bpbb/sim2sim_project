#!/usr/bin/env python3
"""Collect RMA Phase 2 training data from Isaac Lab.

Runs the RMA environment (which has moderate DR + privileged observations)
and collects (observation_history, privileged_info) pairs for training the
adaptation module.

The RMA env provides:
  - "policy" obs (48 dims): standard locomotion observations
  - "critic" obs (52 dims): policy obs + privileged info (4 dims)

Privileged info (last 4 dims of critic obs):
  [0] ground_friction  — mean static friction
  [1] base_mass_offset — mass offset from nominal (kg)
  [2] kp_scale         — stiffness / nominal
  [3] kd_scale         — damping / nominal

Usage:
    python rma/collect_rma_data.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-v0 \
        --num_samples 100000 --output rma/data/go2_rma_phase2.npz --headless

    # With a trained policy checkpoint:
    python rma/collect_rma_data.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-v0 \
        --checkpoint logs/go2/rma/exported/policy.pt --headless
"""

import argparse
import os
import sys

# Add project root for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

# Parse arguments
parser = argparse.ArgumentParser(description="Collect RMA Phase 2 training data")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-v0",
                    help="Isaac Lab RMA task name")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to policy checkpoint (optional, uses random actions if not provided)")
parser.add_argument("--num_samples", type=int, default=100000,
                    help="Number of samples to collect")
parser.add_argument("--output", type=str, default="rma/data/go2_rma_phase2.npz",
                    help="Output file path")
parser.add_argument("--num_envs", type=int, default=1024,
                    help="Number of parallel environments")
parser.add_argument("--history_length", type=int, default=50,
                    help="Number of past observations for history buffer")
parser.add_argument("--policy_obs_dim", type=int, default=48,
                    help="Policy observation dimension")
parser.add_argument("--privileged_dim", type=int, default=4,
                    help="Privileged observation dimension")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports after Isaac Sim is launched
import gymnasium as gym
import numpy as np
import torch

# Import Go2 extensions (registers gym tasks)
sys.path.append(os.path.join(PROJECT_ROOT, "source", "go2"))
import go2  # noqa: F401


def collect_rma_data(
    task: str,
    checkpoint: str | None,
    num_samples: int,
    output_path: str,
    num_envs: int,
    history_length: int,
    policy_obs_dim: int,
    privileged_dim: int,
    device: str,
):
    """Collect RMA training data from Isaac Lab rollouts.

    Args:
        task: Isaac Lab task name (must have critic obs group).
        checkpoint: Path to policy checkpoint (optional).
        num_samples: Number of (history, privileged) pairs to collect.
        output_path: Output .npz file path.
        num_envs: Number of parallel environments.
        history_length: Steps of observation history.
        policy_obs_dim: Policy observation dimension.
        privileged_dim: Privileged observation dimension.
        device: Device to run on.
    """
    print("=" * 60)
    print("RMA Phase 2 Data Collection (Isaac Lab)")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Checkpoint: {checkpoint or 'None (random actions)'}")
    print(f"Target samples: {num_samples}")
    print(f"Parallel envs: {num_envs}")
    print(f"History length: {history_length}")
    print(f"Policy obs dim: {policy_obs_dim}")
    print(f"Privileged dim: {privileged_dim}")
    print()

    # Load environment config
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = num_envs

    # Create environment
    env = gym.make(task, cfg=env_cfg)

    actual_num_envs = env.unwrapped.num_envs if hasattr(env.unwrapped, 'num_envs') else num_envs
    print(f"Environment created with {actual_num_envs} parallel envs")
    print(f"Observation space: {env.observation_space}")

    # Load policy if checkpoint provided
    policy = None
    if checkpoint is not None and os.path.exists(checkpoint):
        print(f"\nLoading checkpoint: {checkpoint}")
        policy = torch.jit.load(checkpoint, map_location=device)
        policy.eval()
        print("Loaded TorchScript policy")
    elif checkpoint is not None:
        print(f"WARNING: Checkpoint not found: {checkpoint}")
        print("Using random actions instead")

    # Observation history buffer: [num_envs, history_length, obs_dim]
    obs_history = torch.zeros(
        actual_num_envs, history_length, policy_obs_dim,
        device=device, dtype=torch.float32
    )

    # Storage for collected data
    obs_history_list = []
    privileged_obs_list = []

    # Reset environment
    obs_dict, _ = env.reset()

    total_steps = 0
    episodes = 0

    print(f"\nCollecting {num_samples} samples...")
    collected = 0

    while collected < num_samples:
        # Get policy observation
        policy_obs = obs_dict["policy"]
        if not isinstance(policy_obs, torch.Tensor):
            policy_obs = torch.as_tensor(policy_obs, device=device, dtype=torch.float32)

        # Update history buffer
        obs_history = torch.roll(obs_history, shifts=-1, dims=1)
        obs_history[:, -1, :] = policy_obs[:, :policy_obs_dim]

        # Get action
        if policy is not None:
            with torch.no_grad():
                actions = policy(policy_obs)
        else:
            actions = torch.randn(actual_num_envs, 12, device=device) * 0.5

        # Step environment
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        # Extract privileged info from critic observations
        if "critic" in obs_dict:
            critic_obs = obs_dict["critic"]
            if not isinstance(critic_obs, torch.Tensor):
                critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float32)

            # Privileged info is the last `privileged_dim` dimensions of critic obs
            privileged = critic_obs[:, -privileged_dim:]

            # Collect every K steps for diversity (after history is filled)
            if total_steps >= history_length and total_steps % 10 == 0:
                obs_history_list.append(obs_history.cpu().numpy().copy())
                privileged_obs_list.append(privileged.cpu().numpy().copy())
                collected = len(obs_history_list) * actual_num_envs
        else:
            if total_steps == 0:
                print("WARNING: No 'critic' key in obs_dict!")
                print(f"Available keys: {list(obs_dict.keys())}")
                print("Make sure you are using the RMA task with privileged observations.")

        total_steps += 1

        # Handle resets — clear history for reset envs
        done = terminated | truncated
        if hasattr(done, 'any') and done.any():
            reset_ids = done.nonzero(as_tuple=False).squeeze(-1)
            obs_history[reset_ids] = 0.0
            episodes += len(reset_ids)

        # Progress update
        if total_steps % 100 == 0:
            print(f"  Steps: {total_steps}, Episodes: {episodes}, Samples: {collected}/{num_samples}")

    env.close()

    # Stack all collected data
    if len(obs_history_list) > 0:
        obs_history_all = np.concatenate(obs_history_list, axis=0)
        privileged_all = np.concatenate(privileged_obs_list, axis=0)

        # Trim to exact number of samples
        obs_history_all = obs_history_all[:num_samples]
        privileged_all = privileged_all[:num_samples]

        print(f"\nCollected data shapes:")
        print(f"  Observation history: {obs_history_all.shape}")
        print(f"  Privileged obs: {privileged_all.shape}")

        # Save to file
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
    else:
        print("\nERROR: No data collected!")
        print("Make sure the environment provides 'critic' observations.")
        print("Use task: Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-v0")

    # Close simulation
    simulation_app.close()


def main():
    collect_rma_data(
        task=args_cli.task,
        checkpoint=args_cli.checkpoint,
        num_samples=args_cli.num_samples,
        output_path=args_cli.output,
        num_envs=args_cli.num_envs,
        history_length=args_cli.history_length,
        policy_obs_dim=args_cli.policy_obs_dim,
        privileged_dim=args_cli.privileged_dim,
        device="cuda:0",
    )


if __name__ == "__main__":
    main()
