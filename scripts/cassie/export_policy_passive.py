#!/usr/bin/env python3
"""Export Isaac Lab trained Cassie PASSIVE ankle model to TorchScript.

For policies trained with passive ankles (10 actions, 46 observations):
  - Balance standing task
  - Passive locomotion task
  - Enhanced passive locomotion

Converts an RSL-RL checkpoint (model_*.pt) into a standalone TorchScript (.pt)
file that can be loaded without Isaac Lab or RSL-RL dependencies.

Usage:
    # Auto-detect from run directory
    python scripts/cassie/export_policy_passive.py \
        logs/cassie/rsl_rl/cassie_sim2sim/2026-02-09_18-30-45

    # Specify checkpoint and output
    python scripts/cassie/export_policy_passive.py \
        --checkpoint logs/cassie/rsl_rl/cassie_sim2sim/<run>/model_19999.pt \
        --output logs/cassie/policies/balance_standing.pt
"""

import argparse
import os
import glob

import torch
import torch.nn as nn


class ActorMLP(nn.Module):
    """Standalone Actor network for deployment."""

    def __init__(self, num_obs, num_actions, hidden_dims=(256, 256, 256), activation="elu"):
        super().__init__()

        if activation == "elu":
            act_fn = nn.ELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        else:
            act_fn = nn.ELU()

        layers = []
        in_dim = num_obs
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn)
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_actions))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs):
        return self.mlp(obs)


def infer_dimensions_from_checkpoint(state_dict):
    """Auto-detect num_obs and num_actions from checkpoint weights.

    Returns:
        tuple: (num_obs, num_actions, hidden_dims)
    """
    # Find first layer weight to get num_obs
    first_layer_key = "actor.0.weight"
    if first_layer_key in state_dict:
        first_weight = state_dict[first_layer_key]
        num_obs = first_weight.shape[1]  # input dimension
        print(f"  Detected num_obs: {num_obs}")
    else:
        print("  Warning: Could not find actor.0.weight, using default num_obs=46")
        num_obs = 46

    # Find all hidden layer dimensions
    hidden_dims = []
    layer_idx = 0
    while True:
        weight_key = f"actor.{layer_idx * 2}.weight"
        if weight_key in state_dict:
            shape = state_dict[weight_key].shape
            hidden_dims.append(shape[0])  # output dimension of this layer
            layer_idx += 1
        else:
            break

    # Last entry is num_actions (output layer)
    if len(hidden_dims) >= 2:
        num_actions = hidden_dims[-1]
        hidden_dims = hidden_dims[:-1]  # Remove output layer from hidden dims
        print(f"  Detected num_actions: {num_actions}")
        print(f"  Detected hidden_dims: {hidden_dims}")
    else:
        # Default for Cassie passive
        num_actions = 10
        hidden_dims = [256, 256, 256]
        print(f"  Using default num_actions: {num_actions}")
        print(f"  Using default hidden_dims: {hidden_dims}")

    return num_obs, num_actions, hidden_dims


def export_passive_policy_to_torchscript(checkpoint_path, output_path):
    """Export passive ankle policy checkpoint to TorchScript.

    Auto-detects dimensions from checkpoint (10 or 12 actions, 46 or 48 obs).

    Args:
        checkpoint_path: Path to model_*.pt from Isaac Lab training.
        output_path: Path for the output TorchScript model.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle both raw state_dict and wrapped checkpoint formats
    print("\nCheckpoint structure:")
    if isinstance(checkpoint, dict):
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    else:
        print("  Raw state dict or traced model")

    # Check if this is already a TorchScript model
    if isinstance(checkpoint, torch.jit.ScriptModule):
        print("\nCheckpoint is already a TorchScript model.")
        print(f"Copying to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        checkpoint.save(output_path)
        print("Done!")
        return True

    # Get model state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("  Using 'model_state_dict'")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            print("  Using 'model'")
        else:
            state_dict = checkpoint
            print("  Using checkpoint directly as state dict")
    else:
        state_dict = checkpoint
        print("  Using checkpoint directly as state dict")

    # Print a few keys to verify structure
    print("\nState dict keys (first 5):")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        if hasattr(state_dict[key], "shape"):
            print(f"  {key}: {state_dict[key].shape}")
        else:
            print(f"  {key}: {type(state_dict[key])}")

    # Auto-detect dimensions from checkpoint
    print("\nAuto-detecting network dimensions...")
    num_obs, num_actions, hidden_dims = infer_dimensions_from_checkpoint(state_dict)

    # Validate dimensions for passive ankle
    if num_obs == 46 and num_actions == 10:
        print("\n✓ Detected passive ankle policy (10 actions, 46 obs)")
    elif num_obs == 48 and num_actions == 12:
        print("\n⚠ Detected active ankle policy (12 actions, 48 obs)")
        print("  This script is for passive policies. Use export_policy.py instead.")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return False
    else:
        print(f"\n⚠ Unexpected dimensions: {num_obs} obs, {num_actions} actions")
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return False

    # Create actor model with detected dimensions
    print(f"\nCreating ActorMLP({num_obs}, {num_actions}, hidden_dims={hidden_dims})")
    actor = ActorMLP(num_obs, num_actions, hidden_dims=hidden_dims, activation="elu")

    # Load weights (RSL-RL saves with 'actor.' prefix -> remap to 'mlp.')
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            new_key = key.replace("actor.", "mlp.")
            actor_state[new_key] = value

    if actor_state:
        print(f"\nLoading {len(actor_state)} actor parameters...")
        actor.load_state_dict(actor_state)
        print("✓ Weights loaded successfully")
    else:
        print("\n✗ Warning: No actor weights found with 'actor.' prefix")
        print("Checkpoint may be in wrong format")
        return False

    actor.eval()

    # Test with dummy input
    print(f"\nTesting network...")
    dummy_obs = torch.zeros(1, num_obs)
    with torch.no_grad():
        dummy_action = actor(dummy_obs)
    print(f"  Input shape: {dummy_obs.shape}")
    print(f"  Output shape: {dummy_action.shape}")
    print(f"  Output range: [{dummy_action.min():.3f}, {dummy_action.max():.3f}]")

    # Trace to TorchScript
    print(f"\nTracing to TorchScript...")
    traced_model = torch.jit.trace(actor, dummy_obs)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving to: {output_path}")
    traced_model.save(output_path)
    print("✓ Export complete!")

    # Verify
    print(f"\nVerifying exported model...")
    loaded = torch.jit.load(output_path)
    with torch.no_grad():
        verify_action = loaded(dummy_obs)
    diff = torch.abs(verify_action - dummy_action).max()
    print(f"  Max difference after reload: {diff:.6f}")
    if diff < 1e-5:
        print("✓ Verification passed")
    else:
        print("⚠ Verification failed - large difference detected")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export passive ankle Cassie policy to TorchScript")
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=str,
        help="Run directory (will use latest checkpoint) OR checkpoint path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint file (model_*.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for TorchScript model",
    )

    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.run_dir:
        # Check if run_dir is actually a checkpoint file
        if args.run_dir.endswith(".pt"):
            checkpoint_path = args.run_dir
        else:
            # Find latest checkpoint in run directory
            pattern = os.path.join(args.run_dir, "model_*.pt")
            checkpoints = sorted(glob.glob(pattern))
            if not checkpoints:
                print(f"Error: No checkpoints found in {args.run_dir}")
                return
            checkpoint_path = checkpoints[-1]
            print(f"Using latest checkpoint: {os.path.basename(checkpoint_path)}")
    else:
        parser.print_help()
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        # Default: put in run_dir/exported/policy.pt
        if args.run_dir and not args.run_dir.endswith(".pt"):
            run_dir = args.run_dir
        else:
            run_dir = os.path.dirname(checkpoint_path)

        output_dir = os.path.join(run_dir, "exported")
        output_path = os.path.join(output_dir, "policy.pt")
        print(f"Using default output: {output_path}")

    # Export
    success = export_passive_policy_to_torchscript(checkpoint_path, output_path)

    if success:
        print("\n" + "="*60)
        print("EXPORT SUCCESSFUL")
        print("="*60)
        print(f"\nPolicy exported to: {output_path}")
        print(f"\nTo deploy in MuJoCo:")
        print(f"  python scripts/cassie/deploy_mujoco.py \\")
        print(f"      scripts/cassie/cassie_balance_standing.yaml \\")
        print(f"      --policy {output_path}")
    else:
        print("\n" + "="*60)
        print("EXPORT FAILED")
        print("="*60)


if __name__ == "__main__":
    main()
