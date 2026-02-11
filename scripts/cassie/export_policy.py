#!/usr/bin/env python3
"""Export Isaac Lab trained Cassie model to TorchScript for MuJoCo deployment.

Converts an RSL-RL checkpoint (model_*.pt) into a standalone TorchScript (.pt)
file that can be loaded without Isaac Lab or RSL-RL dependencies.

The exported model takes a 48-dim observation and returns a 12-dim action
(Isaac Lab joint order, including the 2 passive ankle joints).

Usage:
    python scripts/cassie/export_policy.py \
        --checkpoint logs/cassie/rsl_rl/cassie_direct_dr/<run>/model_29999.pt \
        --output logs/cassie/policies/baseline.pt

    # Or use an already-exported policy from play.py:
    python scripts/cassie/export_policy.py \
        --checkpoint logs/cassie/rsl_rl/cassie/aggressive_dr/<run>/exported/policy.pt \
        --output logs/cassie/policies/aggressive.pt
"""

import argparse
import os

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


def export_isaaclab_to_torchscript(checkpoint_path, output_path, num_obs=48, num_actions=12):
    """Export Isaac Lab RSL-RL checkpoint to TorchScript.

    Args:
        checkpoint_path: Path to model_*.pt from Isaac Lab training.
        output_path: Path for the output TorchScript model.
        num_obs: Number of observations (48 for Cassie without stacking).
        num_actions: Number of actions (12 joints in Isaac Lab).
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle both raw state_dict and wrapped checkpoint formats
    print("\nCheckpoint keys:")
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            print(f"  - {key}")
    else:
        print("  (raw state dict or traced model)")

    # Check if this is already a TorchScript model (e.g. from play.py export)
    if isinstance(checkpoint, torch.jit.ScriptModule):
        print("\nCheckpoint is already a TorchScript model.")
        print(f"Saving to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        checkpoint.save(output_path)
        print("Done!")
        return True

    # Get model state dict
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "actor" in checkpoint:
            state_dict = checkpoint["actor"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Print state dict structure
    print("\nState dict keys:")
    for key in state_dict.keys():
        if hasattr(state_dict[key], "shape"):
            print(f"  - {key}: {state_dict[key].shape}")
        else:
            print(f"  - {key}: {type(state_dict[key])}")

    # Infer network architecture from weight shapes
    hidden_dims = []
    layer_idx = 0

    # RSL-RL naming: actor.0.weight, actor.2.weight, actor.4.weight, ...
    while True:
        weight_key = f"actor.{layer_idx * 2}.weight"
        if weight_key in state_dict:
            shape = state_dict[weight_key].shape
            hidden_dims.append(shape[0])
            print(f"  Layer {layer_idx}: {shape}")
            layer_idx += 1
        else:
            break

    # Last layer is output, remove from hidden_dims
    if hidden_dims:
        hidden_dims = hidden_dims[:-1]

    if not hidden_dims:
        # Cassie default: [256, 256, 256]
        hidden_dims = [256, 256, 256]
        print(f"\nUsing default hidden dims: {hidden_dims}")
    else:
        print(f"\nDetected hidden dims: {hidden_dims}")

    # Create actor model
    actor = ActorMLP(num_obs, num_actions, hidden_dims=hidden_dims, activation="elu")

    # Load weights (RSL-RL saves with 'actor.' prefix -> remap to 'mlp.')
    actor_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor."):
            new_key = key.replace("actor.", "mlp.")
            actor_state[new_key] = value

    if actor_state:
        print(f"\nLoading {len(actor_state)} actor parameters")
        actor.load_state_dict(actor_state)
    else:
        print("\nWarning: No actor weights found with 'actor.' prefix")
        print("Trying direct load...")
        try:
            actor.mlp.load_state_dict(state_dict)
        except Exception as e:
            print(f"Failed to load weights directly: {e}")
            return False

    actor.eval()

    # Test with dummy input
    dummy_obs = torch.zeros(1, num_obs)
    with torch.no_grad():
        dummy_action = actor(dummy_obs)
    print(f"\nTest output shape: {dummy_action.shape}")
    print(f"Test output: {dummy_action}")

    # Export to TorchScript
    print(f"\nExporting to TorchScript: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    traced_model = torch.jit.trace(actor, dummy_obs)
    traced_model.save(output_path)

    # Verify
    print("\nVerifying exported model...")
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        verify_output = loaded_model(dummy_obs)

    if torch.allclose(dummy_action, verify_output):
        print("Verification passed!")
    else:
        print("Verification FAILED - outputs don't match")

    print(f"\nExport complete: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export Isaac Lab Cassie model to TorchScript")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to Isaac Lab checkpoint (model_*.pt or exported/policy.pt)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for TorchScript model",
    )
    parser.add_argument("--num_obs", type=int, default=48, help="Number of observations (default: 48)")
    parser.add_argument("--num_actions", type=int, default=12, help="Number of actions (default: 12)")

    args = parser.parse_args()

    export_isaaclab_to_torchscript(
        args.checkpoint,
        args.output,
        args.num_obs,
        args.num_actions,
    )


if __name__ == "__main__":
    main()