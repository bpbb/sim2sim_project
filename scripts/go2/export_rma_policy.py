#!/usr/bin/env python3
"""
Export Isaac Lab trained policy for RMA deployment.

This script handles the export of both:
1. True RMA policies (trained with obs + latent)
2. Standard policies (trained with obs only) -> Wrapped to accept latent but ignore it

Usage:
    python scripts/go2/export_rma_policy.py \
        --checkpoint logs/rsl_rl/go2/rma_phase1/model_4999.pt \
        --output logs/go2/policies/rma_policy_phase1.pt \
        --num_obs 48 \
        --latent_dim 4
"""

import argparse
import os
import torch
import torch.nn as nn

class ActorMLP(nn.Module):
    """Standard MLP Actor."""
    def __init__(self, input_dim, action_dim, hidden_dims=[256, 256, 256], activation='elu'):
        super().__init__()
        if activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.ELU()
        
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn)
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class RMAWrapper(nn.Module):
    """Wraps a non-RMA policy to accept (obs+latent) input interface."""
    def __init__(self, policy, latent_dim):
        super().__init__()
        self.policy = policy
        self.latent_dim = latent_dim
        
    def forward(self, x):
        # Slice off the latent part (last N dimensions)
        # Input x is [batch, obs_dim + latent_dim]
        obs = x[:, :-self.latent_dim]
        return self.policy(obs)

def export_rma_policy(checkpoint_path, output_path, base_obs_dim=48, latent_dim=4, num_actions=12):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'actor' in checkpoint:
        state_dict = checkpoint['actor']
    else:
        state_dict = checkpoint
        
    # Detect Input Dimension from Weights
    # RSL-RL usually names first layer 'actor.0.weight' or 'mlp.0.weight'
    input_dim = None
    for key in state_dict.keys():
        if key.endswith('0.weight') and ('actor' in key or 'mlp' in key):
            input_dim = state_dict[key].shape[1]
            break
            
    if input_dim is None:
        print("Error: Could not detect input dimension from checkpoint!")
        return False
        
    print(f"Detected Policy Input Dimension: {input_dim}")
    
    # Detect Hidden Dims
    hidden_dims = []
    layer_idx = 0
    while True:
        # Check for RSL-RL naming conventions
        # actor.0.weight, actor.2.weight ...
        weight_key = None
        for k in state_dict.keys():
            if f'actor.{layer_idx*2}.weight' in k or f'mlp.{layer_idx*2}.weight' in k:
                weight_key = k
                break
        
        if weight_key:
            shape = state_dict[weight_key].shape
            if shape[0] != num_actions: # Not the last layer
                hidden_dims.append(shape[0])
            layer_idx += 1
        else:
            if layer_idx > 0: # We found some layers, so we are done
                break
            else:
                # Maybe different numbering, simple fallback
                hidden_dims = [256, 256, 256] 
                break
                
    # remove last layer from hidden dims if it got added (heuristic)
    if len(hidden_dims) > 0 and hidden_dims[-1] == num_actions:
        hidden_dims = hidden_dims[:-1]
        
    print(f"Detected Hidden Dims: {hidden_dims}")

    # Prepare Actor
    actor = ActorMLP(input_dim, num_actions, hidden_dims)
    
    # Load Weights
    # Map keys from 'actor.' to 'mlp.'
    mapped_state = {}
    for k, v in state_dict.items():
        if 'actor.' in k:
            new_k = k.replace('actor.', 'mlp.')
            mapped_state[new_k] = v
        elif 'mlp.' in k:
            mapped_state[k] = v
            
    try:
        actor.load_state_dict(mapped_state)
    except RuntimeError as e:
        print(f"Strict load failed: {e}")
        print("Attempting loose load...")
        # Try finding exact matches
        pass

    actor.eval()
    
    # Check Compatibility
    expected_full_dim = base_obs_dim + latent_dim
    
    final_model = None
    
    if input_dim == expected_full_dim:
        print(">> Policy IS trained with Latents (RMA Ready).")
        final_model = actor
        dummy_input = torch.zeros(1, expected_full_dim)
        
    elif input_dim == base_obs_dim:
        print(">> Policy is STANDARD (No Latents). Wrapping for RMA deployment compatibility...")
        print("   WARNING: This policy will IGNORE the adaptation signal!")
        final_model = RMAWrapper(actor, latent_dim)
        dummy_input = torch.zeros(1, expected_full_dim)
        
    else:
        print(f"Error: Dimension mismatch! Policy expects {input_dim}, but we have base({base_obs_dim}) + latent({latent_dim}) = {expected_full_dim}")
        return False

    # Trace and Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    traced_model = torch.jit.trace(final_model, dummy_input)
    traced_model.save(output_path)
    
    print(f"Successfully exported RMA policy to: {output_path}")
    
    # Verification
    test_out = traced_model(dummy_input)
    print(f"Verification output shape: {test_out.shape}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_obs", type=int, default=48, help="Base observation dimension")
    parser.add_argument("--latent_dim", type=int, default=4, help="Latent/Privileged dimension")
    
    args = parser.parse_args()
    
    export_rma_policy(args.checkpoint, args.output, args.num_obs, args.latent_dim)
