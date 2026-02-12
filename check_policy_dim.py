
import torch
import os

policy_path = "Two-wheel-Legged-Bot/main/logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/model_4999.pt"

if not os.path.exists(policy_path):
    print(f"File not found: {policy_path}")
    exit(1)

try:
    ckpt = torch.load(policy_path, map_location="cpu", weights_only=False)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    exit(1)

state = ckpt.get("model_state_dict", ckpt)
weight = state.get("actor.0.weight")

if weight is not None:
    print(f"Actor input dim: {weight.shape[1]}")
    print(f"Actor output dim: {weight.shape[0]}")
else:
    print("Could not find actor.0.weight")
    print("Keys:", state.keys())
