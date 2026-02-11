#!/usr/bin/env python3
"""RMA Training Script.

This script implements the two-phase RMA training:
- Phase 1: Train base policy with privileged encoder
- Phase 2: Train adaptation module to match encoder

Usage:
    # Phase 1: Train base policy
    python rma/train_rma.py --robot go2 --phase 1

    # Phase 2: Train adaptation module (after Phase 1)
    python rma/train_rma.py --robot go2 --phase 2 --checkpoint logs/rsl_rl/go2/rma_phase1/model_6000.pt

    # Both phases sequentially
    python rma/train_rma.py --robot go2 --phase both
"""

import argparse
import os
from pathlib import Path

import yaml


def load_config(robot: str) -> dict:
    """Load robot-specific RMA configuration."""
    config_path = Path(__file__).parent / "configs" / f"{robot}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f)


def train_phase1(robot: str, config: dict, num_envs: int = 4096):
    """Train base policy with privileged encoder.

    In this phase:
    - Environment provides privileged info (friction, mass, etc.)
    - Encoder converts privileged info to latent vector
    - Base policy learns (obs, latent) → action
    """
    print("=" * 60)
    print(f"RMA PHASE 1: Training Base Policy for {robot.upper()}")
    print("=" * 60)
    print()
    print("This phase trains the base policy using privileged information.")
    print("The encoder converts ground-truth parameters to latent encoding.")
    print()

    # TODO: Integrate with Isaac Lab training
    # This would involve:
    # 1. Creating RMA environment wrapper that provides privileged obs
    # 2. Using RMAPolicy instead of standard policy
    # 3. Training with PPO using the encoder

    phase1_cfg = config["phase1"]
    print(f"Configuration:")
    print(f"  - Max iterations: {phase1_cfg['max_iterations']}")
    print(f"  - Learning rate: {phase1_cfg['learning_rate']}")
    print(f"  - Experiment: {phase1_cfg['experiment_name']}")
    print()

    print("To train Phase 1, run:")
    print(f"  python scripts/rsl_rl/train.py --task Isaac-Velocity-{robot.capitalize()}-Direct-RMA-v0 --num_envs {num_envs}")
    print()

    return True


def train_phase2(robot: str, config: dict, checkpoint: str, num_envs: int = 4096):
    """Train adaptation module to match encoder.

    In this phase:
    - Base policy and encoder are frozen
    - Collect observation histories during rollouts
    - Train adaptation module: obs_history → latent
    - Supervised by encoder output
    """
    print("=" * 60)
    print(f"RMA PHASE 2: Training Adaptation Module for {robot.upper()}")
    print("=" * 60)
    print()
    print("This phase trains the adaptation module using supervised learning.")
    print("Target: match the encoder's latent output from observation history.")
    print()

    if not checkpoint:
        print("ERROR: Phase 2 requires a checkpoint from Phase 1!")
        print("Usage: python rma/train_rma.py --robot go2 --phase 2 --checkpoint <path>")
        return False

    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        return False

    phase2_cfg = config["phase2"]
    print(f"Configuration:")
    print(f"  - Checkpoint: {checkpoint}")
    print(f"  - Max iterations: {phase2_cfg['max_iterations']}")
    print(f"  - Learning rate: {phase2_cfg['learning_rate']}")
    print(f"  - Freeze policy: {phase2_cfg['freeze_policy']}")
    print()

    # TODO: Implement Phase 2 training
    # This would involve:
    # 1. Loading Phase 1 checkpoint
    # 2. Freezing encoder and base policy
    # 3. Rolling out to collect (obs_history, privileged_info) pairs
    # 4. Training adaptation module with MSE loss

    print("Phase 2 training would:")
    print("  1. Load the Phase 1 policy")
    print("  2. Freeze encoder and base policy weights")
    print("  3. Collect rollouts with observation history")
    print("  4. Train adaptation module to predict encoder output")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(description="RMA Training")
    parser.add_argument("--robot", type=str, default="go2", choices=["go2", "cassie"])
    parser.add_argument("--phase", type=str, default="1", choices=["1", "2", "both"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Phase 1 checkpoint for Phase 2")
    parser.add_argument("--num_envs", type=int, default=4096)
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.robot)
    print(f"Loaded config for {args.robot}")

    if args.phase == "1" or args.phase == "both":
        success = train_phase1(args.robot, config, args.num_envs)
        if not success:
            return

    if args.phase == "2" or args.phase == "both":
        checkpoint = args.checkpoint
        if args.phase == "both":
            # Use default Phase 1 checkpoint location
            checkpoint = f"logs/rsl_rl/{config['phase1']['experiment_name']}/model_{config['phase1']['max_iterations']}.pt"

        success = train_phase2(args.robot, config, checkpoint, args.num_envs)
        if not success:
            return

    print("=" * 60)
    print("TRAINING SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
