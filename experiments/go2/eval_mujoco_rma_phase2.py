#!/usr/bin/env python3
"""Evaluate RMA Phase 2 (with adaptation module) in MuJoCo.

This script deploys the complete RMA pipeline:
1. RMA Phase 1 policy (52-dim input: 48 obs + 4 latent)
2. Adaptation module (predicts 4-dim latent from 50-step obs history)
"""

import argparse
import collections
import json
import mujoco
import numpy as np
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

# Import from eval_mujoco
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_mujoco import (
    ExperimentConfig,
    EpisodeData,
    quat_rotate_inverse,
    get_gravity_orientation,
    pd_control,
    compute_metrics,
    MUJOCO_TO_ISAACLAB,
    ISAACLAB_TO_MUJOCO,
    EXPERIMENTS,
)


def run_episode_rma(
    model: mujoco.MjModel,
    policy: torch.jit.ScriptModule,
    adaptation_module: torch.nn.Module,
    config: dict,
    commands: List[float],
    max_steps: int,
    default_angles: np.ndarray,
    default_angles_isaaclab: np.ndarray,
    history_length: int = 50,
) -> EpisodeData:
    """Run one episode with RMA Phase 2 (policy + adaptation module)."""
    data = mujoco.MjData(model)

    # Fixed initial state
    data.qpos[0:3] = [0, 0, 0.35]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = default_angles
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # Config
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    action_scale = config["action_scale"]
    lin_vel_scale = config.get("lin_vel_scale", 2.0)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    control_decimation = config["control_decimation"]

    cmd = np.array(commands, dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()

    # Observation history buffer for adaptation module
    obs_history = collections.deque(maxlen=history_length)
    episode = EpisodeData()
    counter = 0

    for step in range(max_steps * control_decimation):
        # PD control
        tau = pd_control(target_dof_pos, data.qpos[7:], kps,
                        np.zeros_like(kds), data.qvel[6:], kds)

        # Ctrl swap
        data.ctrl[0:3] = tau[3:6]
        data.ctrl[3:6] = tau[0:3]
        data.ctrl[6:9] = tau[9:12]
        data.ctrl[9:12] = tau[6:9]

        mujoco.mj_step(model, data)

        counter += 1
        if counter % control_decimation != 0:
            continue

        # ── Record at control frequency ──
        qj = data.qpos[7:]
        dqj = data.qvel[6:]
        quat = data.qpos[3:7]
        height = data.qpos[2]

        # Body-frame velocities
        base_lin_vel = quat_rotate_inverse(quat, data.qvel[0:3])
        base_ang_vel = quat_rotate_inverse(quat, data.qvel[3:6])
        gravity_orientation = get_gravity_orientation(quat)

        # Reorder to Isaac Lab
        qj_il = qj[MUJOCO_TO_ISAACLAB]
        dqj_il = dqj[MUJOCO_TO_ISAACLAB]

        # Build 48-dim observation (standard locomotion obs)
        obs_48 = np.concatenate([
            base_lin_vel * lin_vel_scale,
            base_ang_vel * ang_vel_scale,
            gravity_orientation,
            cmd * cmd_scale,
            (qj_il - default_angles_isaaclab) * dof_pos_scale,
            dqj_il * dof_vel_scale,
            action,
        ])

        # Add to history buffer
        obs_history.append(obs_48)

        # Adaptation module: predict 4-dim latent from history
        if len(obs_history) == history_length:
            obs_tensor = torch.stack([torch.from_numpy(o) for o in obs_history]).unsqueeze(0).float()  # (1, 50, 48)
            # Move to same device as adaptation module
            device = next(adaptation_module.parameters()).device
            obs_tensor = obs_tensor.to(device)
            with torch.no_grad():
                z_hat = adaptation_module(obs_tensor).squeeze().cpu().numpy()  # (4,)
        else:
            # During warmup (first 50 steps), use neutral latent
            z_hat = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)

        # Build 52-dim input for policy (48 obs + 4 latent)
        obs_52 = np.concatenate([obs_48, z_hat])
        obs_tensor = torch.from_numpy(obs_52).unsqueeze(0).float()

        with torch.no_grad():
            action = policy(obs_tensor).detach().numpy().squeeze()

        action_mujoco = action[ISAACLAB_TO_MUJOCO]
        target_dof_pos = action_mujoco * action_scale + default_angles

        # Store
        episode.body_velocities.append(base_lin_vel.tolist())
        episode.angular_velocities.append(base_ang_vel.tolist())
        episode.heights.append(float(height))
        episode.joint_positions.append(qj.tolist())
        episode.actions.append(action.tolist())
        episode.torques.append(tau.tolist())
        episode.steps += 1

        # Terminate on fall
        if height < 0.15:
            break

    return episode


def evaluate_policy_rma(
    policy_path: str,
    adaptation_path: str,
    config: dict,
    xml_path: str,
    experiment_set: str,
    episodes: int,
    max_steps: int,
    label: str,
    history_length: int = 50,
) -> Dict[str, Any]:
    """Evaluate RMA Phase 2 policy with adaptation module."""
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = config["simulation_dt"]

    # Load RMA Phase 1 policy
    policy = torch.jit.load(policy_path)
    policy.eval()

    # Load adaptation module
    try:
        adaptation_module = torch.jit.load(adaptation_path)
    except Exception:
        # Fallback: load as regular checkpoint
        from train_adaptation_module import AdaptationModuleCNN
        adaptation_module = AdaptationModuleCNN(obs_dim=48, history_len=history_length, latent_dim=4)
        adaptation_module.load_state_dict(torch.load(adaptation_path))
    adaptation_module.eval()

    print(f"✅ Loaded RMA Phase 1 policy: {policy_path}")
    print(f"✅ Loaded adaptation module: {adaptation_path}")

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    default_angles_il = default_angles[MUJOCO_TO_ISAACLAB]

    # Get scenarios
    scenarios = EXPERIMENTS[experiment_set]

    results = {
        "label": label,
        "experiment_set": experiment_set,
        "episodes_per_scenario": episodes,
        "max_steps": max_steps,
        "scenarios": {},
    }

    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {sc.name}  cmd=[{sc.commands[0]:.2f}, {sc.commands[1]:.2f}, {sc.commands[2]:.2f}]")
        print(f"{'='*60}")

        ep_list: List[EpisodeData] = []
        for ep_idx in range(episodes):
            ep = run_episode_rma(model, policy, adaptation_module, config, sc.commands,
                                max_steps, default_angles, default_angles_il,
                                history_length=history_length)
            ep_list.append(ep)
            status = "OK" if ep.steps >= int(max_steps * 0.8) else "FALL"
            print(f"  Episode {ep_idx+1}/{episodes}: steps={ep.steps} [{status}]")

        metrics = compute_metrics(ep_list, sc.commands, max_steps)
        print(f"  -> success={metrics['success_rate']:.0f}%  "
              f"RMSE_vx={metrics['rmse_vx']:.3f}  "
              f"RMSE_vy={metrics['rmse_vy']:.3f}  "
              f"RMSE_wz={metrics['rmse_wz']:.3f}")

        # Find best episode
        best_ep = max(ep_list, key=lambda e: e.steps)

        results["scenarios"][sc.name] = {
            "config": asdict(sc),
            "metrics": metrics,
            "best_episode": asdict(best_ep),
        }

    return results


def save_results(results: Dict, output_dir: str):
    """Save evaluation results."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save NPZ (best episodes)
    npz_data = {}
    for sc_name, sc_data in results["scenarios"].items():
        best_ep = sc_data["best_episode"]
        npz_data[f"{sc_name}_body_vel"] = np.array(best_ep["body_velocities"])
        npz_data[f"{sc_name}_ang_vel"] = np.array(best_ep["angular_velocities"])
        npz_data[f"{sc_name}_commands"] = np.array(sc_data["config"]["commands"])

    np.savez_compressed(out_dir / "eval_data.npz", **npz_data)

    print(f"\nMetrics saved: {out_dir / 'eval_results.json'}")
    print(f"Time-series saved: {out_dir / 'eval_data.npz'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RMA Phase 2 in MuJoCo")
    parser.add_argument("--policy", required=True, help="RMA Phase 1 policy.pt")
    parser.add_argument("--adaptation", required=True, help="Adaptation module .pt")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--experiment", default="directions", choices=["directions", "speeds", "all"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="RMA-Phase2")
    parser.add_argument("--history_length", type=int, default=50)
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = Path(args.config) if Path(args.config).is_absolute() else project_root / args.config
    policy_path = Path(args.policy) if Path(args.policy).is_absolute() else project_root / args.policy
    adaptation_path = Path(args.adaptation) if Path(args.adaptation).is_absolute() else project_root / args.adaptation
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else project_root / args.output_dir

    with open(config_path) as f:
        config = yaml.safe_load(f)

    xml_path = Path(config["xml_path"])
    if not xml_path.is_absolute():
        xml_path = project_root / xml_path

    print("="*60)
    print("RMA Phase 2 Evaluation (Policy + Adaptation Module)")
    print("="*60)
    print(f"  Policy:     {policy_path}")
    print(f"  Adaptation: {adaptation_path}")
    print(f"  Config:     {config_path}")
    print(f"  XML:        {xml_path}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Label:      {args.label}")
    print(f"  Output:     {output_dir}")
    print()

    # Determine experiment sets
    if args.experiment == "all":
        sets = ["directions", "speeds"]
    else:
        sets = [args.experiment]

    all_results = {}
    for exp_set in sets:
        results = evaluate_policy_rma(
            policy_path=str(policy_path),
            adaptation_path=str(adaptation_path),
            config=config,
            xml_path=str(xml_path),
            experiment_set=exp_set,
            episodes=args.episodes,
            max_steps=args.max_steps,
            label=args.label,
            history_length=args.history_length,
        )
        save_results(results, str(output_dir))
        all_results[exp_set] = results

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
