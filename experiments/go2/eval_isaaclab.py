"""Evaluate a Go2 policy in Isaac Lab across multiple scenarios.

Saves results in the SAME format as eval_mujoco.py (eval_results.json +
eval_data.npz) so that compare_dr.py can overlay both for sim2sim comparison.

Usage:
    ~/IsaacLab/isaaclab.sh -p experiments/go2/eval_isaaclab.py \
        --task Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0 \
        --num_envs 1 --headless \
        --experiment directions --episodes 5 --max_steps 500 \
        --output_dir experiments/go2/results/isaaclab_baseline \
        --label isaaclab_baseline
"""

"""Launch Isaac Sim first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# Import cli_args early (no Omniverse dependency) for add_rsl_rl_args
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(_project_root, "scripts", "rsl_rl"))
import cli_args  # noqa: E402

# ── CLI (parsed before AppLauncher to grab custom flags) ────────────────────
parser = argparse.ArgumentParser(description="Evaluate Go2 policy in Isaac Lab")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--experiment", type=str, default="directions",
                    choices=["directions", "speeds", "quick"])
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=500)
parser.add_argument("--output_dir", type=str,
                    default="experiments/go2/results/isaaclab_baseline")
parser.add_argument("--label", type=str, default="isaaclab_baseline")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # force headless for evaluation

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows after Omniverse init."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import Go2 extension (cli_args and _project_root already set above)
sys.path.insert(0, os.path.join(_project_root, "source", "go2"))
try:
    import go2  # noqa: F401
except ImportError:
    pass


# ── Experiment configs (same as eval_mujoco.py) ─────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    commands: List[float]
    description: str

@dataclass
class EpisodeData:
    steps: int = 0
    body_velocities: List[List[float]] = field(default_factory=list)
    angular_velocities: List[List[float]] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    joint_positions: List[List[float]] = field(default_factory=list)
    actions: List[List[float]] = field(default_factory=list)
    torques: List[List[float]] = field(default_factory=list)

EXPERIMENTS: Dict[str, List[ExperimentConfig]] = {
    "directions": [
        ExperimentConfig("Forward",    [0.5, 0.0,  0.0], "Standard forward walking"),
        ExperimentConfig("Backward",   [-0.5, 0.0, 0.0], "Reverse walking"),
        ExperimentConfig("Left",       [0.0, 0.5,  0.0], "Lateral left"),
        ExperimentConfig("Right",      [0.0, -0.5, 0.0], "Lateral right"),
        ExperimentConfig("Turn_Left",  [0.0, 0.0,  0.5], "Rotate left"),
        ExperimentConfig("Turn_Right", [0.0, 0.0, -0.5], "Rotate right"),
        ExperimentConfig("Diagonal",   [0.5, 0.5,  0.0], "Diagonal forward-left"),
        ExperimentConfig("Circle",     [0.5, 0.0,  0.3], "Forward with turning"),
    ],
    "speeds": [
        ExperimentConfig("Slow",   [0.25, 0.0, 0.0], "Slow forward"),
        ExperimentConfig("Normal", [0.5,  0.0, 0.0], "Normal forward"),
        ExperimentConfig("Fast",   [1.0,  0.0, 0.0], "Fast forward"),
        ExperimentConfig("Sprint", [1.5,  0.0, 0.0], "Maximum forward"),
    ],
    "quick": [
        ExperimentConfig("Forward", [0.5, 0.0, 0.0], "Quick sanity test"),
    ],
}


# ── Metrics (same logic as eval_mujoco.py) ───────────────────────────────────

def compute_metrics(episodes, commands, max_steps):
    success_threshold = int(max_steps * 0.8)
    n_success = sum(1 for ep in episodes if ep.steps >= success_threshold)
    survival_steps = [ep.steps for ep in episodes]

    vx_sq, vy_sq, wz_sq = [], [], []
    all_heights, action_smoothness, torque_magnitudes = [], [], []

    for ep in episodes:
        if not ep.body_velocities:
            continue
        vels = np.array(ep.body_velocities)
        ang = np.array(ep.angular_velocities)
        vx_sq.extend(((vels[:, 0] - commands[0]) ** 2).tolist())
        vy_sq.extend(((vels[:, 1] - commands[1]) ** 2).tolist())
        wz_sq.extend(((ang[:, 2] - commands[2]) ** 2).tolist())
        all_heights.extend(ep.heights)
        acts = np.array(ep.actions)
        if len(acts) > 1:
            diffs = np.diff(acts, axis=0)
            action_smoothness.extend(np.sqrt(np.sum(diffs ** 2, axis=1)).tolist())
        if ep.torques:
            torque_magnitudes.extend([float(np.linalg.norm(t)) for t in ep.torques])

    rmse_vx = float(np.sqrt(np.mean(vx_sq))) if vx_sq else 0.0
    rmse_vy = float(np.sqrt(np.mean(vy_sq))) if vy_sq else 0.0
    rmse_wz = float(np.sqrt(np.mean(wz_sq))) if wz_sq else 0.0
    overall_rmse = float(np.sqrt(np.mean(vx_sq + vy_sq + wz_sq))) if vx_sq else 0.0

    return {
        "success_rate": n_success / len(episodes) * 100,
        "mean_survival_steps": float(np.mean(survival_steps)),
        "std_survival_steps": float(np.std(survival_steps)),
        "rmse_vx": rmse_vx, "rmse_vy": rmse_vy, "rmse_wz": rmse_wz,
        "overall_rmse": overall_rmse,
        "mean_height": float(np.mean(all_heights)) if all_heights else 0.0,
        "std_height": float(np.std(all_heights)) if all_heights else 0.0,
        "mean_action_smoothness": float(np.mean(action_smoothness)) if action_smoothness else 0.0,
        "mean_torque_magnitude": float(np.mean(torque_magnitudes)) if torque_magnitudes else 0.0,
    }


# ── Save helpers ─────────────────────────────────────────────────────────────

def _convert_numpy(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    return obj


def save_results(results, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics_only = {
        "label": results["label"],
        "experiment": results["experiment"],
        "scenarios": {},
    }
    for name, sc_data in results["scenarios"].items():
        metrics_only["scenarios"][name] = {
            "config": sc_data["config"],
            "metrics": sc_data["metrics"],
        }

    with open(out / "eval_results.json", "w") as f:
        json.dump(_convert_numpy(metrics_only), f, indent=2)

    npz_data = {}
    for name, sc_data in results["scenarios"].items():
        best = sc_data["best_episode"]
        prefix = name.lower().replace(" ", "_")
        npz_data[f"{prefix}_body_vel"] = np.array(best["body_velocities"])
        npz_data[f"{prefix}_ang_vel"] = np.array(best["angular_velocities"])
        npz_data[f"{prefix}_heights"] = np.array(best["heights"])
        npz_data[f"{prefix}_joint_pos"] = np.array(best["joint_positions"])
        npz_data[f"{prefix}_actions"] = np.array(best["actions"])
        npz_data[f"{prefix}_torques"] = np.array(best["torques"])
        npz_data[f"{prefix}_steps"] = np.array(best["steps"])
        npz_data[f"{prefix}_commands"] = np.array(sc_data["config"]["commands"])

    np.savez_compressed(out / "eval_data.npz", **npz_data)
    print(f"\nSaved: {out / 'eval_results.json'}")
    print(f"Saved: {out / 'eval_data.npz'}")


# ── Main evaluation loop ────────────────────────────────────────────────────

def main():
    # Parse env config
    env_cfg = parse_env_cfg(
        args_cli.task, device="cuda:0",
        num_envs=args_cli.num_envs, use_fabric=True,
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Discover checkpoint
    robot_name = agent_cfg.experiment_name.split("_")[0]
    log_root_path = os.path.abspath(
        os.path.join("logs", robot_name, "rsl_rl", agent_cfg.experiment_name)
    )
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[eval_isaaclab] Task:       {args_cli.task}")
    print(f"[eval_isaaclab] Checkpoint: {resume_path}")
    print(f"[eval_isaaclab] Experiment: {args_cli.experiment}")
    print(f"[eval_isaaclab] Episodes:   {args_cli.episodes}")
    print(f"[eval_isaaclab] Max steps:  {args_cli.max_steps}")

    # Create env + load policy
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped

    scenarios = EXPERIMENTS[args_cli.experiment]
    results = {"label": args_cli.label, "experiment": args_cli.experiment, "scenarios": {}}

    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {sc.name}  cmd=[{sc.commands[0]:.2f}, {sc.commands[1]:.2f}, {sc.commands[2]:.2f}]")
        print(f"{'='*60}")

        cmd_tensor = torch.tensor(
            [sc.commands], dtype=torch.float32, device=base_env.device
        ).repeat(base_env.num_envs, 1)

        ep_list = []

        for ep_idx in range(args_cli.episodes):
            # Reset env
            obs, _ = env.get_observations()
            episode = EpisodeData()

            for step in range(args_cli.max_steps):
                # Override commands each step
                cmd_term = base_env.command_manager.get_term("base_velocity")
                cmd_term.vel_command_b[:] = cmd_tensor

                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)

                # Record from env 0
                robot = base_env.scene["robot"]
                lin_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
                ang_vel = robot.data.root_ang_vel_b[0].cpu().numpy()
                height = float(robot.data.root_pos_w[0, 2].cpu())
                jpos = robot.data.joint_pos[0].cpu().numpy()
                act = actions[0].cpu().numpy()
                torque = robot.data.applied_torque[0].cpu().numpy() if hasattr(robot.data, 'applied_torque') else act

                episode.body_velocities.append(lin_vel.tolist())
                episode.angular_velocities.append(ang_vel.tolist())
                episode.heights.append(height)
                episode.joint_positions.append(jpos.tolist())
                episode.actions.append(act.tolist())
                episode.torques.append(torque.tolist())
                episode.steps += 1

                if height < 0.15:
                    break

            ep_list.append(episode)
            status = "OK" if episode.steps >= int(args_cli.max_steps * 0.8) else "FALL"
            print(f"  Episode {ep_idx+1}/{args_cli.episodes}: steps={episode.steps} [{status}]")

        metrics = compute_metrics(ep_list, sc.commands, args_cli.max_steps)
        print(f"  -> success={metrics['success_rate']:.0f}%  "
              f"RMSE_vx={metrics['rmse_vx']:.3f}  "
              f"RMSE_vy={metrics['rmse_vy']:.3f}  "
              f"RMSE_wz={metrics['rmse_wz']:.3f}")

        best_ep = max(ep_list, key=lambda e: e.steps)
        results["scenarios"][sc.name] = {
            "config": asdict(sc),
            "metrics": metrics,
            "best_episode": asdict(best_ep),
        }

    # Resolve output dir
    output_dir = Path(args_cli.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(_project_root) / output_dir
    save_results(results, str(output_dir))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Scenario':<15} {'Success':>8} {'RMSE_vx':>8} {'RMSE_vy':>8} {'RMSE_wz':>8} {'Steps':>8}")
    print(f"  {'-'*55}")
    for name, sc in results["scenarios"].items():
        m = sc["metrics"]
        print(f"  {name:<15} {m['success_rate']:>7.0f}% "
              f"{m['rmse_vx']:>8.3f} {m['rmse_vy']:>8.3f} "
              f"{m['rmse_wz']:>8.3f} {m['mean_survival_steps']:>8.0f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
