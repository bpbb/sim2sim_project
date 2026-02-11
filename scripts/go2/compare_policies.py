#!/usr/bin/env python3
"""Compare Go2 policies across DR levels and terrains in MuJoCo.

Evaluates policies on both flat and rough terrain to measure robustness
and terrain adaptation. Generates comprehensive comparison plots and metrics.

Usage:
    python scripts/go2/compare_policies.py
    python scripts/go2/compare_policies.py --trials 5 --duration 30
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from deploy_mujoco import (
    MUJOCO_TO_ISAACLAB, ISAACLAB_TO_MUJOCO,
    quat_rotate_inverse, get_gravity_orientation, pd_control
)


# ── Policy and Terrain Configuration ────────────────────────────────────────

POLICIES = {
    "baseline": "logs/go2/policies/baseline.pt",
    "moderate": "logs/go2/policies/moderate.pt",
    "aggressive": "logs/go2/policies/aggressive.pt",
    "rough_baseline": "logs/go2/policies/rough_baseline.pt",
    "rough_moderate": "logs/go2/policies/rough_moderate.pt",
    "actuator_net": "logs/go2/policies/actuator_net.pt",
}

TERRAINS = {
    "flat": "unitree_mujoco/unitree_robots/go2/scene_flat.xml",
    "rough": "unitree_mujoco/unitree_robots/go2/scene_isaaclab_terrain.xml",
}

COLORS = {
    "baseline": "#3498db",     # blue
    "moderate": "#f39c12",     # orange
    "aggressive": "#2ecc71",   # green
    "actuator_net": "#9b59b6", # purple
}


# ── Single Policy Evaluation ────────────────────────────────────────────────

def run_trial(
    policy: torch.nn.Module,
    m: mujoco.MjModel,
    d: mujoco.MjData,
    config: Dict,
    max_duration: float = 20.0,
    verbose: bool = False,
) -> Dict:
    """Run a single evaluation trial for a policy on a terrain.

    Returns:
        dict with keys: max_dist, survival_time, fell, vx_error, vy_error, ...
    """
    # Config parameters
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    default_angles_isaaclab = default_angles[MUJOCO_TO_ISAACLAB]

    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    cmd = np.array(config["cmd_init"], dtype=np.float32)
    lin_vel_scale = config.get("lin_vel_scale", 2.0)

    start_pos = np.array(config.get("base_init_pos", [0.0, 0, 0.42]))

    # Reset state
    d.qpos[0:3] = start_pos
    d.qpos[3:7] = config.get("base_init_quat", [1, 0, 0, 0])
    d.qpos[7:] = default_angles
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(config["num_obs"], dtype=np.float32)

    sim_time = 0.0
    max_dist = 0.0
    fell = False
    completed = False

    # Logging arrays
    times, vxs, vys, yaws, torques = [], [], [], [], []
    xs, ys, zs, gravs = [], [], [], []

    while sim_time < max_duration:
        # PD control
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        # Ctrl swap: actuator order (FR,FL,RR,RL) != joint order (FL,FR,RL,RR)
        d.ctrl[0:3] = tau[3:6]   # FR
        d.ctrl[3:6] = tau[0:3]   # FL
        d.ctrl[6:9] = tau[9:12]  # RR
        d.ctrl[9:12] = tau[6:9]  # RL

        mujoco.mj_step(m, d)
        sim_time += m.opt.timestep

        # Observation update (at control decimation rate)
        if int(sim_time / m.opt.timestep) % config["control_decimation"] == 0:
            # Body frame velocities
            base_lin_vel = quat_rotate_inverse(d.qpos[3:7], d.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(d.qpos[3:7], d.qvel[3:6])
            grav = get_gravity_orientation(d.qpos[3:7])

            # Log state
            times.append(sim_time)
            vxs.append(base_lin_vel[0])
            vys.append(base_lin_vel[1])
            yaws.append(base_ang_vel[2])
            torques.append(np.linalg.norm(tau))
            xs.append(d.qpos[0])
            ys.append(d.qpos[1])
            zs.append(d.qpos[2])
            gravs.append(grav)

            # Terrain boundary check (hfield x-bound)
            if hasattr(m, "hfield_size") and len(m.hfield_size) > 0:
                hfield_x_bound = m.hfield_size[0, 0]
                if d.qpos[0] >= hfield_x_bound - 0.1:
                    completed = True
                    break

            # Fall detection
            if d.qpos[2] < 0.18:  # base too low
                fell = True
                break
            if grav[2] > 0.1:  # tilted past horizontal
                fell = True
                break

            # Update max displacement (Euclidean from start)
            displacement = np.linalg.norm(d.qpos[0:2] - start_pos[0:2])
            max_dist = max(max_dist, displacement)

            # Build observation
            qj_isaaclab = d.qpos[7:][MUJOCO_TO_ISAACLAB]
            dqj_isaaclab = d.qvel[6:][MUJOCO_TO_ISAACLAB]

            obs[0:3] = base_lin_vel * lin_vel_scale
            obs[3:6] = base_ang_vel * ang_vel_scale
            obs[6:9] = grav
            obs[9:12] = cmd * cmd_scale
            obs[12:24] = (qj_isaaclab - default_angles_isaaclab) * dof_pos_scale
            obs[24:36] = dqj_isaaclab * dof_vel_scale
            obs[36:48] = action

            # Policy inference
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
            with torch.no_grad():
                action = policy(obs_tensor).detach().numpy().squeeze()

            # Reorder action: Isaac Lab -> MuJoCo
            target_dof_pos = action[ISAACLAB_TO_MUJOCO] * action_scale + default_angles

    # Compute metrics
    vxs, vys, yaws = np.array(vxs), np.array(vys), np.array(yaws)
    vx_rmse = np.sqrt(np.mean((vxs - cmd[0])**2)) if len(vxs) > 0 else 0
    vy_rmse = np.sqrt(np.mean((vys - cmd[1])**2)) if len(vys) > 0 else 0
    yaw_rmse = np.sqrt(np.mean((yaws - cmd[2])**2)) if len(yaws) > 0 else 0
    avg_torque = np.mean(torques) if len(torques) > 0 else 0
    z_var = np.var(zs) if len(zs) > 0 else 0

    # Tilt metric: XY component of gravity vector
    gravs = np.array(gravs)
    avg_tilt = np.mean(np.linalg.norm(gravs[:, :2], axis=1)) if len(gravs) > 0 else 0

    return {
        "max_dist": max_dist,
        "survival_time": sim_time,
        "fell": fell,
        "completed": completed,
        "vx_rmse": vx_rmse,
        "vy_rmse": vy_rmse,
        "yaw_rmse": yaw_rmse,
        "avg_torque": avg_torque,
        "z_var": z_var,
        "avg_tilt": avg_tilt,
        "log": {
            "t": np.array(times),
            "vx": vxs,
            "vy": vys,
            "yaw": yaws,
            "x": np.array(xs),
            "y": np.array(ys),
            "z": np.array(zs),
            "cmd": cmd,
        }
    }


def evaluate_policy(
    policy_name: str,
    policy_path: str,
    terrain_name: str,
    terrain_path: str,
    config: Dict,
    num_trials: int = 3,
    max_duration: float = 20.0,
) -> List[Dict]:
    """Evaluate a policy on a terrain for multiple trials."""
    # Load policy
    policy = torch.jit.load(policy_path)
    policy.eval()

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(terrain_path)
    d = mujoco.MjData(m)
    m.opt.timestep = config["simulation_dt"]

    results = []
    for trial_idx in range(num_trials):
        result = run_trial(policy, m, d, config, max_duration)
        results.append(result)

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(all_results: Dict, output_dir: Path):
    """Generate comprehensive comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Plot 1: Success Rate by Terrain ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    policies = list(all_results.keys())
    terrains = list(all_results[policies[0]].keys())
    x = np.arange(len(terrains))
    width = 0.8 / len(policies)

    for i, pol in enumerate(policies):
        success_rates = []
        for ter in terrains:
            trials = all_results[pol][ter]
            n_success = sum(1 for t in trials if not t["fell"])
            success_rates.append(n_success / len(trials) * 100)

        bars = ax.bar(x + i * width, success_rates, width, label=pol.capitalize(),
                      color=COLORS.get(pol, "#95a5a6"), edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width * len(policies) / 2)
    ax.set_xticklabels([t.capitalize() for t in terrains])
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Success Rate by Terrain and DR Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_comparison.png", dpi=150)
    plt.close()
    print(f"  [1/4] Saved: {output_dir / 'success_rate_comparison.png'}")

    # ── Plot 2: Average Distance Traveled ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, pol in enumerate(policies):
        avg_dists = []
        for ter in terrains:
            trials = all_results[pol][ter]
            avg_dists.append(np.mean([t["max_dist"] for t in trials]))

        bars = ax.bar(x + i * width, avg_dists, width, label=pol.capitalize(),
                      color=COLORS.get(pol, "#95a5a6"), edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, avg_dists):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f"{val:.2f}m", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * len(policies) / 2)
    ax.set_xticklabels([t.capitalize() for t in terrains])
    ax.set_ylabel("Average Distance Traveled (m)")
    ax.set_title("Distance Traveled by Terrain and DR Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "distance_comparison.png", dpi=150)
    plt.close()
    print(f"  [2/4] Saved: {output_dir / 'distance_comparison.png'}")

    # ── Plot 3: Velocity Tracking Error (RMSE) ───────────────────────────────
    fig, axes = plt.subplots(1, len(terrains), figsize=(12, 5), sharey=True)
    if len(terrains) == 1:
        axes = [axes]

    for ter_idx, ter in enumerate(terrains):
        ax = axes[ter_idx]
        x_comp = np.arange(3)
        width_comp = 0.8 / len(policies)

        for pol_idx, pol in enumerate(policies):
            trials = all_results[pol][ter]
            vx_rmse = np.mean([t["vx_rmse"] for t in trials])
            vy_rmse = np.mean([t["vy_rmse"] for t in trials])
            yaw_rmse = np.mean([t["yaw_rmse"] for t in trials])

            ax.bar(x_comp + pol_idx * width_comp, [vx_rmse, vy_rmse, yaw_rmse],
                   width_comp, label=pol.capitalize() if ter_idx == 0 else "",
                   color=COLORS.get(pol, "#95a5a6"), edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_comp + width_comp * len(policies) / 2)
        ax.set_xticklabels(["vx", "vy", "wz"])
        ax.set_title(f"{ter.capitalize()} Terrain", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        if ter_idx == 0:
            ax.set_ylabel("RMSE")
            ax.legend()

    plt.suptitle("Velocity Tracking Error by Component", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "tracking_error_comparison.png", dpi=150)
    plt.close()
    print(f"  [3/4] Saved: {output_dir / 'tracking_error_comparison.png'}")

    # ── Plot 4: Velocity Timeseries (best trial per policy/terrain) ──────────
    n_ter = len(terrains)
    n_pol = len(policies)
    fig, axes = plt.subplots(n_ter, n_pol, figsize=(4 * n_pol, 3.5 * n_ter),
                              sharex=True, squeeze=False)

    for ter_idx, ter in enumerate(terrains):
        for pol_idx, pol in enumerate(policies):
            ax = axes[ter_idx, pol_idx]
            trials = all_results[pol][ter]
            # Pick trial with longest survival time
            best_trial = max(trials, key=lambda t: t["survival_time"])
            log = best_trial["log"]

            ax.plot(log["t"], log["vx"], color="#e74c3c", linewidth=1.5, label="vx")
            ax.plot(log["t"], log["vy"], color="#3498db", linewidth=1.5, label="vy")
            ax.plot(log["t"], log["yaw"], color="#2ecc71", linewidth=1.5, label="wz")

            cmd = log["cmd"]
            ax.axhline(cmd[0], color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.4)
            ax.axhline(cmd[1], color="#3498db", linestyle="--", linewidth=1.0, alpha=0.4)
            ax.axhline(cmd[2], color="#2ecc71", linestyle="--", linewidth=1.0, alpha=0.4)

            ax.grid(alpha=0.3)
            if ter_idx == 0:
                ax.set_title(pol.replace("_", " ").capitalize(), fontsize=11, fontweight="bold")
            if pol_idx == 0:
                ax.set_ylabel(f"{ter.capitalize()}", fontsize=10)
            if ter_idx == n_ter - 1:
                ax.set_xlabel("Time (s)")
            if ter_idx == 0 and pol_idx == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Velocity Tracking — MuJoCo (best trial per policy/terrain)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_timeseries.png", dpi=150)
    plt.close()
    print(f"  [4/4] Saved: {output_dir / 'velocity_timeseries.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def print_summary_table(all_results: Dict):
    """Print formatted summary table."""
    print("\n" + "="*120)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*120)

    policies = list(all_results.keys())
    terrains = list(all_results[policies[0]].keys())

    for ter in terrains:
        print(f"\n{ter.upper()} TERRAIN:")
        print("-" * 120)
        print(f"{'Policy':<15} | {'Trials':<7} | {'Success':<8} | {'Avg Dist':<9} | "
              f"{'Avg Time':<9} | {'Vx RMSE':<9} | {'Z-Var':<10} | {'Tilt':<8}")
        print("-" * 120)

        for pol in policies:
            trials = all_results[pol][ter]
            n_trials = len(trials)
            n_success = sum(1 for t in trials if not t["fell"])
            success_rate = n_success / n_trials * 100

            avg_dist = np.mean([t["max_dist"] for t in trials])
            avg_time = np.mean([t["survival_time"] for t in trials])
            avg_vx_rmse = np.mean([t["vx_rmse"] for t in trials])
            avg_z_var = np.mean([t["z_var"] for t in trials])
            avg_tilt = np.mean([t["avg_tilt"] for t in trials])

            print(f"{pol:<15} | {n_trials:<7} | {success_rate:>6.0f}% | {avg_dist:>8.2f}m | "
                  f"{avg_time:>8.2f}s | {avg_vx_rmse:>8.3f} | {avg_z_var:>9.5f} | {avg_tilt:>7.3f}")

    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Go2 policies across DR levels and terrains",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per policy/terrain")
    parser.add_argument("--duration", type=float, default=20.0, help="Max duration per trial (seconds)")
    parser.add_argument("--config", type=str, default="scripts/go2/go2_isaaclab.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, default="scripts/go2/comparison_results",
                        help="Output directory for plots")
    parser.add_argument("--policies", type=str, nargs="+", default=None,
                        help="Policies to evaluate (default: all)")
    parser.add_argument("--terrains", type=str, nargs="+", default=None,
                        help="Terrains to evaluate (default: all)")
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / args.config if not Path(args.config).is_absolute() else Path(args.config)
    output_dir = project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    # Load config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Filter policies/terrains
    policies_to_eval = args.policies if args.policies else list(POLICIES.keys())
    terrains_to_eval = args.terrains if args.terrains else list(TERRAINS.keys())

    print("\n" + "="*80)
    print("Go2 DR Policy Comparison — MuJoCo Evaluation")
    print("="*80)
    print(f"  Policies:  {', '.join(policies_to_eval)}")
    print(f"  Terrains:  {', '.join(terrains_to_eval)}")
    print(f"  Trials:    {args.trials} per policy/terrain")
    print(f"  Duration:  {args.duration}s per trial")
    print(f"  Output:    {output_dir}")
    print("="*80 + "\n")

    all_results = {}

    for pol_name in policies_to_eval:
        if pol_name not in POLICIES:
            print(f"[SKIP] Unknown policy: {pol_name}")
            continue

        pol_path = project_root / POLICIES[pol_name]
        if not pol_path.exists():
            print(f"[SKIP] Policy not found: {pol_path}")
            continue

        all_results[pol_name] = {}

        for ter_name in terrains_to_eval:
            if ter_name not in TERRAINS:
                print(f"[SKIP] Unknown terrain: {ter_name}")
                continue

            ter_path = project_root / TERRAINS[ter_name]
            if not ter_path.exists():
                print(f"[SKIP] Terrain not found: {ter_path}")
                continue

            print(f"Evaluating {pol_name.upper()} on {ter_name.upper()}...")
            results = evaluate_policy(
                pol_name, str(pol_path), ter_name, str(ter_path),
                config, args.trials, args.duration
            )
            all_results[pol_name][ter_name] = results

            # Quick progress summary
            n_success = sum(1 for r in results if not r["fell"])
            avg_dist = np.mean([r["max_dist"] for r in results])
            print(f"  -> Success: {n_success}/{args.trials}, Avg dist: {avg_dist:.2f}m\n")

    # Print summary table
    print_summary_table(all_results)

    # Generate plots
    print("Generating comparison plots...")
    plot_comparison(all_results, output_dir)

    print(f"\n✓ All done! Results saved to: {output_dir}/\n")


if __name__ == "__main__":
    main()
