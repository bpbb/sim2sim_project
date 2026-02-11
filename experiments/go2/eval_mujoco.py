#!/usr/bin/env python3
"""Evaluate a Go2 policy in headless MuJoCo across multiple scenarios.

Self-contained script — no Isaac Lab dependency required. Reuses the simulation
loop from scripts/go2/deploy_mujoco.py (joint mapping, ctrl swap, obs
construction, PD control) and experiment patterns from the archived
run_experiments.py.

Supports two torque modes:
  1. PD control (default) — analytical Kp/Kd
  2. ActuatorNet — trained neural network for torque prediction

Usage:
    # PD control (baseline/moderate/aggressive)
    python experiments/go2/eval_mujoco.py \
        --policy logs/go2/policies/baseline.pt \
        --config scripts/go2/go2_isaaclab.yaml \
        --experiment directions --episodes 5 --max_steps 500 \
        --output_dir experiments/go2/results/baseline --label baseline

    # ActuatorNet
    python experiments/go2/eval_mujoco.py \
        --policy logs/go2/policies/actuator_net.pt \
        --config scripts/go2/go2_isaaclab.yaml \
        --actuator_net --experiment directions --episodes 5 --max_steps 500 \
        --output_dir experiments/go2/results/actuator_net --label actuator_net
"""

import argparse
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import mujoco
import numpy as np
import torch
import yaml

# ── Joint order mappings (identical to deploy_mujoco.py) ────────────────────
# MuJoCo:    [FL_h, FL_t, FL_c, FR_h, FR_t, FR_c, RL_h, RL_t, RL_c, RR_h, RR_t, RR_c]
# Isaac Lab: [FL_h, FR_h, RL_h, RR_h, FL_t, FR_t, RL_t, RR_t, FL_c, FR_c, RL_c, RR_c]
MUJOCO_TO_ISAACLAB = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
ISAACLAB_TO_MUJOCO = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]


# ── Utility functions (from deploy_mujoco.py) ───────────────────────────────

def quat_rotate_inverse(q, v):
    """Rotate vector by inverse of quaternion (world -> body frame)."""
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    t = 2.0 * np.cross(np.array([q_x, q_y, q_z]), v)
    return v - q_w * t + np.cross(np.array([q_x, q_y, q_z]), t)


def get_gravity_orientation(quaternion):
    """Get projected gravity vector from quaternion."""
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD control for joint torques."""
    return (target_q - q) * kp + (target_dq - dq) * kd


# ── ActuatorNet wrapper (from deploy_mujoco.py) ─────────────────────────────

class ActuatorNetWrapper:
    """Wrapper for using trained ActuatorNet in MuJoCo."""

    def __init__(self, model_path, scaler_path, num_joints=12):
        try:
            import joblib
            self.scaler = joblib.load(scaler_path)
        except ImportError:
            import pickle
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.num_joints = num_joints
        self.joint_onehot = np.eye(num_joints)
        print(f"[ActuatorNet] Loaded model: {model_path}")
        print(f"[ActuatorNet] Loaded scaler: {scaler_path}")

    def compute_torques(self, pos_desired, pos_current, velocity):
        """Predict torques: (pos_desired, pos_current, vel, joint_id) -> torque."""
        raw = np.stack([pos_desired, pos_current, velocity], axis=1)  # [12, 3]
        features = np.hstack([raw, self.joint_onehot])  # [12, 15]
        features_scaled = self.scaler.transform(features)
        with torch.no_grad():
            torques = self.model(torch.FloatTensor(features_scaled)).numpy()
        return torques.squeeze()


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Configuration for a single test scenario."""
    name: str
    commands: List[float]  # [vx, vy, wz]
    description: str


@dataclass
class EpisodeData:
    """Per-episode time-series data collected during evaluation."""
    steps: int = 0
    body_velocities: List[List[float]] = field(default_factory=list)   # body-frame [vx, vy, vz]
    angular_velocities: List[List[float]] = field(default_factory=list) # body-frame [wx, wy, wz]
    heights: List[float] = field(default_factory=list)
    joint_positions: List[List[float]] = field(default_factory=list)    # MuJoCo order
    actions: List[List[float]] = field(default_factory=list)            # Isaac Lab order
    torques: List[List[float]] = field(default_factory=list)            # applied ctrl


# ── Predefined test scenarios ────────────────────────────────────────────────

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


# ── Core simulation ──────────────────────────────────────────────────────────

def run_episode(
    model: mujoco.MjModel,
    policy: torch.jit.ScriptModule,
    config: dict,
    commands: List[float],
    max_steps: int,
    default_angles: np.ndarray,
    default_angles_isaaclab: np.ndarray,
    actuator_net: Optional[ActuatorNetWrapper] = None,
    rma_mode: bool = False,
) -> EpisodeData:
    """Run one headless episode and record time-series data.

    Returns EpisodeData with per-control-step recordings.
    Terminates early if the base height drops below 0.15 m.
    """
    data = mujoco.MjData(model)

    # Fixed initial state for fair comparison
    data.qpos[0:3] = [0, 0, 0.35]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = default_angles
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # Unpack config
    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    action_scale = config["action_scale"]
    lin_vel_scale = config.get("lin_vel_scale", 2.0)
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
    num_obs = 52 if rma_mode else config["num_obs"]  # Override to 52 for RMA
    control_decimation = config["control_decimation"]

    cmd = np.array(commands, dtype=np.float32)
    action = np.zeros(config["num_actions"], dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    episode = EpisodeData()
    counter = 0

    for step in range(max_steps * control_decimation):
        if actuator_net is not None:
            # ActuatorNet: neural network torque prediction
            tau = actuator_net.compute_torques(
                target_dof_pos, data.qpos[7:], data.qvel[6:]
            )
        else:
            # PD control
            tau = pd_control(target_dof_pos, data.qpos[7:], kps,
                             np.zeros_like(kds), data.qvel[6:], kds)

        # Ctrl swap: actuator order (FR,FL,RR,RL) ≠ joint order (FL,FR,RL,RR)
        data.ctrl[0:3] = tau[3:6]    # FR
        data.ctrl[3:6] = tau[0:3]    # FL
        data.ctrl[6:9] = tau[9:12]   # RR
        data.ctrl[9:12] = tau[6:9]   # RL
        mujoco.mj_step(model, data)

        counter += 1
        if counter % control_decimation != 0:
            continue

        # ── Record data at control frequency ────────────────────────────
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

        # Build observation
        obs[0:3] = base_lin_vel * lin_vel_scale
        obs[3:6] = base_ang_vel * ang_vel_scale
        obs[6:9] = gravity_orientation
        obs[9:12] = cmd * cmd_scale
        obs[12:24] = (qj_il - default_angles_isaaclab) * dof_pos_scale
        obs[24:36] = dqj_il * dof_vel_scale
        obs[36:48] = action

        # RMA mode: append 4 dummy privileged observations (48:52)
        # Values represent "unrandomized" environment (friction=1.0, mass_offset=0.0, kp_scale=1.0, kd_scale=1.0)
        if num_obs == 52:
            obs[48] = 1.0    # ground_friction
            obs[49] = 0.0    # base_mass_offset
            obs[50] = 1.0    # kp_scale
            obs[51] = 1.0    # kd_scale

        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
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
        episode.torques.append(data.ctrl.copy().tolist())
        episode.steps += 1

        # Early termination on fall
        if height < 0.15:
            break

    return episode


# ── Metrics computation ──────────────────────────────────────────────────────

def compute_metrics(
    episodes: List[EpisodeData],
    commands: List[float],
    max_steps: int,
) -> Dict[str, Any]:
    """Compute aggregate metrics over a list of episodes."""
    success_threshold = int(max_steps * 0.8)

    n_success = sum(1 for ep in episodes if ep.steps >= success_threshold)
    survival_steps = [ep.steps for ep in episodes]

    vx_sq, vy_sq, wz_sq = [], [], []
    all_heights = []
    action_smoothness = []

    for ep in episodes:
        if len(ep.body_velocities) == 0:
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

    rmse_vx = float(np.sqrt(np.mean(vx_sq))) if vx_sq else 0.0
    rmse_vy = float(np.sqrt(np.mean(vy_sq))) if vy_sq else 0.0
    rmse_wz = float(np.sqrt(np.mean(wz_sq))) if wz_sq else 0.0
    overall_rmse = float(np.sqrt(np.mean(vx_sq + vy_sq + wz_sq))) if vx_sq else 0.0

    torque_magnitudes = []
    for ep in episodes:
        if ep.torques:
            torque_magnitudes.extend(
                [float(np.linalg.norm(t)) for t in ep.torques]
            )

    return {
        "success_rate": n_success / len(episodes) * 100,
        "mean_survival_steps": float(np.mean(survival_steps)),
        "std_survival_steps": float(np.std(survival_steps)),
        "rmse_vx": rmse_vx,
        "rmse_vy": rmse_vy,
        "rmse_wz": rmse_wz,
        "overall_rmse": overall_rmse,
        "mean_height": float(np.mean(all_heights)) if all_heights else 0.0,
        "std_height": float(np.std(all_heights)) if all_heights else 0.0,
        "mean_action_smoothness": float(np.mean(action_smoothness)) if action_smoothness else 0.0,
        "mean_torque_magnitude": float(np.mean(torque_magnitudes)) if torque_magnitudes else 0.0,
    }


# ── Evaluate all scenarios ───────────────────────────────────────────────────

def evaluate_policy(
    policy_path: str,
    config: dict,
    xml_path: str,
    experiment_set: str,
    episodes: int,
    max_steps: int,
    label: str,
    actuator_net: Optional[ActuatorNetWrapper] = None,
    rma_mode: bool = False,
) -> Dict[str, Any]:
    """Run every scenario in *experiment_set* and return structured results."""
    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = config["simulation_dt"]

    # Load policy
    policy = torch.jit.load(policy_path)
    policy.eval()

    default_angles = np.array(config["default_angles"], dtype=np.float32)
    default_angles_il = default_angles[MUJOCO_TO_ISAACLAB]

    scenarios = EXPERIMENTS[experiment_set]
    results: Dict[str, Any] = {"label": label, "experiment": experiment_set, "scenarios": {}}

    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {sc.name}  cmd=[{sc.commands[0]:.2f}, {sc.commands[1]:.2f}, {sc.commands[2]:.2f}]")
        print(f"{'='*60}")

        ep_list: List[EpisodeData] = []
        for ep_idx in range(episodes):
            ep = run_episode(model, policy, config, sc.commands,
                             max_steps, default_angles, default_angles_il,
                             actuator_net=actuator_net, rma_mode=rma_mode)
            ep_list.append(ep)
            status = "OK" if ep.steps >= int(max_steps * 0.8) else "FALL"
            print(f"  Episode {ep_idx+1}/{episodes}: steps={ep.steps} [{status}]")

        metrics = compute_metrics(ep_list, sc.commands, max_steps)
        print(f"  -> success={metrics['success_rate']:.0f}%  "
              f"RMSE_vx={metrics['rmse_vx']:.3f}  "
              f"RMSE_vy={metrics['rmse_vy']:.3f}  "
              f"RMSE_wz={metrics['rmse_wz']:.3f}")

        # Find best episode (longest, break ties by lowest overall RMSE)
        best_ep = max(ep_list, key=lambda e: e.steps)

        results["scenarios"][sc.name] = {
            "config": asdict(sc),
            "metrics": metrics,
            "best_episode": asdict(best_ep),
        }

    return results


# ── Save helpers ─────────────────────────────────────────────────────────────

def _convert_numpy(obj):
    """Recursively convert numpy types to Python natives for JSON."""
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


def save_results(results: Dict[str, Any], output_dir: str):
    """Save eval_results.json and eval_data.npz."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON metrics (without heavy time-series)
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

    json_path = out / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(_convert_numpy(metrics_only), f, indent=2)
    print(f"\nMetrics saved: {json_path}")

    # NPZ with best-episode time-series per scenario
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

    npz_path = out / "eval_data.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Time-series saved: {npz_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Go2 policy in headless MuJoCo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", type=str, required=True,
                        help="Path to TorchScript policy (.pt)")
    parser.add_argument("--config", type=str, default="scripts/go2/go2_isaaclab.yaml",
                        help="YAML config (physics params, obs scales)")
    parser.add_argument("--experiment", type=str, default="directions",
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Scenario set to run")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per scenario")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max policy steps per episode (500 = 10 s at 50 Hz)")
    parser.add_argument("--output_dir", type=str, default="experiments/go2/results/eval",
                        help="Where to save results")
    parser.add_argument("--label", type=str, default="policy",
                        help="Human-readable label for this evaluation run")
    parser.add_argument("--actuator_net", action="store_true",
                        help="Use ActuatorNet for torque computation instead of PD")
    parser.add_argument("--actuator_net_path", type=str, default=None,
                        help="Path to actuator net model (default: actuator_net/models/go2/)")
    parser.add_argument("--rma", action="store_true",
                        help="RMA mode: append 4 dummy privileged observations (52 dims total)")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # --policy overrides whatever is in yaml
    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = project_root / policy_path

    xml_path = Path(config["xml_path"])
    if not xml_path.is_absolute():
        xml_path = project_root / xml_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # Load actuator net if requested
    anet = None
    if args.actuator_net:
        if args.actuator_net_path:
            anet_model = args.actuator_net_path
            anet_scaler = args.actuator_net_path.replace("actuator_net_jit.pt", "feature_scaler.pkl")
        else:
            anet_model = str(project_root / "actuator_net/models/go2/actuator_net_jit.pt")
            anet_scaler = str(project_root / "actuator_net/models/go2/feature_scaler.pkl")
        anet = ActuatorNetWrapper(anet_model, anet_scaler)

    torque_mode = "ActuatorNet" if anet else "PD"

    print("=" * 60)
    print("Go2 MuJoCo Policy Evaluation")
    print("=" * 60)
    print(f"  Policy:     {policy_path}")
    print(f"  Config:     {config_path}")
    print(f"  XML:        {xml_path}")
    print(f"  Torque:     {torque_mode}")
    print(f"  Experiment: {args.experiment}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Max steps:  {args.max_steps}")
    print(f"  Label:      {args.label}")
    print(f"  Output:     {output_dir}")

    # Determine experiment sets
    if args.experiment == "all":
        sets = ["directions", "speeds"]
    else:
        sets = [args.experiment]

    all_results = {}
    for exp_set in sets:
        results = evaluate_policy(
            policy_path=str(policy_path),
            config=config,
            xml_path=str(xml_path),
            experiment_set=exp_set,
            episodes=args.episodes,
            max_steps=args.max_steps,
            label=args.label,
            actuator_net=anet,
            rma_mode=args.rma,
        )
        save_results(results, str(output_dir))
        all_results[exp_set] = results

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for exp_set, res in all_results.items():
        print(f"\n  {exp_set.upper()}:")
        print(f"  {'Scenario':<15} {'Success':>8} {'RMSE_vx':>8} {'RMSE_vy':>8} {'RMSE_wz':>8} {'Steps':>8}")
        print(f"  {'-'*55}")
        for name, sc in res["scenarios"].items():
            m = sc["metrics"]
            print(f"  {name:<15} {m['success_rate']:>7.0f}% "
                  f"{m['rmse_vx']:>8.3f} {m['rmse_vy']:>8.3f} "
                  f"{m['rmse_wz']:>8.3f} {m['mean_survival_steps']:>8.0f}")


if __name__ == "__main__":
    main()
