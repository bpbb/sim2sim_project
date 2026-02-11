#!/usr/bin/env python3
"""Compare Go2 policies trained with different DR levels.

Loads eval_results.json and eval_data.npz from each DR run produced by
eval_mujoco.py (and optionally eval_isaaclab.py), then generates
publication-quality comparison plots and a text summary.

Plots:
    1. success_rate_comparison.png   — grouped bar: scenarios x DR levels
    2. rmse_comparison.png           — per-scenario + per-component RMSE
    3. velocity_tracking_timeseries.png — small multiples: one col per policy
    4. tracking_error.png            — smoothed |actual-cmd| over time per policy
    5. overall_comparison.png        — 2x2 summary figure
    +  sim2sim_comparison.png        — Isaac Lab vs MuJoCo (if --isaaclab given)

Usage:
    python experiments/go2/compare_dr.py \
        --baseline  experiments/go2/results/baseline \
        --moderate  experiments/go2/results/moderate \
        --aggressive experiments/go2/results/aggressive \
        --actuator_net experiments/go2/results/actuator_net \
        --output_dir experiments/go2/results/comparison

    # With Isaac Lab comparison:
    python experiments/go2/compare_dr.py \
        --baseline experiments/go2/results/baseline \
        --isaaclab experiments/go2/results/isaaclab_baseline \
        --output_dir experiments/go2/results/comparison
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Colour scheme ────────────────────────────────────────────────────────────
COLORS = {
    "baseline":     "#3498db",  # blue
    "moderate":     "#f39c12",  # orange
    "aggressive":   "#2ecc71",  # green
    "actuator_net": "#9b59b6",  # purple
}
DR_LABELS = ["baseline", "moderate", "aggressive", "actuator_net"]


# ── Load helpers ─────────────────────────────────────────────────────────────

def load_eval(result_dir: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Load eval_results.json and eval_data.npz from *result_dir*."""
    d = Path(result_dir)
    with open(d / "eval_results.json") as f:
        metrics = json.load(f)
    npz = dict(np.load(d / "eval_data.npz", allow_pickle=True))
    return metrics, npz


def _scenario_names(metrics: Dict) -> List[str]:
    return list(metrics["scenarios"].keys())


# ── Plot 1: Success rate comparison ─────────────────────────────────────────

def plot_success_rate(
    all_metrics: Dict[str, Dict],
    save_path: Path,
):
    """Grouped bar chart: scenarios x DR levels, with 80 % threshold."""
    scenarios = _scenario_names(list(all_metrics.values())[0])
    present = [dr for dr in DR_LABELS if dr in all_metrics]
    n = len(scenarios)
    x = np.arange(n)
    width = 0.8 / max(len(present), 1)

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 5))

    for i, dr in enumerate(present):
        rates = [all_metrics[dr]["scenarios"].get(s, {}).get("metrics", {}).get("success_rate", 0)
                 for s in scenarios]
        bars = ax.bar(x + i * width, rates, width, label=dr.capitalize(),
                      color=COLORS[dr], edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.6, label="80 % threshold")
    ax.set_xticks(x + width * len(present) / 2)
    ax.set_xticklabels(scenarios, rotation=35, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Success Rate by Scenario and DR Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [1/6] {save_path}")


# ── Plot 2: RMSE comparison ─────────────────────────────────────────────────

def plot_rmse(
    all_metrics: Dict[str, Dict],
    save_path: Path,
):
    """Two panels: per-scenario overall RMSE + per-component (vx/vy/wz)."""
    scenarios = _scenario_names(list(all_metrics.values())[0])
    present = [dr for dr in DR_LABELS if dr in all_metrics]
    n = len(scenarios)
    x = np.arange(n)
    width = 0.8 / max(len(present), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(12, n * 1.5), 5))

    # Panel 1 — per-scenario overall RMSE
    for i, dr in enumerate(present):
        vals = [all_metrics[dr]["scenarios"].get(s, {}).get("metrics", {}).get("overall_rmse", 0)
                for s in scenarios]
        ax1.bar(x + i * width, vals, width, label=dr.capitalize(),
                color=COLORS[dr], edgecolor="black", linewidth=0.5)

    ax1.set_xticks(x + width * len(present) / 2)
    ax1.set_xticklabels(scenarios, rotation=35, ha="right")
    ax1.set_ylabel("Overall RMSE")
    ax1.set_title("Per-Scenario Overall RMSE")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2 — per-component aggregated RMSE across all scenarios
    components = ["rmse_vx", "rmse_vy", "rmse_wz"]
    comp_labels = ["vx", "vy", "wz"]
    x2 = np.arange(len(components))

    width2 = 0.8 / max(len(present), 1)
    for i, dr in enumerate(present):
        vals = []
        for comp in components:
            sc_vals = [all_metrics[dr]["scenarios"][s]["metrics"][comp]
                       for s in all_metrics[dr]["scenarios"]]
            vals.append(float(np.mean(sc_vals)))
        ax2.bar(x2 + i * width2, vals, width2, label=dr.capitalize(),
                color=COLORS[dr], edgecolor="black", linewidth=0.5)

    ax2.set_xticks(x2 + width2 * len(present) / 2)
    ax2.set_xticklabels(comp_labels)
    ax2.set_ylabel("Mean RMSE")
    ax2.set_title("Per-Component RMSE (averaged over scenarios)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [2/6] {save_path}")


# ── Plot 3: Velocity tracking — small multiples (one column per policy) ──────

# Scenarios to show: vx-only, vy-only, vx+vy combined
_TRACKING_SCENARIOS = [
    ("forward",  "Forward (vx=0.5)"),
    ("left",     "Left (vy=0.5)"),
    ("diagonal", "Diagonal (vx=0.5, vy=0.5)"),
]

_VEL_COMPONENTS = [
    (0, "vx", "body_vel", "#e74c3c"),   # red
    (1, "vy", "body_vel", "#3498db"),    # blue
    (2, "wz", "ang_vel",  "#2ecc71"),    # green
]

_SMOOTH_WINDOW = 25  # 0.5s rolling average at 50 Hz


def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """1-D rolling mean with same-length output (edge-padded)."""
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed


def compute_transient_metrics(velocity: np.ndarray, command: float, dt: float = 0.02) -> Dict[str, float]:
    """Compute transient response metrics: settling time, overshoot, steady-state error.

    Args:
        velocity: velocity signal over time [T]
        command: commanded velocity value
        dt: timestep in seconds (default 0.02 = 50Hz)

    Returns:
        dict with settling_time (s), overshoot (%), steady_state_error
    """
    if abs(command) < 1e-6:  # Skip zero commands
        return {
            "settling_time": 0.0,
            "overshoot_pct": 0.0,
            "steady_state_error": 0.0,
        }

    error = np.abs(velocity - command)
    tolerance = 0.05 * abs(command)  # ±5% band

    # Settling time: first time entering and staying in tolerance for 1 second (50 steps)
    window_size = int(1.0 / dt)  # 1 second
    settling_idx = None
    for i in range(len(error) - window_size):
        if np.all(error[i:i+window_size] < tolerance):
            settling_idx = i
            break

    settling_time = settling_idx * dt if settling_idx is not None else np.inf

    # Overshoot: max deviation before settling (as percentage)
    if settling_idx is not None and settling_idx > 0:
        overshoot = np.max(np.abs(velocity[:settling_idx] - command))
    else:
        overshoot = np.max(np.abs(velocity - command))
    overshoot_pct = (overshoot / abs(command)) * 100.0

    # Steady-state error: mean error after settling (or last 2 seconds if never settled)
    if settling_idx is not None:
        ss_error = np.mean(error[settling_idx:])
    else:
        last_2s = int(2.0 / dt)
        ss_error = np.mean(error[-last_2s:]) if len(error) >= last_2s else np.mean(error)

    return {
        "settling_time": settling_time,
        "overshoot_pct": overshoot_pct,
        "steady_state_error": ss_error,
    }


def plot_velocity_timeseries(
    all_npz: Dict[str, Dict[str, np.ndarray]],
    all_metrics: Dict[str, Dict],
    save_path: Path,
    scenario: str = "forward",
):
    """Small-multiples: rows = policies, cols = scenarios (HORIZONTAL layout).

    Each subplot shows smoothed vx / vy / wz as separate colored lines
    with faded raw signal behind, plus command references as dashed lines.
    Shared y-axes per row so magnitudes are directly comparable.
    """
    present = [dr for dr in DR_LABELS if dr in all_npz]
    n_policies = len(present)
    n_scenarios = len(_TRACKING_SCENARIOS)

    # SWAPPED: rows = policies (horizontal layout)
    fig, axes = plt.subplots(
        n_policies, n_scenarios,
        figsize=(5.0 * n_scenarios, 3.0 * n_policies),
        sharex=True, sharey="row",
        squeeze=False,
    )

    for row, dr in enumerate(present):
        for col, (sc_key, sc_title) in enumerate(_TRACKING_SCENARIOS):
            ax = axes[row, col]
            npz = all_npz[dr]

            for comp_idx, comp_label, vel_key, comp_color in _VEL_COMPONENTS:
                bv_key = f"{sc_key}_{vel_key}"
                cmd_key = f"{sc_key}_commands"

                if bv_key not in npz:
                    continue

                vel = npz[bv_key][:, comp_idx]
                cmd = npz[cmd_key]
                t = np.arange(len(vel)) * 0.02  # 50 Hz

                # Raw signal (faded)
                ax.plot(t, vel, color=comp_color, linewidth=0.4, alpha=0.25)
                # Smoothed signal
                ax.plot(t, _rolling_mean(vel, _SMOOTH_WINDOW),
                        color=comp_color, linewidth=1.8, label=comp_label)

                # Command reference
                cmd_val = float(cmd[comp_idx]) if comp_idx < 2 else float(cmd[2])
                ax.axhline(y=cmd_val, color=comp_color, linestyle="--",
                           linewidth=1.0, alpha=0.4)

            ax.grid(alpha=0.2)

            # Column header = scenario name
            if row == 0:
                ax.set_title(sc_title,
                             fontsize=11, fontweight="bold")
            # Row label = policy name
            if col == 0:
                ax.set_ylabel(f"{dr.replace('_', ' ').capitalize()}\n(m/s, rad/s)",
                             fontsize=10, fontweight="bold",
                             color=COLORS[dr])
            # X label
            if row == n_policies - 1:
                ax.set_xlabel("Time (s)")
            # Legend only in top-right
            if row == 0 and col == n_scenarios - 1:
                ax.legend(fontsize=7, loc="upper right")

    plt.suptitle("Velocity Tracking — MuJoCo (best episode per scenario)",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [3/6] {save_path}")


# ── Plot 3b: Tracking error over time ────────────────────────────────────────

def plot_tracking_error(
    all_npz: Dict[str, Dict[str, np.ndarray]],
    all_metrics: Dict[str, Dict],
    save_path: Path,
):
    """One row per scenario, showing smoothed tracking error per policy.

    Error = sqrt((vx-cmd_vx)^2 + (vy-cmd_vy)^2) for linear,
    and |wz-cmd_wz| for angular (separate panel).
    This gives a single clean line per policy — easy to compare.
    """
    present = [dr for dr in DR_LABELS if dr in all_npz]
    n_scenarios = len(_TRACKING_SCENARIOS)

    fig, axes = plt.subplots(
        n_scenarios, 2,
        figsize=(12, 3.5 * n_scenarios),
        sharex=True,
        squeeze=False,
    )

    for row, (sc_key, sc_title) in enumerate(_TRACKING_SCENARIOS):
        ax_lin = axes[row, 0]  # linear velocity error
        ax_ang = axes[row, 1]  # angular velocity error

        for dr in present:
            npz = all_npz[dr]
            bv_key = f"{sc_key}_body_vel"
            av_key = f"{sc_key}_ang_vel"
            cmd_key = f"{sc_key}_commands"

            if bv_key not in npz:
                continue

            body_vel = npz[bv_key]   # [T, 3]
            ang_vel = npz[av_key]    # [T, 3]
            cmd = npz[cmd_key]       # [3]
            t = np.arange(len(body_vel)) * 0.02

            # Linear error: sqrt((vx-cmd_vx)^2 + (vy-cmd_vy)^2)
            lin_err = np.sqrt(
                (body_vel[:, 0] - float(cmd[0])) ** 2
                + (body_vel[:, 1] - float(cmd[1])) ** 2
            )
            # Angular error: |wz - cmd_wz|
            ang_err = np.abs(ang_vel[:, 2] - float(cmd[2]))

            ax_lin.plot(t, _rolling_mean(lin_err, _SMOOTH_WINDOW),
                        color=COLORS[dr], linewidth=2.0,
                        label=dr.replace("_", " ").capitalize())
            ax_ang.plot(t, _rolling_mean(ang_err, _SMOOTH_WINDOW),
                        color=COLORS[dr], linewidth=2.0,
                        label=dr.replace("_", " ").capitalize())

        for ax in [ax_lin, ax_ang]:
            ax.grid(alpha=0.3)
            ax.set_ylim(bottom=0)

        if row == 0:
            ax_lin.set_title("Linear Vel Error (m/s)", fontsize=11)
            ax_ang.set_title("Angular Vel Error (rad/s)", fontsize=11)
            ax_lin.legend(fontsize=8)
        ax_lin.set_ylabel(sc_title, fontsize=10)
        if row == n_scenarios - 1:
            ax_lin.set_xlabel("Time (s)")
            ax_ang.set_xlabel("Time (s)")

    plt.suptitle("Tracking Error Over Time (smoothed, lower = better)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [4/6] {save_path}")


# ── Plot 4b: Transient metrics comparison ───────────────────────────────────

def plot_transient_metrics(
    all_npz: Dict[str, Dict[str, np.ndarray]],
    save_path: Path,
):
    """Grouped bar chart showing settling time, overshoot, and steady-state error."""
    present = [dr for dr in DR_LABELS if dr in all_npz]
    scenarios_to_analyze = ["forward", "left", "diagonal"]  # Focus on key scenarios

    # Compute transient metrics for each policy × scenario × component
    results = {dr: {sc: {} for sc in scenarios_to_analyze} for dr in present}

    for dr in present:
        npz = all_npz[dr]
        for sc_key in scenarios_to_analyze:
            bv_key = f"{sc_key}_body_vel"
            av_key = f"{sc_key}_ang_vel"
            cmd_key = f"{sc_key}_commands"

            if bv_key not in npz or cmd_key not in npz:
                continue

            body_vel = npz[bv_key]  # [T, 3]
            ang_vel = npz[av_key]   # [T, 3]
            cmd = npz[cmd_key]      # [3]

            # Analyze vx (forward velocity) - most important
            vx_metrics = compute_transient_metrics(body_vel[:, 0], float(cmd[0]))
            results[dr][sc_key]["vx"] = vx_metrics

            # Analyze wz (yaw rate) for turning scenarios
            if abs(float(cmd[2])) > 0.01:
                wz_metrics = compute_transient_metrics(ang_vel[:, 2], float(cmd[2]))
                results[dr][sc_key]["wz"] = wz_metrics

    # Create 3-panel plot: settling time, overshoot, steady-state error
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics_to_plot = [
        ("settling_time", "Settling Time (s)", "lower is better"),
        ("overshoot_pct", "Overshoot (%)", "lower is better"),
        ("steady_state_error", "Steady-State Error (m/s)", "lower is better"),
    ]

    x_labels = []
    for sc in scenarios_to_analyze:
        x_labels.append(f"{sc.capitalize()}\nvx")

    x = np.arange(len(x_labels))
    width = 0.2
    offsets = np.linspace(-width * (len(present) - 1) / 2, width * (len(present) - 1) / 2, len(present))

    for ax_idx, (metric_key, ylabel, subtitle) in enumerate(metrics_to_plot):
        ax = axes[ax_idx]

        for dr_idx, dr in enumerate(present):
            values = []
            for sc in scenarios_to_analyze:
                if "vx" in results[dr][sc] and metric_key in results[dr][sc]["vx"]:
                    val = results[dr][sc]["vx"][metric_key]
                    # Cap settling time at 10s for visualization
                    if metric_key == "settling_time" and val == np.inf:
                        val = 10.0
                    values.append(val)
                else:
                    values.append(0.0)

            ax.bar(x + offsets[dr_idx], values, width, label=dr.replace("_", " ").capitalize(),
                   color=COLORS[dr], alpha=0.8)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Scenario", fontsize=10)
        ax.set_title(f"{ylabel}\n({subtitle})", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 2:  # Legend on rightmost plot
            ax.legend(fontsize=9, loc="upper right")

    plt.suptitle("Transient Response Metrics — MuJoCo Deployment",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] {save_path} (transient metrics)")


# ── Plot 5: Overall comparison (2x2 summary) ────────────────────────────────

def plot_overall(
    all_metrics: Dict[str, Dict],
    save_path: Path,
):
    """2x2 figure: overall success, overall RMSE, action smoothness, table."""
    present = [dr for dr in DR_LABELS if dr in all_metrics]

    # Aggregate across scenarios
    def agg(dr, key):
        vals = [all_metrics[dr]["scenarios"][s]["metrics"][key]
                for s in all_metrics[dr]["scenarios"]]
        return float(np.mean(vals))

    fig = plt.figure(figsize=(12, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # (0, 0) Overall success rate
    ax = fig.add_subplot(gs[0, 0])
    rates = [agg(dr, "success_rate") for dr in present]
    bars = ax.bar(present, rates, color=[COLORS[dr] for dr in present],
                  edgecolor="black", linewidth=0.5)
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.6)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Overall Success Rate")
    ax.grid(axis="y", alpha=0.3)

    # (0, 1) Overall RMSE
    ax = fig.add_subplot(gs[0, 1])
    rmses = [agg(dr, "overall_rmse") for dr in present]
    bars = ax.bar(present, rmses, color=[COLORS[dr] for dr in present],
                  edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Overall RMSE")
    ax.set_title("Overall Velocity Tracking RMSE")
    ax.grid(axis="y", alpha=0.3)

    # (1, 0) Action smoothness + torque
    ax = fig.add_subplot(gs[1, 0])
    x_idx = np.arange(len(present))
    w = 0.35
    smooth = [agg(dr, "mean_action_smoothness") for dr in present]
    torque = [agg(dr, "mean_torque_magnitude") for dr in present]
    ax.bar(x_idx - w / 2, smooth, w, label="Action smoothness",
           color=[COLORS[dr] for dr in present], edgecolor="black", linewidth=0.5)
    ax.bar(x_idx + w / 2, torque, w, label="Torque magnitude",
           color=[COLORS[dr] for dr in present], edgecolor="black", linewidth=0.5,
           alpha=0.5, hatch="//")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([dr.capitalize() for dr in present])
    ax.set_ylabel("Magnitude")
    ax.set_title("Action Smoothness & Torque")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (1, 1) Summary table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    cols = ["DR Level", "Success%", "RMSE", "Smooth", "Torque", "Ht (m)"]
    rows = []
    for dr in present:
        rows.append([
            dr.capitalize(),
            f"{agg(dr, 'success_rate'):.1f}",
            f"{agg(dr, 'overall_rmse'):.3f}",
            f"{agg(dr, 'mean_action_smoothness'):.3f}",
            f"{agg(dr, 'mean_torque_magnitude'):.1f}",
            f"{agg(dr, 'mean_height'):.3f}",
        ])
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Color header
    for j in range(len(cols)):
        table[0, j].set_facecolor("#d5d5d5")

    ax.set_title("Summary Table", pad=20)

    plt.suptitle("Go2 DR Comparison — Overall", fontsize=14, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [5/6] {save_path}")


# ── Plot 5: Sim2Sim comparison (Isaac Lab vs MuJoCo) ─────────────────────────

_SIM2SIM_SCENARIOS = [
    ("forward",  "Forward (vx=0.5)"),
    ("left",     "Left (vy=0.5)"),
    ("diagonal", "Diagonal (vx=0.5, vy=0.5)"),
]


def plot_sim2sim_comparison(
    mujoco_npz: Dict[str, np.ndarray],
    isaaclab_npz: Dict[str, np.ndarray],
    save_path: Path,
    label: str = "baseline",
):
    """Compare velocity tracking between Isaac Lab and MuJoCo for one policy.

    3 rows (scenarios) x 3 cols (vx / vy / wz).
    Solid = Isaac Lab, dashed = MuJoCo.
    """
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)

    color_il = "#e74c3c"   # red for Isaac Lab
    color_mj = "#3498db"   # blue for MuJoCo

    for row, (sc_key, sc_title) in enumerate(_SIM2SIM_SCENARIOS):
        for col, (comp_idx, comp_label, vel_key) in enumerate([
            (0, "vx (m/s)", "body_vel"),
            (1, "vy (m/s)", "body_vel"),
            (2, "wz (rad/s)", "ang_vel"),
        ]):
            ax = axes[row, col]

            bv_key = f"{sc_key}_{vel_key}"
            cmd_key = f"{sc_key}_commands"

            # Isaac Lab
            if bv_key in isaaclab_npz:
                vel = isaaclab_npz[bv_key]
                t = np.arange(len(vel)) * 0.02
                ax.plot(t, vel[:, comp_idx], color=color_il,
                        linewidth=1.5, label="Isaac Lab", alpha=0.85)

            # MuJoCo
            if bv_key in mujoco_npz:
                vel = mujoco_npz[bv_key]
                t = np.arange(len(vel)) * 0.02
                ax.plot(t, vel[:, comp_idx], color=color_mj,
                        linewidth=1.5, linestyle="--", label="MuJoCo", alpha=0.85)

            # Command reference
            for npz_src in [isaaclab_npz, mujoco_npz]:
                if cmd_key in npz_src:
                    cmd = npz_src[cmd_key]
                    cmd_val = float(cmd[comp_idx]) if comp_idx < 2 else float(cmd[2])
                    ax.axhline(y=cmd_val, color="black", linestyle=":",
                               linewidth=1.0, alpha=0.4, label=f"cmd={cmd_val:.2f}")
                    break

            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)

            if row == 0:
                ax.set_title(comp_label, fontsize=11)
            if col == 0:
                ax.set_ylabel(sc_title, fontsize=10)
            if row == 2:
                ax.set_xlabel("Time (s)")

    plt.suptitle(f"Sim2Sim Comparison — Isaac Lab vs MuJoCo ({label.capitalize()})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [+] {save_path} (sim2sim)")


# ── Text summary ─────────────────────────────────────────────────────────────

def write_summary(all_metrics: Dict[str, Dict], save_path: Path):
    """Write a plain-text summary file."""
    present = [dr for dr in DR_LABELS if dr in all_metrics]
    scenarios = _scenario_names(list(all_metrics.values())[0])

    lines = ["Go2 DR Comparison Summary", "=" * 60, ""]

    for dr in present:
        lines.append(f"--- {dr.upper()} ---")
        for s in scenarios:
            m = all_metrics[dr]["scenarios"].get(s, {}).get("metrics", {})
            lines.append(
                f"  {s:<15}  success={m.get('success_rate', 0):5.1f}%  "
                f"RMSE(vx={m.get('rmse_vx', 0):.3f} vy={m.get('rmse_vy', 0):.3f} "
                f"wz={m.get('rmse_wz', 0):.3f})  "
                f"steps={m.get('mean_survival_steps', 0):.0f}"
            )
        lines.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary: {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare Go2 policies across DR levels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", type=str, help="Path to baseline eval results dir")
    parser.add_argument("--moderate", type=str, help="Path to moderate eval results dir")
    parser.add_argument("--aggressive", type=str, help="Path to aggressive eval results dir")
    parser.add_argument("--actuator_net", type=str, help="Path to actuator_net eval results dir")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/go2/results/comparison",
                        help="Where to save comparison plots")
    parser.add_argument("--scenario", type=str, default="forward",
                        help="Scenario name for velocity time-series plot (lowercase)")
    parser.add_argument("--isaaclab", type=str, default=None,
                        help="Path to Isaac Lab eval results dir (for sim2sim comparison)")
    parser.add_argument("--isaaclab_label", type=str, default="baseline",
                        help="DR label for the Isaac Lab results (for plot title)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = project_root / out
    out.mkdir(parents=True, exist_ok=True)

    # Load available results
    all_metrics: Dict[str, Dict] = {}
    all_npz: Dict[str, Dict[str, np.ndarray]] = {}

    for dr, path in [("baseline", args.baseline),
                     ("moderate", args.moderate),
                     ("aggressive", args.aggressive),
                     ("actuator_net", args.actuator_net)]:
        if path is None:
            continue
        p = Path(path) if Path(path).is_absolute() else project_root / path
        if not (p / "eval_results.json").exists():
            print(f"Warning: {p / 'eval_results.json'} not found, skipping {dr}")
            continue
        m, n = load_eval(str(p))
        all_metrics[dr] = m
        all_npz[dr] = n

    if not all_metrics:
        print("Error: No results found. Run eval_mujoco.py first.")
        return

    present = [dr for dr in DR_LABELS if dr in all_metrics]
    print(f"\nLoaded results for: {', '.join(present)}")
    print(f"Output directory:   {out}\n")

    # Generate plots
    plot_success_rate(all_metrics, out / "success_rate_comparison.png")
    plot_rmse(all_metrics, out / "rmse_comparison.png")
    plot_velocity_timeseries(all_npz, all_metrics, out / "velocity_tracking_timeseries.png",
                             scenario=args.scenario)
    plot_tracking_error(all_npz, all_metrics, out / "tracking_error.png")
    plot_transient_metrics(all_npz, out / "transient_metrics.png")
    plot_overall(all_metrics, out / "overall_comparison.png")
    write_summary(all_metrics, out / "summary.txt")

    # Optional: sim2sim comparison (Isaac Lab vs MuJoCo)
    if args.isaaclab:
        il_path = Path(args.isaaclab) if Path(args.isaaclab).is_absolute() else project_root / args.isaaclab
        if (il_path / "eval_data.npz").exists():
            il_npz = dict(np.load(il_path / "eval_data.npz", allow_pickle=True))
            # Use the matching DR level from MuJoCo for comparison
            mj_dr = args.isaaclab_label
            if mj_dr in all_npz:
                mj_npz = all_npz[mj_dr]
            elif all_npz:
                mj_dr = list(all_npz.keys())[0]
                mj_npz = all_npz[mj_dr]
            else:
                mj_npz = {}
            plot_sim2sim_comparison(mj_npz, il_npz,
                                   out / "sim2sim_comparison.png",
                                   label=mj_dr)
        else:
            print(f"  Warning: {il_path / 'eval_data.npz'} not found, skipping sim2sim plot")

    print("\nDone!")


if __name__ == "__main__":
    main()
