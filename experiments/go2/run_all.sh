#!/usr/bin/env bash
# Run Go2 DR comparison: evaluate all DR levels + actuator net then generate plots.
#
# Expects policies at:
#   $POLICY_DIR/baseline.pt
#   $POLICY_DIR/moderate.pt
#   $POLICY_DIR/aggressive.pt
#   $POLICY_DIR/actuator_net.pt
#
# ActuatorNet model at:
#   $ACTUATOR_NET_PATH (default: $PROJECT_ROOT/actuator_net/models/go2/actuator_net_jit.pt)
#
# Usage:
#   bash experiments/go2/run_all.sh
#   EPISODES=10 MAX_STEPS=1000 bash experiments/go2/run_all.sh
#   POLICY_DIR=/path/to/policies bash experiments/go2/run_all.sh
set -euo pipefail

# ── Configurable via environment variables ──────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
POLICY_DIR="${POLICY_DIR:-$PROJECT_ROOT/logs/go2/policies}"
CONFIG="${CONFIG:-$PROJECT_ROOT/scripts/go2/go2_isaaclab.yaml}"
EXPERIMENT="${EXPERIMENT:-directions}"
EPISODES="${EPISODES:-5}"
MAX_STEPS="${MAX_STEPS:-500}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/experiments/go2/results}"
ACTUATOR_NET_PATH="${ACTUATOR_NET_PATH:-$PROJECT_ROOT/actuator_net/models/go2/actuator_net_jit.pt}"

echo "============================================================"
echo "Go2 DR Comparison — Full Pipeline"
echo "============================================================"
echo "  Project root:     $PROJECT_ROOT"
echo "  Policy dir:       $POLICY_DIR"
echo "  Config:           $CONFIG"
echo "  Experiment:       $EXPERIMENT"
echo "  Episodes:         $EPISODES"
echo "  Max steps:        $MAX_STEPS"
echo "  Results dir:      $RESULTS_DIR"
echo "  ActuatorNet path: $ACTUATOR_NET_PATH"
echo "============================================================"

EVAL_SCRIPT="$PROJECT_ROOT/experiments/go2/eval_mujoco.py"
COMPARE_SCRIPT="$PROJECT_ROOT/experiments/go2/compare_dr.py"

# ── Evaluate each DR level ──────────────────────────────────────────────────
for DR_LEVEL in baseline moderate aggressive; do
    POLICY="$POLICY_DIR/${DR_LEVEL}.pt"

    if [ ! -f "$POLICY" ]; then
        echo ""
        echo "[SKIP] $POLICY not found — skipping $DR_LEVEL"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "  Evaluating: $DR_LEVEL"
    echo "  Policy:     $POLICY"
    echo "============================================================"

    python "$EVAL_SCRIPT" \
        --policy "$POLICY" \
        --config "$CONFIG" \
        --experiment "$EXPERIMENT" \
        --episodes "$EPISODES" \
        --max_steps "$MAX_STEPS" \
        --output_dir "$RESULTS_DIR/$DR_LEVEL" \
        --label "$DR_LEVEL"
done

# ── Evaluate ActuatorNet policy ───────────────────────────────────────────────
ACTUATOR_NET_POLICY="$POLICY_DIR/actuator_net.pt"

if [ -f "$ACTUATOR_NET_POLICY" ] && [ -f "$ACTUATOR_NET_PATH" ]; then
    echo ""
    echo "============================================================"
    echo "  Evaluating: actuator_net"
    echo "  Policy:     $ACTUATOR_NET_POLICY"
    echo "  Model:      $ACTUATOR_NET_PATH"
    echo "============================================================"

    python "$EVAL_SCRIPT" \
        --policy "$ACTUATOR_NET_POLICY" \
        --config "$CONFIG" \
        --experiment "$EXPERIMENT" \
        --episodes "$EPISODES" \
        --max_steps "$MAX_STEPS" \
        --output_dir "$RESULTS_DIR/actuator_net" \
        --label "actuator_net" \
        --actuator_net \
        --actuator_net_path "$ACTUATOR_NET_PATH"
else
    if [ ! -f "$ACTUATOR_NET_POLICY" ]; then
        echo ""
        echo "[SKIP] $ACTUATOR_NET_POLICY not found — skipping actuator_net"
    fi
    if [ ! -f "$ACTUATOR_NET_PATH" ]; then
        echo ""
        echo "[SKIP] $ACTUATOR_NET_PATH not found — skipping actuator_net"
    fi
fi

# ── Generate comparison plots ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Generating comparison plots"
echo "============================================================"

COMPARE_ARGS=""
for DR_LEVEL in baseline moderate aggressive actuator_net; do
    DIR="$RESULTS_DIR/$DR_LEVEL"
    if [ -f "$DIR/eval_results.json" ]; then
        COMPARE_ARGS="$COMPARE_ARGS --$DR_LEVEL $DIR"
    fi
done

if [ -z "$COMPARE_ARGS" ]; then
    echo "[ERROR] No eval results found. Train and export policies first."
    exit 1
fi

python "$COMPARE_SCRIPT" \
    $COMPARE_ARGS \
    --output_dir "$RESULTS_DIR/comparison"

echo ""
echo "============================================================"
echo "  All done!  Results in: $RESULTS_DIR/comparison/"
echo "============================================================"
