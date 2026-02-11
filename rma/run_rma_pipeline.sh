#!/bin/bash
# Complete RMA training pipeline.
#
# This script runs the full RMA pipeline:
#   Phase 1: Train base policy with privileged encoder
#   Phase 2: Train adaptation module to match encoder
#
# Usage:
#   ./rma/run_rma_pipeline.sh go2
#   ./rma/run_rma_pipeline.sh cassie

set -e

ROBOT=${1:-go2}

echo "=============================================="
echo "RMA TRAINING PIPELINE FOR $ROBOT"
echo "=============================================="

cd "$(dirname "$0")/.."

# Robot-specific settings
if [ "$ROBOT" = "cassie" ]; then
    PHASE1_ITERS=8000
    PHASE2_ITERS=3000
    OBS_DIM=46
else
    PHASE1_ITERS=6000
    PHASE2_ITERS=2000
    OBS_DIM=48
fi

echo ""
echo "Configuration:"
echo "  Robot: $ROBOT"
echo "  Phase 1 iterations: $PHASE1_ITERS"
echo "  Phase 2 iterations: $PHASE2_ITERS"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Train Base Policy with Privileged Encoder
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> PHASE 1: Training Base Policy with Privileged Encoder"
echo "    This trains the policy using ground-truth environment parameters."
echo ""

# TODO: Integrate with Isaac Lab
# python scripts/rsl_rl/train.py \
#     --task Isaac-Velocity-${ROBOT^}-Direct-RMA-v0 \
#     --num_envs 4096 \
#     --max_iterations $PHASE1_ITERS

echo "Phase 1 would run:"
echo "  python scripts/rsl_rl/train.py --task Isaac-Velocity-${ROBOT^}-Direct-RMA-v0 --num_envs 4096"
echo ""

PHASE1_CHECKPOINT="logs/rsl_rl/${ROBOT}/rma_phase1/model_${PHASE1_ITERS}.pt"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Train Adaptation Module
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> PHASE 2: Training Adaptation Module"
echo "    This trains the adaptation module to estimate environment encoding"
echo "    from observation history."
echo ""

# TODO: Implement Phase 2 training
# python rma/train_rma.py \
#     --robot $ROBOT \
#     --phase 2 \
#     --checkpoint $PHASE1_CHECKPOINT

echo "Phase 2 would run:"
echo "  python rma/train_rma.py --robot $ROBOT --phase 2 --checkpoint $PHASE1_CHECKPOINT"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE
# ══════════════════════════════════════════════════════════════════════════════
echo "=============================================="
echo "PIPELINE INFO"
echo "=============================================="
echo ""
echo "RMA Training consists of two phases:"
echo ""
echo "Phase 1: Base Policy Training"
echo "  - Environment provides privileged info (friction, mass, etc.)"
echo "  - Encoder: privileged_info → latent_vector"
echo "  - Policy: (obs, latent_vector) → action"
echo "  - Train with PPO"
echo ""
echo "Phase 2: Adaptation Module Training"
echo "  - Freeze encoder and base policy"
echo "  - Adaptation: obs_history → estimated_latent"
echo "  - Supervised by encoder output (MSE loss)"
echo ""
echo "Deployment:"
echo "  - Use adaptation module instead of encoder"
echo "  - No privileged info needed!"
echo ""
