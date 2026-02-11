#!/bin/bash
# Fine-tune Isaac Lab policy in MuJoCo for better deployment performance

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║        MuJoCo Domain Adaptation - Policy Fine-Tuning                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
POLICY="${1:-logs/go2/policies/moderate.pt}"
CONFIG="${2:-scripts/go2/go2_isaaclab.yaml}"
OUTPUT="${3:-experiments/go2/adapted_policies/moderate_adapted.pt}"
STEPS="${4:-50000}"

echo "Policy:  $POLICY"
echo "Config:  $CONFIG"
echo "Output:  $OUTPUT"
echo "Steps:   $STEPS"
echo ""

# Check policy exists
if [ ! -f "$POLICY" ]; then
    echo "❌ Error: Policy file not found: $POLICY"
    echo ""
    echo "Available policies:"
    ls -1 logs/go2/policies/*.pt 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Run fine-tuning
echo "Starting fine-tuning..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python experiments/go2/finetune_mujoco.py \
    --policy "$POLICY" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --steps "$STEPS" \
    --num_envs 64 \
    --horizon 100 \
    --lr 3e-4

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Fine-tuning complete!"
echo ""
echo "Next steps:"
echo "  1. Test adapted policy:"
echo "     python experiments/go2/eval_mujoco.py --policy $OUTPUT --config $CONFIG --experiment directions"
echo ""
echo "  2. Compare before/after:"
echo "     python experiments/go2/compare_dr.py --baseline <original> --moderate <adapted> --output_dir comparison/"
echo ""
