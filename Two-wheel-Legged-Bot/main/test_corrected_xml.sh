#!/bin/bash
# Test the corrected XML with different PD scales

echo "================================================================================"
echo "TESTING CORRECTED FLAMINGO XML WITH DIFFERENT PD SCALES"
echo "================================================================================"
echo ""

POLICY="logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/model_4999.pt"
XML="assets/flamingo_correct.xml"
INIT_HEIGHT="0.5562"
DURATION="15.0"

echo "Policy: $POLICY"
echo "XML: $XML"
echo "Init height: $INIT_HEIGHT"
echo "Duration: ${DURATION}s"
echo ""

# Test 1: Standing balance with pd_scale=1.0
echo "================================================================================"
echo "TEST 1: Standing Balance (pd_scale=1.0)"
echo "================================================================================"
python scripts/transfer_flamingo_sim2sim.py \
    --policy $POLICY \
    --xml $XML \
    --init_height $INIT_HEIGHT \
    --pd_scale 1.0 \
    --cmd_vx 0.0 \
    --cmd_wz 0.0 \
    --duration $DURATION

echo ""
echo "Press Enter to continue to next test..."
read

# Test 2: Standing balance with pd_scale=2.0
echo "================================================================================"
echo "TEST 2: Standing Balance (pd_scale=2.0)"
echo "================================================================================"
python scripts/transfer_flamingo_sim2sim.py \
    --policy $POLICY \
    --xml $XML \
    --init_height $INIT_HEIGHT \
    --pd_scale 2.0 \
    --cmd_vx 0.0 \
    --cmd_wz 0.0 \
    --duration $DURATION

echo ""
echo "Press Enter to continue to next test..."
read

# Test 3: Standing balance with pd_scale=0.5
echo "================================================================================"
echo "TEST 3: Standing Balance (pd_scale=0.5)"
echo "================================================================================"
python scripts/transfer_flamingo_sim2sim.py \
    --policy $POLICY \
    --xml $XML \
    --init_height $INIT_HEIGHT \
    --pd_scale 0.5 \
    --cmd_vx 0.0 \
    --cmd_wz 0.0 \
    --duration $DURATION

echo ""
echo "Press Enter to continue to next test..."
read

# Test 4: Forward motion with pd_scale=1.0
echo "================================================================================"
echo "TEST 4: Forward Motion 0.5 m/s (pd_scale=1.0)"
echo "================================================================================"
python scripts/transfer_flamingo_sim2sim.py \
    --policy $POLICY \
    --xml $XML \
    --init_height $INIT_HEIGHT \
    --pd_scale 1.0 \
    --cmd_vx 0.5 \
    --cmd_wz 0.0 \
    --duration $DURATION

echo ""
echo "Press Enter to continue to next test..."
read

# Test 5: Forward motion with pd_scale=2.0
echo "================================================================================"
echo "TEST 5: Forward Motion 0.5 m/s (pd_scale=2.0)"
echo "================================================================================"
python scripts/transfer_flamingo_sim2sim.py \
    --policy $POLICY \
    --xml $XML \
    --init_height $INIT_HEIGHT \
    --pd_scale 2.0 \
    --cmd_vx 0.5 \
    --cmd_wz 0.0 \
    --duration $DURATION

echo ""
echo "================================================================================"
echo "ALL TESTS COMPLETE!"
echo "================================================================================"
