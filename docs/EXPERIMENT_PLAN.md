# Sim-to-Sim Transfer Experiment Plan

## Overview

This document outlines experiments to comprehensively evaluate sim-to-sim transfer from Isaac Lab to MuJoCo using the trained Go2 policy.

**Working Policy:** `logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt`

---

## Experiment Categories

### 1. Transfer Validation Experiments

| Exp ID | Name | Purpose | Commands |
|--------|------|---------|----------|
| E1.1 | Forward Walking | Baseline transfer test | vx=0.5, vy=0, wz=0 |
| E1.2 | Backward Walking | Reverse locomotion | vx=-0.5, vy=0, wz=0 |
| E1.3 | Lateral Left | Sideways motion | vx=0, vy=0.5, wz=0 |
| E1.4 | Lateral Right | Sideways motion | vx=0, vy=-0.5, wz=0 |
| E1.5 | Turn Left | Rotation | vx=0, vy=0, wz=0.5 |
| E1.6 | Turn Right | Rotation | vx=0, vy=0, wz=-0.5 |
| E1.7 | Diagonal | Combined motion | vx=0.5, vy=0.5, wz=0 |
| E1.8 | Circle Walk | Complex trajectory | vx=0.5, vy=0, wz=0.3 |

```bash
# Run all transfer validation experiments
python go2_experiment/evaluate_transfer.py --model logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt --episodes 10
```

---

### 2. Speed Range Experiments

| Exp ID | Name | Purpose | Commands |
|--------|------|---------|----------|
| E2.1 | Slow Walk | Low speed | vx=0.25, vy=0, wz=0 |
| E2.2 | Normal Walk | Medium speed | vx=0.5, vy=0, wz=0 |
| E2.3 | Fast Walk | High speed | vx=1.0, vy=0, wz=0 |
| E2.4 | Sprint | Max speed | vx=1.5, vy=0, wz=0 |

**Expected Result:** Policy should work better at trained speeds (0-1 m/s) and degrade at extremes.

---

### 3. Ablation Experiments

#### 3.1 Domain Randomization Ablation

| Exp ID | Training | Transfer | Purpose |
|--------|----------|----------|---------|
| E3.1 | With DR | MuJoCo | Baseline (current) |
| E3.2 | Without DR | MuJoCo | Show DR importance |

```bash
# Need to train No-DR policy first
python scripts/rsl_rl/train.py --task Isaac-Velocity-Go2-Direct-NoDR-v0 --num_envs 4096 --max_iterations 6000
```

#### 3.2 Training Duration Ablation

| Exp ID | Iterations | Transfer | Purpose |
|--------|------------|----------|---------|
| E3.3 | 2000 | MuJoCo | Early stopping |
| E3.4 | 4000 | MuJoCo | Mid training |
| E3.5 | 6000 | MuJoCo | Current (best) |
| E3.6 | 10000 | MuJoCo | Extended training |

Models available:
- `model_2000.pt` - Early
- `model_4000.pt` - Mid
- `model_6000.pt` - Current (best)

---

### 4. Robustness Experiments

#### 4.1 PD Gain Sensitivity

| Exp ID | Kp Scale | Kd Scale | Purpose |
|--------|----------|----------|---------|
| E4.1 | 0.5× | 1.0× | Lower stiffness |
| E4.2 | 1.0× | 1.0× | Baseline |
| E4.3 | 1.5× | 1.0× | Higher stiffness |
| E4.4 | 1.0× | 0.5× | Lower damping |
| E4.5 | 1.0× | 1.5× | Higher damping |

```bash
# Test with different PD gains
python go2_experiment/test_go2_transfer.py --model <path> --pd_scale 0.5
python go2_experiment/test_go2_transfer.py --model <path> --pd_scale 1.5
```

#### 4.2 Initial Condition Perturbation

| Exp ID | Perturbation | Purpose |
|--------|--------------|---------|
| E4.6 | Height +0.1m | Start higher |
| E4.7 | Height -0.1m | Start lower |
| E4.8 | Roll ±10° | Tilted start |
| E4.9 | Velocity push | Moving start |

---

### 5. Cross-Simulator Comparison

| Exp ID | Source | Target | Purpose |
|--------|--------|--------|---------|
| E5.1 | Isaac Lab | Isaac Lab | Same-sim baseline |
| E5.2 | Isaac Lab | MuJoCo | Sim-to-sim transfer |
| E5.3 | Isaac Lab | MuJoCo (different XML) | Model sensitivity |

---

## Metrics to Record

### Primary Metrics
| Metric | Description | Good Value |
|--------|-------------|------------|
| Success Rate | % episodes > 400 steps | ≥ 80% |
| Mean Episode Length | Average survival | ≥ 400 steps |
| RMSE vx | Forward velocity error | < 0.1 m/s |
| RMSE vy | Lateral velocity error | < 0.1 m/s |
| RMSE wz | Angular velocity error | < 0.2 rad/s |

### Secondary Metrics
| Metric | Description | Purpose |
|--------|-------------|---------|
| Mean Reward | Cumulative reward | Overall performance |
| Height Variance | Body height stability | Gait smoothness |
| Action Rate L2 | Action smoothness | Motion quality |
| Torque Mean | Average joint torque | Energy efficiency |

---

## Experiment Schedule

### Phase 1: Baseline (Current)
- [x] E1.1 Forward walking (Done: 100% at vx=0.5)
- [x] Full evaluation with 5 commands (Done: 40% overall)
- [ ] Record Isaac Lab baseline performance

### Phase 2: Comprehensive Transfer
- [ ] E1.1-E1.8 All directions
- [ ] E2.1-E2.4 Speed range
- [ ] Generate comparison plots

### Phase 3: Ablation Studies
- [ ] Train No-DR policy (E3.2)
- [ ] Compare with DR policy
- [ ] Test different training durations (E3.3-E3.6)

### Phase 4: Robustness
- [ ] PD gain sweep (E4.1-E4.5)
- [ ] Initial condition tests (E4.6-E4.9)

---

## Expected Outcomes

### Hypothesis 1: DR Improves Transfer
- **Prediction:** DR policy achieves 2-5× higher success rate than No-DR
- **Reference:** Stanford CS224R showed +495 vs -849 reward difference

### Hypothesis 2: Forward Motion Transfers Best
- **Prediction:** Forward/backward > lateral > turning
- **Reason:** Training distribution focuses on forward motion

### Hypothesis 3: Training Duration Matters
- **Prediction:** 6000 iterations optimal, diminishing returns after
- **Test:** Compare model_2000, model_4000, model_6000

---

## Commands Reference

```bash
# Basic evaluation
python go2_experiment/evaluate_transfer.py \
    --model logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt \
    --episodes 10

# Single command test
python go2_experiment/test_go2_transfer.py \
    --model logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt \
    --vx 0.5 --vy 0.0 --wz 0.0

# Visualization
python go2_experiment/test_go2_transfer.py \
    --model logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt \
    --visualize

# Generate plots
python go2_experiment/plot_transfer_results.py
```

---

## File Locations

| File | Purpose |
|------|---------|
| `go2_experiment/evaluate_transfer.py` | Comprehensive evaluation |
| `go2_experiment/test_go2_transfer.py` | Quick tests & visualization |
| `go2_experiment/plot_transfer_results.py` | Publication plots |
| `go2_experiment/transfer_metrics.json` | Saved metrics |
| `go2_experiment/transfer_summary.png` | Summary figure |
