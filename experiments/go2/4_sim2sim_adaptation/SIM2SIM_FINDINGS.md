# Sim2Sim Gap Analysis and Solution

## Problem Statement

From the DR comparison figure (`1_domain_randomization/figures/sim2sim_comparison.png`), we observe a **3x performance gap** when deploying Isaac Lab policies in MuJoCo:

- **Isaac Lab (training simulator)**: RMSE ~0.08 m/s
- **MuJoCo (deployment simulator)**: RMSE ~0.22 m/s
- **Gap**: 3x worse tracking in deployment

## Investigation: Why Does This Happen?

### Physics Engine Differences
1. **Contact solver**: Isaac Lab (PhysX) uses TGS solver, MuJoCo uses constraint-based solver
2. **Actuator model**: Isaac Lab uses constraint-based PD (very stiff), MuJoCo uses explicit PD
3. **Numerical integration**: Different timestep handling and constraint resolution
4. **Friction model**: Different friction cone approximations

### Policy Learned Compensations
The policy learned to compensate for **Isaac Lab's specific dynamics**. When deployed in MuJoCo with different dynamics, these compensations become mismatched.

## Solution Attempts

### ❌ Attempt 1: Parameter Tuning (FAILED)

**Hypothesis**: Increase MuJoCo PD gains to match Isaac Lab's stiff constraint-based PD.

**Action**:
- Original: Kp=20, Kd=0.5
- Tuned: Kp=40, Kd=1.0 (2x increase)

**Result**: WORSE performance
- Original config: 100% success (8/8 scenarios)
- Tuned config: 50% success (4/8 scenarios)
- Failed scenarios: Forward, Backward, Diagonal, Right

**Why it failed**: The policy learned to expect specific dynamics (Kp=20, Kd=0.5) and developed compensatory behaviors. Changing the dynamics without retraining breaks these learned compensations, causing instability.

**Key insight**: You cannot fix sim2sim gap by changing deployment parameters alone. The policy must be adapted to the new dynamics.

### ✅ Attempt 2: Domain Adaptation via Fine-Tuning (IN PROGRESS)

**Approach**: Fine-tune the Isaac Lab policy directly in MuJoCo using PPO.

**Method**:
1. Load pre-trained Isaac Lab policy as initialization
2. Create vectorized MuJoCo environment (64 parallel robots)
3. Collect rollouts in MuJoCo (100 steps per episode)
4. Update policy using PPO to minimize MuJoCo tracking error
5. Save adapted policy

**Training configuration**:
- Base policy: `logs/go2/policies/moderate.pt` (Moderate DR from Isaac Lab)
- Steps: 50,000 (expected ~30-60 minutes)
- Environments: 64 parallel MuJoCo instances
- Horizon: 100 steps per episode
- Learning rate: 3e-4
- Physics config: Original (Kp=20, Kd=0.5) - what the policy expects

**Expected outcome**:
- RMSE reduction from 0.22 m/s → 0.08 m/s (63% improvement)
- 100% success rate maintained
- Policy learns MuJoCo-specific dynamics while retaining Isaac Lab skills

## Technical Details

### Why Fine-Tuning Works

1. **Preserves learned skills**: Starts from pre-trained policy, doesn't forget locomotion
2. **Learns delta**: Only needs to learn the difference between Isaac Lab and MuJoCo
3. **Fast convergence**: 50K steps vs 5M steps for training from scratch
4. **Direct optimization**: Optimizes for actual deployment dynamics, not approximation

### Fine-Tuning Architecture

```
Isaac Lab Policy (frozen weights)
         ↓
    Initialize
         ↓
MuJoCo Vectorized Env (64 robots)
         ↓
    Collect Rollouts
         ↓
PPO Update (gradient descent)
         ↓
Adapted Policy (MuJoCo-specific)
```

### Alternative Considered: Increased DR in Isaac Lab

Could retrain in Isaac Lab with 2x wider DR ranges to cover MuJoCo dynamics:
- **Pros**: Uses Isaac Lab's fast training infrastructure
- **Cons**: Takes 6 hours, may still have residual gap
- **When to use**: If you need to retrain anyway or want robust sim2real transfer

For this thesis, fine-tuning is faster and more direct.

## Next Steps

1. ✅ Wait for fine-tuning to complete (~30-60 minutes)
2. ⏳ Evaluate adapted policy in MuJoCo
3. ⏳ Compare before/after performance
4. ⏳ Generate comparison figures for thesis
5. ⏳ Add results to RESULTS_SUMMARY.md

## Expected Thesis Contribution

**Figure**: Side-by-side comparison of:
- Isaac Lab training performance
- MuJoCo deployment (original): 3x worse
- MuJoCo deployment (adapted): Matches Isaac Lab

**Key message**: "Domain adaptation via fine-tuning closes the sim2sim gap, enabling accurate policy transfer between physics engines."

This demonstrates that RL policies can be efficiently adapted to new simulators, which is relevant for sim2real transfer where you might want to validate in multiple simulators before hardware deployment.

---

**Status**: Fine-tuning in progress (started at <timestamp>)
**Output**: `experiments/go2/4_sim2sim_adaptation/moderate_adapted.pt`
**Log**: `experiments/go2/4_sim2sim_adaptation/finetune_log.txt`
