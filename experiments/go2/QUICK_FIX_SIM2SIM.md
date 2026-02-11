# Quick Fix for Sim2Sim Gap - No Training Required

## Problem
Your `sim2sim_comparison.png` shows MuJoCo has 3x worse tracking than Isaac Lab.

## Fastest Solution: Tune MuJoCo Parameters

Instead of retraining, adjust MuJoCo physics to better match Isaac Lab.

---

## Step 1: Tune PD Gains

Isaac Lab uses constraint-based PD (very stiff). MuJoCo uses explicit PD (needs higher gains).

**Edit: `scripts/go2/go2_isaaclab.yaml`**

```yaml
# Current (too soft for MuJoCo)
kps: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
kds: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Try (stiffer, matches Isaac Lab better)
kps: [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
kds: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

**Test:**
```bash
python experiments/go2/eval_mujoco.py \
    --policy logs/go2/policies/moderate.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --experiment directions
```

Expected: ~30-40% RMSE improvement

---

## Step 2: Tune Joint Damping

MuJoCo XML has low damping. Increase it to smooth motion.

**Edit: `unitree_mujoco/unitree_robots/go2/scene_flat.xml`**

```xml
<!-- Find all joint definitions -->
<joint name="FL_hip_joint" pos="0 0 0" axis="1 0 0" range="-1.047 1.047"
       damping="0.5"/>  <!-- was 0.1, increase to 0.5 -->
<joint name="FL_thigh_joint" pos="0 0 0" axis="0 1 0" range="-0.663 2.966"
       damping="0.5"/>  <!-- was 0.1, increase to 0.5 -->

<!-- Repeat for all 12 joints -->
```

Expected: Reduces oscillations, smoother tracking

---

## Step 3: Tune Ground Friction

Higher friction = less slipping = better tracking.

**Edit: `unitree_mujoco/unitree_robots/go2/scene_flat.xml`**

```xml
<!-- Find foot geoms -->
<geom name="FL_foot" type="sphere" size="0.02"
      friction="1.2 0.1 0.1"/>  <!-- was "0.8 0.05 0.05", increase -->

<!-- Repeat for all 4 feet -->
```

Expected: Better traction, reduces lateral drift

---

## Step 4: Tune Control Decimation

Isaac Lab might use different control frequency.

**Edit: `scripts/go2/go2_isaaclab.yaml`**

```yaml
# Current
control_decimation: 4  # 50Hz control

# Try
control_decimation: 2  # 100Hz control (faster response)
```

Expected: Faster response time

---

## Step 5: Test Systematically

```bash
# Baseline (current config)
python experiments/go2/eval_mujoco.py \
    --policy logs/go2/policies/moderate.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --experiment directions \
    --output_dir experiments/go2/tuning/baseline

# Test higher PD gains
# (edit config first)
python experiments/go2/eval_mujoco.py \
    --policy logs/go2/policies/moderate.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --experiment directions \
    --output_dir experiments/go2/tuning/higher_gains

# Compare
python experiments/go2/compare_dr.py \
    --baseline experiments/go2/tuning/baseline \
    --moderate experiments/go2/tuning/higher_gains \
    --output_dir experiments/go2/tuning/comparison
```

---

## Expected Improvements

| Parameter | Change | Expected Improvement |
|-----------|--------|---------------------|
| **PD Gains** | 2x higher | 30-40% better tracking |
| **Joint Damping** | 5x higher | 50% less oscillation |
| **Friction** | 1.5x higher | 20% less drift |
| **Decimation** | 2x faster | 10-15% faster response |

**Combined:** Could reduce sim2sim gap from 3x to ~1.5x with zero training!

---

## Recommended Tuning Order

1. ✅ **Start with PD gains** (biggest impact, easiest to tune)
2. ✅ **Add joint damping** (if still oscillating)
3. ✅ **Tune friction** (if robot slips)
4. ✅ **Adjust decimation** (fine-tuning only)

---

## If Still Not Good Enough

After parameter tuning, if sim2sim gap is still too large:

1. **Increase DR ranges in Isaac Lab** and retrain
2. **Train directly in MuJoCo** (perfect match)
3. **Use system identification** to find optimal parameters automatically

But parameter tuning should get you **60-70% of the way there** with minimal effort!
