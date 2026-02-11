# Cassie Policy Export Guide

## Two Export Scripts

### 1. `export_policy_passive.py` - For Passive Ankle Policies (10 actions, 46 obs)

Use this for:
- ✅ Balance standing task
- ✅ Passive locomotion task
- ✅ Enhanced passive locomotion

**Features:**
- Auto-detects dimensions from checkpoint
- Validates passive ankle structure (10 actions, 46 obs)
- Safe for both standing and locomotion policies

**Usage:**

```bash
# Option 1: Point to run directory (uses latest checkpoint)
python scripts/cassie/export_policy_passive.py \
    logs/cassie/rsl_rl/cassie_sim2sim/2026-02-09_18-30-45

# Option 2: Specify checkpoint
python scripts/cassie/export_policy_passive.py \
    --checkpoint logs/cassie/rsl_rl/cassie_sim2sim/<run>/model_19999.pt \
    --output logs/cassie/policies/balance_standing.pt

# Option 3: Checkpoint path only (auto output)
python scripts/cassie/export_policy_passive.py \
    logs/cassie/rsl_rl/cassie_sim2sim/<run>/model_19999.pt
```

**Output:** Creates `policy.pt` in `<run_dir>/exported/` (or custom path)

---

### 2. `export_policy.py` - For Active Ankle Policies (12 actions, 48 obs)

Use this for:
- ⚠️ Old active ankle policies (historical, not recommended)

**Usage:**

```bash
python scripts/cassie/export_policy.py \
    --checkpoint logs/cassie/rsl_rl/cassie/<run>/model_29999.pt \
    --output logs/cassie/policies/active.pt \
    --num_obs 48 \
    --num_actions 12
```

---

## Quick Reference

| Policy Type | Actions | Obs | Script |
|-------------|---------|-----|--------|
| **Balance Standing** | 10 | 46 | `export_policy_passive.py` |
| **Passive Locomotion** | 10 | 46 | `export_policy_passive.py` |
| **Enhanced Passive** | 10 | 46 | `export_policy_passive.py` |
| Active Ankle (old) | 12 | 48 | `export_policy.py` |

---

## Full Workflow Example

### Balance Standing

```bash
# 1. Train
python scripts/rsl_rl/train.py --task Isaac-Balance-Standing-Cassie-v0 --headless

# 2. Export (after 10K-20K iterations)
python scripts/cassie/export_policy_passive.py \
    logs/cassie/rsl_rl/cassie_sim2sim/2026-02-09_18-30-45

# 3. Deploy
python scripts/cassie/deploy_mujoco.py \
    scripts/cassie/cassie_balance_standing.yaml \
    --policy logs/cassie/rsl_rl/cassie_sim2sim/2026-02-09_18-30-45/exported/policy.pt
```

### Enhanced Passive Locomotion

```bash
# 1. Train
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Enhanced-v0 --headless

# 2. Export (after 30K iterations)
python scripts/cassie/export_policy_passive.py \
    logs/cassie/rsl_rl/cassie_sim2sim/2026-02-10_08-15-23

# 3. Deploy (use same config as baseline)
python scripts/cassie/deploy_mujoco.py \
    scripts/cassie/cassie_passive.yaml \
    --policy logs/cassie/rsl_rl/cassie_sim2sim/2026-02-10_08-15-23/exported/policy.pt
```

---

## Troubleshooting

### Error: "size mismatch for mlp.0.weight"

**Problem:** Using wrong export script for policy type

**Solution:**
- If checkpoint has 10 actions → use `export_policy_passive.py`
- If checkpoint has 12 actions → use `export_policy.py`

### Error: "No checkpoints found"

**Problem:** Wrong run directory path

**Solution:** Check the full path
```bash
ls logs/cassie/rsl_rl/cassie_sim2sim/
# Copy the actual run directory name
```

### Error: "No actor weights found"

**Problem:** Checkpoint is in wrong format

**Solution:** Make sure you're using `model_*.pt` not `optimizer.pt` or other files

---

## Output Structure

After export, your run directory will have:

```
logs/cassie/rsl_rl/cassie_sim2sim/2026-02-09_18-30-45/
├── model_10000.pt
├── model_20000.pt
└── exported/
    └── policy.pt    ← This is what you deploy
```

The `policy.pt` is a standalone TorchScript model that can be loaded without Isaac Lab or RSL-RL.
