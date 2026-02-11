# Go2 Sim2Sim Improvement Guide

This guide provides step-by-step instructions for improving ActuatorNet and completing RMA Phase 2 for enhanced sim-to-sim transfer performance.

---

## 📊 Current Status Summary

### ✅ What Works Now
- **Moderate DR**: 100% success on flat terrain, 0.108 m/s RMSE (BEST BASELINE)
- **ActuatorNet**: 100% success, reduced oscillation, but worse forward tracking (0.156 m/s)
- **RMA Phase 1**: Trained and exported, but not deployable (62.5% success with dummy obs)

### 🎯 Improvement Goals
1. **ActuatorNet v2**: Match Moderate DR's tracking accuracy (0.108 m/s) while keeping oscillation reduction
2. **RMA Phase 2**: Complete adaptation module to enable online dynamics estimation (target ≥95% success)

---

## Part 1: ActuatorNet v2 — Enhanced Training

### Background
Current ActuatorNet (Experiment 1C) successfully reduces oscillation but has worse forward velocity tracking than Moderate DR (0.156 vs 0.108 m/s). The likely causes:
1. **Training data mismatch**: Collected from Moderate DR policy, not perfectly aligned with MuJoCo dynamics
2. **Network capacity**: [128, 64, 32] may be insufficient for complex joint dynamics
3. **Feature engineering**: Current features (q_des, q_cur, q_vel, joint_id) may miss important context

### Improvement Strategy

#### 1.1 Enhanced Data Collection

**Goal**: Collect 2M samples (vs 1.49M current) with better coverage of the dynamics space.

```bash
# Run enhanced data collection (more episodes, longer duration)
python actuator_net/collect_training_data.py \
    --policy logs/go2/rsl_rl/go2_sim2sim_moderate/*/exported/policy.pt \
    --xml unitree_mujoco/unitree_robots/go2/scene_flat.xml \
    --config scripts/go2/go2_isaaclab.yaml \
    --num_episodes 1000 \
    --episode_length 1000 \
    --output actuator_net/data/go2/training_data_v2.csv \
    --seed 42
```

**Expected output**: `training_data_v2.csv` (~200 MB, 2M samples)

**Key changes**:
- More episodes (500 → 1000) for better randomization coverage
- Longer episodes (500 → 1000 steps) to capture steady-state behavior
- Explicit seed for reproducibility

#### 1.2 Feature Engineering Enhancements

**Current features** (15-dim):
```
[q_des, q_cur, q_vel, one_hot(joint_id)]  # 1 + 1 + 1 + 12 = 15
```

**Enhanced features** (22-dim):
```python
# Add these features to capture dynamics context:
features = [
    q_des,                    # Desired position (1)
    q_cur,                    # Current position (1)
    q_vel,                    # Current velocity (1)
    q_acc,                    # Estimated acceleration (finite difference) (1)
    q_des - q_cur,           # Position error (1)
    base_lin_vel_x,          # Body velocity context (1)
    base_ang_vel_z,          # Yaw rate context (1)
    contact_forces,          # Contact state (4 legs → 4)
    *one_hot(joint_id, 12)   # Joint identity (12)
]
# Total: 1+1+1+1+1+1+1+4+12 = 23 dims
```

**Rationale**:
- **Acceleration**: Captures momentum/inertia effects
- **Position error**: Direct feedback signal
- **Body velocity**: Locomotion context (torque needs differ at high speed)
- **Contact forces**: Ground reaction forces affect joint loading

**Modify**: `actuator_net/train_actuator_net.py` to extract and include these features.

#### 1.3 Network Architecture v2

**Current**: [128, 64, 32] with LayerNorm + ELU + Dropout(0.1)

**Proposed v2**: [256, 256, 128, 64] with Residual connections + LayerNorm + SiLU

```python
class ActuatorNetV2(nn.Module):
    def __init__(self, input_dim=23):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        # Encoder: 23 → 256
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)

        # Residual block 1: 256 → 256
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)

        # Residual block 2: 256 → 128
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)

        # Output: 128 → 1
        self.fc5 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.SiLU()  # Smoother than ELU

    def forward(self, x):
        x = self.input_norm(x)

        # Encoder
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.dropout(x)

        # Residual block 1
        identity = x
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = self.ln3(self.fc3(x))
        x = self.activation(x + identity)  # Residual connection

        # Residual block 2
        x = self.activation(self.ln4(self.fc4(x)))
        x = self.dropout(x)

        # Output
        x = self.activation(self.fc5(x))
        x = self.fc_out(x)
        return x
```

**Benefits**:
- Residual connections: Help gradient flow and enable deeper networks
- SiLU activation: Smoother gradients than ELU
- Larger capacity: 256 → 256 → 128 captures complex dynamics

#### 1.4 Training Configuration v2

```python
# Training hyperparameters
config = {
    'batch_size': 512,          # Larger batch (was 256)
    'epochs': 150,              # More epochs (was 100)
    'learning_rate': 5e-4,      # Lower LR for stability (was 1e-3)
    'weight_decay': 1e-4,       # Stronger regularization (was 1e-5)
    'scheduler': 'CosineAnnealingWarmRestarts',  # Better than ReduceLROnPlateau
    'T_0': 10,                  # Cosine cycle length
    'T_mult': 2,                # Cycle length multiplier
    'early_stopping_patience': 20,
}
```

#### 1.5 Training Script

```bash
python actuator_net/train_actuator_net_v2.py \
    --data actuator_net/data/go2/training_data_v2.csv \
    --output_dir actuator_net/models/go2_v2 \
    --architecture v2 \
    --features enhanced \
    --epochs 150 \
    --batch_size 512 \
    --lr 5e-4 \
    --patience 20
```

**Expected outputs**:
- `actuator_net/models/go2_v2/actuator_net.pt` (checkpoint)
- `actuator_net/models/go2_v2/actuator_net_jit.pt` (TorchScript for deployment)
- `actuator_net/models/go2_v2/training_log.csv` (loss curves)
- `actuator_net/models/go2_v2/scaler.pkl` (StandardScaler for features)

#### 1.6 Evaluation

```bash
# Evaluate ActuatorNet v2
python experiments/go2/eval_mujoco.py \
    --policy logs/go2/rsl_rl/go2_sim2sim_moderate/*/exported/policy.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --experiment directions \
    --episodes 5 \
    --max_steps 500 \
    --output_dir experiments/go2/results/actuator_net_v2 \
    --label "ActuatorNet-v2" \
    --actuator_net \
    --actuator_net_path actuator_net/models/go2_v2/actuator_net_jit.pt

# Compare v1 vs v2
python experiments/go2/compare_dr.py \
    --baseline experiments/go2/results/moderate \
    --actuator_net experiments/go2/results/actuator_net \
    --actuator_net_v2 experiments/go2/results/actuator_net_v2 \
    --output_dir experiments/go2/results/comparison_actuator_net_v2
```

**Success criteria**:
- Forward RMSE ≤ 0.12 m/s (close to Moderate DR's 0.108)
- Success rate = 100% on all scenarios
- Oscillation amplitude < Moderate DR baseline

---

## Part 2: RMA Phase 2 — Adaptation Module

### Background
RMA Phase 1 policy is trained and works in Isaac Lab (with privileged observations), but cannot deploy to MuJoCo without the adaptation module. Phase 2 trains this module to estimate the 4 privileged parameters from observation history.

### Implementation Workflow

#### 2.1 Data Collection for Adaptation Module

**Goal**: Collect 100K (observation_history, privileged_info) pairs from the trained Phase 1 policy.

**Option A: Isaac Lab Collection (Faster, Cleaner)**

```bash
# Collect from Isaac Lab training environment
python experiments/go2/rma/collect_adaptation_data_isaaclab.py \
    --checkpoint logs/go2/rsl_rl/go2_rma/2026-02-09_20-37-48/model_4999.pt \
    --num_samples 100000 \
    --num_envs 1024 \
    --output experiments/go2/rma/adaptation_data_isaaclab.npz
```

**Expected runtime**: ~30 minutes with 1024 parallel envs

**Option B: MuJoCo Collection (Preferred for Sim-to-Sim)**

```bash
# Collect from MuJoCo with randomized physics
python experiments/go2/rma/collect_adaptation_data_mujoco.py \
    --policy logs/go2/rsl_rl/go2_rma/2026-02-09_20-37-48/exported/policy.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --num_samples 50000 \
    --num_workers 8 \
    --output experiments/go2/rma/adaptation_data_mujoco.npz \
    --randomize_physics \
    --friction_range 0.5 1.5 \
    --mass_range -0.5 1.0 \
    --kp_range 0.8 1.2 \
    --kd_range 0.8 1.2
```

**Expected runtime**: ~2 hours with 8 parallel workers

**Why MuJoCo collection is better**:
- Captures MuJoCo-specific observation patterns (contact dynamics, numerical artifacts)
- Avoids distribution shift between Isaac Lab (training) and MuJoCo (deployment)
- Adaptation module trains on the same data distribution it will see during deployment

**Data format** (`adaptation_data.npz`):
```python
{
    'obs_history': (N, 50, 48),  # N samples, 50-step history, 48-dim obs
    'privileged': (N, 4),         # Ground-truth [friction, mass, kp, kd]
    'commands': (N, 3),           # Velocity commands [vx, vy, wz]
    'episode_id': (N,),           # Episode identifier for train/val split
}
```

#### 2.2 Adaptation Module Architecture

**Baseline Architecture** (from exam2.tex):
```
Input: obs_history (50 × 48 = 2400) → Flatten → [256, 128] → Output: z_hat (4)
```

**Proposed Enhanced Architectures**:

**Option 1: Temporal CNN (Recommended)**
```python
class AdaptationModuleCNN(nn.Module):
    def __init__(self, obs_dim=48, history_len=50, latent_dim=4):
        super().__init__()
        # Input: (batch, history_len, obs_dim) = (B, 50, 48)

        # 1D CNN over time axis
        self.conv1 = nn.Conv1d(obs_dim, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global pooling

        # MLP decoder
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, latent_dim)

        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, obs_history):
        # obs_history: (B, T, obs_dim) → (B, obs_dim, T) for Conv1d
        x = obs_history.transpose(1, 2)

        x = self.activation(self.ln1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        x = self.dropout(x)
        x = self.activation(self.ln2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
        x = self.pool(x).squeeze(-1)  # (B, 256)

        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x  # (B, latent_dim)
```

**Option 2: Transformer (Most Powerful)**
```python
class AdaptationModuleTransformer(nn.Module):
    def __init__(self, obs_dim=48, history_len=50, latent_dim=4):
        super().__init__()
        self.embed_dim = 128

        # Input projection
        self.input_proj = nn.Linear(obs_dim, self.embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, obs_history):
        # obs_history: (B, T, obs_dim)
        x = self.input_proj(obs_history)  # (B, T, embed_dim)
        x = self.transformer(x)           # (B, T, embed_dim)
        x = x.mean(dim=1)                 # Global average pooling → (B, embed_dim)
        x = self.decoder(x)               # (B, latent_dim)
        return x
```

#### 2.3 Training Script

```bash
python experiments/go2/rma/train_adaptation_module.py \
    --data experiments/go2/rma/adaptation_data_mujoco.npz \
    --architecture transformer \
    --history_length 50 \
    --latent_dim 4 \
    --batch_size 256 \
    --epochs 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --patience 15 \
    --output_dir experiments/go2/rma/adaptation_module
```

**Training configuration**:
```python
config = {
    'loss_fn': 'mse',  # or 'weighted_mse' with task-specific weights
    'optimizer': 'adamw',
    'scheduler': 'cosine_annealing',
    'train_val_split': 0.8,
    'normalize_latent': True,  # Normalize z to [-1, 1]
    'clip_grad_norm': 1.0,
}
```

**Expected outputs**:
- `adaptation_module.pt` (PyTorch checkpoint)
- `adaptation_jit.pt` (TorchScript for deployment)
- `training_curves.png` (loss plots)
- `validation_metrics.json` (MSE per latent dimension)

#### 2.4 Deployment Integration

**Deploy script**: `experiments/go2/rma/deploy_rma_phase2.py`

```python
# Load models
policy = torch.jit.load('logs/go2/rsl_rl/go2_rma/.../exported/policy.pt')
adaptation_module = torch.jit.load('experiments/go2/rma/adaptation_jit.pt')

# Observation history buffer
obs_history = collections.deque(maxlen=50)

for step in range(max_steps):
    # 1. Build 48-dim observation from MuJoCo
    obs = build_observation(data)
    obs_history.append(obs)

    # 2. Adaptation module: predict latent z_hat
    if len(obs_history) == 50:
        obs_tensor = torch.stack(list(obs_history)).unsqueeze(0)  # (1, 50, 48)
        z_hat = adaptation_module(obs_tensor)  # (1, 4)
    else:
        z_hat = torch.zeros(1, 4)  # Use zeros during warmup

    # 3. Policy: concatenate obs + z_hat
    obs_with_latent = torch.cat([obs, z_hat], dim=-1)  # (1, 52)
    action = policy(obs_with_latent)

    # 4. Apply action and step physics
    ...
```

**Key details**:
- First 50 steps (1 second): buffer filling, use dummy z_hat
- After 50 steps: use predicted z_hat
- Latent estimate stabilizes after ~1 second

#### 2.5 Evaluation

```bash
# Evaluate RMA Phase 2 (full pipeline)
python experiments/go2/eval_mujoco_rma_phase2.py \
    --policy logs/go2/rsl_rl/go2_rma/2026-02-09_20-37-48/exported/policy.pt \
    --adaptation experiments/go2/rma/adaptation_jit.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --experiment directions \
    --episodes 5 \
    --max_steps 500 \
    --output_dir experiments/go2/results/rma_phase2 \
    --label "RMA-Phase2"

# Compare all approaches
python experiments/go2/compare_dr.py \
    --baseline experiments/go2/results/baseline \
    --moderate experiments/go2/results/moderate \
    --actuator_net experiments/go2/results/actuator_net \
    --rma_phase1 experiments/go2/results/rma_flat \
    --rma_phase2 experiments/go2/results/rma_phase2 \
    --output_dir experiments/go2/results/final_comparison
```

**Success criteria**:
- Success rate ≥ 95% (vs 62.5% Phase 1 with dummy obs)
- Forward RMSE ≤ 0.15 m/s
- Backward/Right scenarios succeed (failed in Phase 1)
- Adaptation convergence time < 0.5s

#### 2.6 Debugging Tips

If Phase 2 results are poor:

**Issue 1: Adaptation doesn't converge**
- Check validation MSE — should be < 0.01 per dimension
- Visualize predicted vs true latent: `plot_latent_predictions(val_data)`
- Try longer history (50 → 100 steps)

**Issue 2: Policy behavior doesn't change**
- Verify z_hat is non-zero: print latent values during deployment
- Check if policy is sensitive to latent: sweep z_hat values and observe behavior change
- Ensure Phase 1 policy properly uses privileged info

**Issue 3: High training loss**
- Reduce learning rate (1e-3 → 5e-4)
- Increase model capacity (transformer layers 2 → 4)
- Check data quality: visualize observation histories

---

## Part 3: Expected Results Summary

### ActuatorNet v2
| Metric | Moderate DR (baseline) | ActuatorNet v1 | ActuatorNet v2 (target) |
|--------|----------------------|----------------|------------------------|
| Success rate | 100% | 100% | 100% |
| Forward RMSE | 0.108 m/s | 0.156 m/s | ≤ 0.12 m/s |
| Oscillation amplitude | High | Low ✅ | Low ✅ |

### RMA Phase 2
| Metric | Phase 1 (dummy obs) | Phase 2 (target) |
|--------|-------------------|-----------------|
| Flat success | 62.5% | ≥ 95% |
| Rough success | 87.5% | ≥ 95% |
| Forward RMSE | 0.482 m/s | ≤ 0.15 m/s |
| Backward success | 0% | 100% |
| Right success | 0% | 100% |

---

## Part 4: Time Estimates

### ActuatorNet v2
- Data collection: 2 hours
- Training: 1 hour (150 epochs)
- Evaluation: 30 minutes
- **Total**: ~4 hours

### RMA Phase 2
- Data collection (MuJoCo): 2 hours
- Adaptation training: 2 hours (100 epochs)
- Deployment integration: 1 hour
- Evaluation: 30 minutes
- **Total**: ~6 hours

### Combined Improvements
- **Total time**: ~10 hours (1-2 days with debugging)

---

## Part 5: Quick Start Commands

### Prerequisites
```bash
# Verify all dependencies
python -c "import torch; import mujoco; import numpy; print('✅ Ready')"

# Check existing results
ls -lh experiments/go2/results/*/eval_results.json
```

### Run ActuatorNet v2
```bash
bash experiments/go2/prepare_actuatornet_phase2.sh
```

### Run RMA Phase 2
```bash
bash experiments/go2/prepare_rma_phase2.sh
```

### Generate Final Comparison
```bash
python experiments/go2/generate_final_comparison.py \
    --include baseline moderate aggressive actuator_net actuator_net_v2 rma_phase2 \
    --output_dir experiments/go2/results/final_thesis_comparison
```

---

## Part 6: When to Stop

You don't need to complete both improvements. Here's a decision matrix:

| Scenario | Recommendation |
|----------|---------------|
| **Tight deadline** | Skip both. Moderate DR (0.108 RMSE, 100% success) is already excellent. |
| **Want better tracking** | Do ActuatorNet v2 only (~4 hours). |
| **Want adaptive policy** | Do RMA Phase 2 only (~6 hours). |
| **Want comprehensive comparison** | Do both (~10 hours). |
| **Already have good results** | Write the thesis! These are optional enhancements. |

**Reality check**: Your current Moderate DR baseline is already publishable (100% success, 0.108 m/s RMSE). The improvements are incremental, not game-changing.

---

## Contact & Support

If you encounter issues:
1. Check `experiments/go2/logs/` for error messages
2. Verify model files exist with correct shapes
3. Test with minimal example first (1 episode, 100 steps)
4. Compare outputs with expected formats in this guide

Good luck! 🚀
