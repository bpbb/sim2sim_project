# ActuatorNet for Go2 Sim-to-Sim Transfer

This document describes the ActuatorNet approach for improving sim-to-sim transfer from Isaac Lab (PhysX) to MuJoCo for the Unitree Go2 quadruped robot.

## Table of Contents

1. [Overview](#overview)
2. [The Problem: Actuator Dynamics Mismatch](#the-problem-actuator-dynamics-mismatch)
3. [The Solution: ActuatorNet](#the-solution-actuatornet)
4. [Architecture](#architecture)
5. [Data Collection](#data-collection)
6. [Training](#training)
7. [Deployment](#deployment)
8. [Evaluation](#evaluation)
9. [Results](#results)
10. [File Structure](#file-structure)
11. [Usage Guide](#usage-guide)

---

## Overview

**ActuatorNet** is a neural network that learns the mapping from desired joint positions to actual torques, capturing the complex actuator dynamics that differ between physics simulators.

### Key Insight

When transferring a policy trained in Isaac Lab (PhysX) to MuJoCo, the robot often falls immediately. This happens because:

1. **Isaac Lab** uses implicit PD control with specific actuator models
2. **MuJoCo** has different contact dynamics and actuator behavior
3. A simple PD controller `τ = Kp(q_des - q) - Kd*q̇` doesn't capture these differences

**ActuatorNet bridges this gap** by learning the actual torque response from MuJoCo's physics, allowing policies trained in Isaac Lab to work in MuJoCo.

---

## The Problem: Actuator Dynamics Mismatch

### Why Policies Fail to Transfer

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRANSFER FAILURE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Isaac Lab (Training)          MuJoCo (Deployment)              │
│  ┌─────────────────┐           ┌─────────────────┐              │
│  │ DCMotor Model   │           │ PD Controller   │              │
│  │ - saturation    │    ≠      │ - linear model  │              │
│  │ - velocity dep  │           │ - no saturation │              │
│  │ - friction      │           │ - different     │              │
│  └─────────────────┘           │   friction      │              │
│           │                    └─────────────────┘              │
│           ▼                            │                        │
│    Policy learns                       ▼                        │
│    Isaac dynamics              Wrong torques                    │
│           │                    applied                          │
│           ▼                            │                        │
│    Works in Isaac              ▼       ▼                        │
│                                Robot falls!                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Specific Differences

| Aspect | Isaac Lab (PhysX) | MuJoCo |
|--------|-------------------|--------|
| PD Gains | Kp=20, Kd=0.5 (DCMotor) | Kp=20, Kd=0.5 (position actuator) |
| Torque Computation | Implicit in physics step | Explicit `ctrl` input |
| Saturation | Effort limit (23.5 Nm) | Depends on configuration |
| Contact Dynamics | PhysX solver | MuJoCo convex solver |
| Integration | GPU-accelerated | CPU (typically) |

---

## The Solution: ActuatorNet

Instead of using a fixed PD controller, we train a neural network to predict the correct torques for MuJoCo:

```
┌─────────────────────────────────────────────────────────────────┐
│                   ACTUATORNET SOLUTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Policy Output          ActuatorNet              MuJoCo         │
│  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐    │
│  │ q_desired   │──────▶│ Neural Net  │──────▶│ τ (torque)  │    │
│  │ (12 joints) │       │             │       │ (12 joints) │    │
│  └─────────────┘       │ Input:      │       └─────────────┘    │
│                        │ - q_des     │              │           │
│  Current State         │ - q_current │              ▼           │
│  ┌─────────────┐       │ - q̇         │       Robot walks!      │
│  │ q_current   │──────▶│ - joint_id  │                          │
│  │ q̇ (vel)     │       └─────────────┘                          │
│  └─────────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Learns MuJoCo dynamics**: The network is trained on data from MuJoCo simulation
2. **Captures nonlinearities**: Can model friction, saturation, velocity-dependent effects
3. **Joint-specific**: One-hot encoding allows learning per-joint characteristics
4. **End-to-end**: Directly maps (q_des, q, q̇) → τ

---

## Architecture

### Network Structure

```python
ActuatorNet(
    input_dim=15,  # 3 features + 12 one-hot
    hidden_dims=[128, 64, 32],
    output_dim=1   # torque for single joint
)
```

### Input Features (per joint)

| Feature | Dimension | Description |
|---------|-----------|-------------|
| `pos_desired` | 1 | Target joint position (rad) |
| `pos_current` | 1 | Current joint position (rad) |
| `velocity` | 1 | Current joint velocity (rad/s) |
| `joint_onehot` | 12 | One-hot encoding of joint ID |
| **Total** | **15** | |

### Layer Details

```
Input (15)
    → Linear(15, 128) → LayerNorm → ELU → Dropout(0.1)
    → Linear(128, 64) → LayerNorm → ELU → Dropout(0.1)
    → Linear(64, 32)  → LayerNorm → ELU → Dropout(0.1)
    → Linear(32, 1)
Output (1 torque)
```

### Feature Scaling

All input features are normalized using `StandardScaler` from scikit-learn:
- Fitted on training data
- Saved as `feature_scaler.pkl`
- Applied at inference time

---

## Data Collection

### Process

1. **Load MuJoCo model** with Go2 robot
2. **Run random motions** to explore joint space
3. **Run systematic sweeps** for each joint
4. **Record tuples**: `(q_des, q_current, velocity, torque)`

### Data Collection Script

```bash
python actuator_net/collect_mujoco_data.py \
    --robot go2 \
    --output actuator_net/data/go2/actuator_data.csv \
    --episodes 100 \
    --sweeps 10
```

### Collection Strategy

| Method | Description | Purpose |
|--------|-------------|---------|
| **Random Motion** | Random target offsets, 100 episodes × 500 steps | Cover typical operating range |
| **Joint Sweeps** | Sweep each joint from -0.8 to +0.8 rad | Systematic coverage |

### Data Format (CSV)

```csv
joint_idx,joint_name,pos_desired,pos_current,velocity,torque
0,FL_hip_joint,0.15,0.12,0.3,0.6
0,FL_hip_joint,0.16,0.13,0.2,0.58
...
```

### Data Statistics (typical)

- **Total samples**: ~600,000 (100 episodes × 500 steps × 12 joints)
- **Position range**: -0.8 to +0.8 rad
- **Velocity range**: -5 to +5 rad/s
- **Torque range**: -20 to +20 Nm

---

## Training

### Training Script

```bash
python actuator_net/train_actuator_net.py \
    --data actuator_net/data/go2/actuator_data.csv \
    --output actuator_net/models/go2 \
    --epochs 100 \
    --batch_size 256 \
    --num_joints 12
```

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | lr=1e-3, weight_decay=1e-5 |
| Scheduler | ReduceLROnPlateau | patience=10, factor=0.5 |
| Loss | MSE | Mean Squared Error |
| Train/Val Split | 80/20 | Random split |
| Epochs | 100 | With early stopping |

### Training Output

```
actuator_net/models/go2/
├── actuator_net_best.pt      # Best validation loss (PyTorch state dict)
├── actuator_net_jit.pt       # TorchScript for deployment
├── actuator_net_final.pt     # Final epoch model
└── feature_scaler.pkl        # StandardScaler for inputs
```

### Expected Training Curve

```
Epoch  10/100 - Train Loss: 0.085000, Val Loss: 0.082000
Epoch  20/100 - Train Loss: 0.045000, Val Loss: 0.043000
Epoch  50/100 - Train Loss: 0.012000, Val Loss: 0.011500
Epoch 100/100 - Train Loss: 0.008000, Val Loss: 0.007800
```

---

## Deployment

### Transfer Script

```bash
python scripts/transfer_go2_mujoco_actuatornet.py \
    --policy logs/rsl_rl/go2/v7/model_200.pt \
    --actuator_net actuator_net/models/go2/actuator_net_jit.pt \
    --scaler actuator_net/models/go2/feature_scaler.pkl \
    --cmd 0.5 0.0 0.0
```

### Integration Code

```python
class ActuatorNetWrapper:
    def __init__(self, model_path, scaler_path, num_joints=12):
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.scaler = joblib.load(scaler_path)
        self.joint_onehot = np.eye(num_joints)

    def compute_torques(self, pos_desired, pos_current, velocity):
        # Build features: [pos_desired, pos_current, velocity, one_hot]
        raw_features = np.stack([pos_desired, pos_current, velocity], axis=1)
        features = np.hstack([raw_features, self.joint_onehot])

        # Scale and predict
        features_scaled = self.scaler.transform(features)
        with torch.no_grad():
            torques = self.model(torch.FloatTensor(features_scaled)).numpy()

        return torques.flatten()
```

### Control Loop

```python
# In MuJoCo simulation loop:
for step in range(max_steps):
    # Get policy action (desired joint positions)
    action = policy(obs)
    target_pos = default_pos + action_scale * action

    # Convert to MuJoCo joint order
    target_pos_mujoco = target_pos[ISAAC_TO_MUJOCO]

    # Compute torques using ActuatorNet
    current_pos = data.qpos[7:19]
    current_vel = data.qvel[6:18]
    tau = actuator_net.compute_torques(target_pos_mujoco, current_pos, current_vel)

    # Apply torques
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)
```

---

## Evaluation

### Evaluation Script

```bash
python scripts/experiments/go2/evaluate_actuatornet_transfer.py \
    --policy logs/rsl_rl/go2/v7/model_200.pt \
    --actuator_net actuator_net/models/go2/actuator_net_jit.pt \
    --scaler actuator_net/models/go2/feature_scaler.pkl \
    --output scripts/experiments/go2/results/actuatornet_comparison
```

### Metrics Compared

| Metric | Description |
|--------|-------------|
| **Survival Rate** | % episodes that don't fall (>80% of max steps) |
| **Episode Length** | Average steps before termination |
| **RMSE vx, vy** | Velocity tracking error |
| **Height Stability** | Mean and variance of base height |
| **Action Smoothness** | L2 norm of action rate |

### Test Commands

```python
command_sets = [
    [0.5, 0.0, 0.0],   # Forward walk
    [1.0, 0.0, 0.0],   # Fast forward
    [0.0, 0.5, 0.0],   # Lateral walk
    [0.0, 0.0, 0.5],   # Turn in place
    [0.5, 0.3, 0.0],   # Diagonal
]
```

---

## Results

### Typical Improvement (Expected)

| Metric | PD Controller | ActuatorNet | Improvement |
|--------|---------------|-------------|-------------|
| Survival Rate | ~20% | ~70% | +50% |
| Episode Length | ~100 steps | ~400 steps | +300 steps |
| RMSE vx | 0.35 m/s | 0.20 m/s | -43% |
| Height Stability | 0.25 m | 0.30 m | +0.05 m |

### Why ActuatorNet Helps

1. **Learned dynamics match MuJoCo**: The network was trained on MuJoCo data
2. **Captures nonlinear effects**: Joint friction, velocity-dependent damping
3. **Compensates for simulator differences**: Contact forces, integration methods
4. **Joint-specific behavior**: Hip vs thigh vs calf joints have different characteristics

---

## File Structure

```
sim2sim_project/
├── actuator_net/
│   ├── __init__.py
│   ├── collect_mujoco_data.py      # Data collection script
│   ├── train_actuator_net.py       # Training script
│   ├── isaac_actuator.py           # Isaac Lab actuator reference
│   ├── data/
│   │   └── go2/
│   │       └── actuator_data.csv   # Training data
│   └── models/
│       └── go2/
│           ├── actuator_net_jit.pt      # Deployed model
│           ├── actuator_net_best.pt     # Best checkpoint
│           └── feature_scaler.pkl       # Input scaler
├── scripts/
│   ├── transfer_go2_mujoco_actuatornet.py    # Transfer script
│   └── experiments/go2/
│       └── evaluate_actuatornet_transfer.py  # Evaluation script
└── docs/
    └── ACTUATOR_NET_GO2.md          # This documentation
```

---

## Usage Guide

### Complete Workflow

#### Step 1: Collect Data from MuJoCo

```bash
# Collect actuator dynamics data
python actuator_net/collect_mujoco_data.py \
    --robot go2 \
    --output actuator_net/data/go2/actuator_data.csv \
    --episodes 100 \
    --sweeps 10
```

#### Step 2: Train ActuatorNet

```bash
# Train the network
python actuator_net/train_actuator_net.py \
    --data actuator_net/data/go2/actuator_data.csv \
    --output actuator_net/models/go2 \
    --epochs 100
```

#### Step 3: Train Policy in Isaac Lab

```bash
# Train with aggressive domain randomization
python scripts/rsl_rl/train.py \
    --task Isaac-Velocity-Go2-Direct-v7 \
    --headless \
    --max_iterations 500
```

#### Step 4: Transfer to MuJoCo with ActuatorNet

```bash
# Test transfer with ActuatorNet
python scripts/transfer_go2_mujoco_actuatornet.py \
    --policy logs/rsl_rl/go2/v7/model_200.pt \
    --actuator_net actuator_net/models/go2/actuator_net_jit.pt \
    --cmd 0.5 0.0 0.0
```

#### Step 5: Evaluate Performance

```bash
# Run full comparison experiment
python scripts/experiments/go2/evaluate_actuatornet_transfer.py \
    --policy logs/rsl_rl/go2/v7/model_200.pt \
    --actuator_net actuator_net/models/go2/actuator_net_jit.pt \
    --output scripts/experiments/go2/results/actuatornet_comparison
```

---

## References

1. **ActuatorNet Project**: https://github.com/sunzhon/actuator_net
2. **Humanoid-Gym**: Isaac Gym to MuJoCo sim-to-sim transfer
3. **Isaac Lab Documentation**: https://isaac-sim.github.io/IsaacLab/
4. **MuJoCo Documentation**: https://mujoco.readthedocs.io/

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Robot still falls | Insufficient training data | Collect more data with wider range |
| High torque oscillations | Scaler mismatch | Re-fit scaler on new data |
| Poor velocity tracking | Policy not trained with DR | Use V7/V8 config with aggressive DR |
| NaN in torques | Input out of training range | Clip inputs to training bounds |

### Tips for Better Results

1. **Use early checkpoints**: Models at 100-200 iterations often transfer better
2. **Aggressive DR during training**: Actuator gains ±40%, friction 0.3-1.5
3. **Match joint ordering**: Isaac Lab and MuJoCo have different conventions
4. **Verify PD gains match**: Kp=20, Kd=0.5 for Go2

---

*Last updated: February 2025*
