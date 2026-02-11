#!/usr/bin/env bash
# Prepare ActuatorNet Phase 2 Improvement
#
# This script sets up everything needed for ActuatorNet v2 training with enhanced features.
# Run this BEFORE starting the actual training to verify prerequisites and create directories.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "ActuatorNet v2 Preparation Script"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# ── Step 1: Verify prerequisites ────────────────────────────────────────────
echo "[1/7] Verifying prerequisites..."

# Check Python packages
python -c "import torch; import mujoco; import numpy; import pandas" 2>/dev/null || {
    echo "❌ ERROR: Missing required packages"
    echo "Install: pip install torch mujoco numpy pandas"
    exit 1
}

# Check moderate DR policy exists
MODERATE_POLICY=$(find logs/go2/rsl_rl/go2_sim2sim_moderate -name "policy.pt" -path "*/exported/*" 2>/dev/null | head -1)
if [ -z "$MODERATE_POLICY" ]; then
    echo "❌ ERROR: Moderate DR policy not found"
    echo "Expected: logs/go2/rsl_rl/go2_sim2sim_moderate/*/exported/policy.pt"
    exit 1
fi

echo "✅ Moderate DR policy found: $MODERATE_POLICY"

# Check MuJoCo XML exists
XML_PATH="unitree_mujoco/unitree_robots/go2/scene_flat.xml"
if [ ! -f "$XML_PATH" ]; then
    echo "❌ ERROR: MuJoCo XML not found at $XML_PATH"
    exit 1
fi

echo "✅ MuJoCo XML found"

# ── Step 2: Create directory structure ─────────────────────────────────────
echo ""
echo "[2/7] Creating directory structure..."

mkdir -p actuator_net/data/go2_v2
mkdir -p actuator_net/models/go2_v2
mkdir -p actuator_net/logs
mkdir -p experiments/go2/results/actuator_net_v2

echo "✅ Directories created"

# ── Step 3: Create enhanced data collection script ─────────────────────────
echo ""
echo "[3/7] Creating enhanced data collection script..."

cat > actuator_net/collect_training_data_v2.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Enhanced ActuatorNet data collection with additional features.

Collects: q_des, q_cur, q_vel, q_acc, error, base_vel, contact_forces, joint_id
"""

import argparse
import mujoco
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add experiments/go2 to path for imports
# Script is in actuator_net/, so we need to go to experiments/go2/
SCRIPT_DIR = Path(__file__).resolve().parent  # .../actuator_net
PROJECT_ROOT = SCRIPT_DIR.parent              # .../
EVAL_DIR = PROJECT_ROOT / "experiments" / "go2"
sys.path.insert(0, str(EVAL_DIR))

# Import from eval_mujoco
import eval_mujoco
quat_rotate_inverse = eval_mujoco.quat_rotate_inverse
MUJOCO_TO_ISAACLAB = eval_mujoco.MUJOCO_TO_ISAACLAB
ISAACLAB_TO_MUJOCO = eval_mujoco.ISAACLAB_TO_MUJOCO

def collect_episode(model, policy, config, max_steps=1000):
    """Collect enhanced training data for one episode."""
    data = mujoco.MjData(model)

    # Reset
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    data.qpos[0:3] = [0, 0, 0.35]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = default_angles
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # Random command
    cmd = np.array([
        np.random.uniform(-0.5, 0.5),  # vx
        np.random.uniform(-0.3, 0.3),  # vy
        np.random.uniform(-0.3, 0.3),  # wz
    ], dtype=np.float32)

    samples = []
    action = np.zeros(12, dtype=np.float32)
    prev_qvel = data.qvel[6:].copy()

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    action_scale = config["action_scale"]
    control_decimation = config["control_decimation"]

    for step in range(max_steps * control_decimation):
        # PD control (ground truth)
        target_dof_pos = action[ISAACLAB_TO_MUJOCO] * action_scale + default_angles
        qpos_cur = data.qpos[7:]
        qvel_cur = data.qvel[6:]
        tau = kps * (target_dof_pos - qpos_cur) - kds * qvel_cur

        # Apply control (with swap)
        data.ctrl[0:3] = tau[3:6]   # FR
        data.ctrl[3:6] = tau[0:3]   # FL
        data.ctrl[6:9] = tau[9:12]  # RR
        data.ctrl[9:12] = tau[6:9]  # RL

        mujoco.mj_step(model, data)

        # Record at control frequency
        if step % control_decimation == 0:
            # Build observation
            quat = data.qpos[3:7]
            base_lin_vel = quat_rotate_inverse(quat, data.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(quat, data.qvel[3:6])

            # Compute acceleration (finite difference)
            q_acc = (data.qvel[6:] - prev_qvel) / (control_decimation * model.opt.timestep)
            prev_qvel = data.qvel[6:].copy()

            # Contact forces (per leg)
            contact_forces = np.zeros(4)
            for i in range(data.ncon):
                contact = data.contact[i]
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                contact_norm = np.linalg.norm(force[:3])
                # Map contact to leg (simplified)
                if contact_norm > 1.0:
                    geom_id = contact.geom1
                    if geom_id < 4:
                        contact_forces[geom_id] = min(contact_norm, 100.0)

            # Record each joint
            for j in range(12):
                sample = {
                    'q_des': target_dof_pos[j],
                    'q_cur': qpos_cur[j],
                    'q_vel': qvel_cur[j],
                    'q_acc': q_acc[j],
                    'error': target_dof_pos[j] - qpos_cur[j],
                    'base_lin_vel_x': base_lin_vel[0],
                    'base_ang_vel_z': base_ang_vel[2],
                    'contact_force': contact_forces[j // 3],  # 3 joints per leg
                    'joint_id': j,
                    'tau': tau[j],
                }
                samples.append(sample)

            # Get next action
            obs_il = data.qpos[7:][MUJOCO_TO_ISAACLAB]
            dqj_il = data.qvel[6:][MUJOCO_TO_ISAACLAB]
            default_angles_il = default_angles[MUJOCO_TO_ISAACLAB]
            gravity = quat_rotate_inverse(quat, np.array([0, 0, -1]))

            obs = np.concatenate([
                base_lin_vel * 2.0,
                base_ang_vel * 0.25,
                gravity,
                cmd * np.array([2.0, 2.0, 0.25]),
                (obs_il - default_angles_il) * 1.0,
                dqj_il * 0.05,
                action,
            ])

            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0).float()).squeeze().numpy()

        # Check if fallen
        if data.qpos[2] < 0.15:
            break

    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True, help='Path to policy.pt')
    parser.add_argument('--xml', required=True, help='Path to MuJoCo XML')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--episode_length', type=int, default=1000)
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load
    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = config["simulation_dt"]

    policy = torch.jit.load(args.policy)
    policy.eval()

    print(f"Collecting {args.num_episodes} episodes...")
    all_samples = []

    for ep in range(args.num_episodes):
        samples = collect_episode(model, policy, config, max_steps=args.episode_length)
        all_samples.extend(samples)

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}/{args.num_episodes}: {len(samples)} samples")

    # Save
    df = pd.DataFrame(all_samples)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Saved {len(df)} samples to {args.output}")
    print(f"   Size: {Path(args.output).stat().st_size / 1e6:.1f} MB")

if __name__ == '__main__':
    main()
PYTHON_EOF

chmod +x actuator_net/collect_training_data_v2.py
echo "✅ Created collect_training_data_v2.py"

# ── Step 4: Create training script v2 ──────────────────────────────────────
echo ""
echo "[4/7] Creating ActuatorNet v2 training script..."

cat > actuator_net/train_actuator_net_v2.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Train ActuatorNet v2 with enhanced features and architecture."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

class ActuatorNetV2(nn.Module):
    """Enhanced ActuatorNet with residual connections."""

    def __init__(self, input_dim=23):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        # Encoder
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)

        # Residual block 1
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 256)
        self.ln3 = nn.LayerNorm(256)

        # Residual block 2
        self.fc4 = nn.Linear(256, 128)
        self.ln4 = nn.LayerNorm(128)

        # Output
        self.fc5 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.SiLU()

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
        x = self.activation(x + identity)

        # Residual block 2
        x = self.activation(self.ln4(self.fc4(x)))
        x = self.dropout(x)

        # Output
        x = self.activation(self.fc5(x))
        return self.fc_out(x)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  Total samples: {len(df):,}")

    # Features
    feature_cols = ['q_des', 'q_cur', 'q_vel', 'q_acc', 'error',
                   'base_lin_vel_x', 'base_ang_vel_z', 'contact_force']
    X_numeric = df[feature_cols].values

    # One-hot joint ID
    joint_ids = df['joint_id'].values
    joint_onehot = np.eye(12)[joint_ids]

    X = np.concatenate([X_numeric, joint_onehot], axis=1)  # (N, 23)
    y = df['tau'].values.reshape(-1, 1)

    print(f"  Features shape: {X.shape}")
    print(f"  Targets shape: {y.shape}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Save scaler
    with open(output_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Saved scaler to {output_dir / 'scaler.pkl'}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ActuatorNetV2(input_dim=23).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    print(f"\nTraining on {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1:3d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'actuator_net_v2.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Export TorchScript
    model.load_state_dict(torch.load(output_dir / 'actuator_net_v2.pt'))
    model.eval()
    example_input = torch.randn(1, 23).to(device)
    traced = torch.jit.trace(model, example_input)
    torch.jit.save(traced, output_dir / 'actuator_net_v2_jit.pt')

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Saved to: {output_dir}")

if __name__ == '__main__':
    train()
PYTHON_EOF

chmod +x actuator_net/train_actuator_net_v2.py
echo "✅ Created train_actuator_net_v2.py"

# ── Step 5: Create README ───────────────────────────────────────────────────
echo ""
echo "[5/7] Creating ActuatorNet v2 README..."

cat > actuator_net/README_V2.md << 'README_EOF'
# ActuatorNet v2 — Enhanced Training Guide

## Quick Start

### 1. Collect Training Data (2 hours)
```bash
python actuator_net/collect_training_data_v2.py \
    --policy logs/go2/rsl_rl/go2_sim2sim_moderate/*/exported/policy.pt \
    --xml unitree_mujoco/unitree_robots/go2/scene_flat.xml \
    --config scripts/go2/go2_isaaclab.yaml \
    --num_episodes 1000 \
    --episode_length 1000 \
    --output actuator_net/data/go2_v2/training_data.csv \
    --seed 42
```

### 2. Train ActuatorNet v2 (1 hour)
```bash
python actuator_net/train_actuator_net_v2.py \
    --data actuator_net/data/go2_v2/training_data.csv \
    --output_dir actuator_net/models/go2_v2 \
    --epochs 150 \
    --batch_size 512 \
    --lr 5e-4 \
    --patience 20
```

### 3. Evaluate
```bash
python experiments/go2/eval_mujoco.py \
    --policy logs/go2/rsl_rl/go2_sim2sim_moderate/*/exported/policy.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --experiment directions \
    --episodes 5 \
    --max_steps 500 \
    --output_dir experiments/go2/results/actuator_net_v2 \
    --label "ActuatorNet-v2" \
    --actuator_net \
    --actuator_net_path actuator_net/models/go2_v2/actuator_net_v2_jit.pt
```

## What's New in v2?

### Enhanced Features (8 → 23 dimensions)
- **v1**: q_des, q_cur, q_vel, joint_id
- **v2**: + q_acc, error, base_vel, contact_forces

### Improved Architecture
- Residual connections for better gradient flow
- SiLU activation (smoother than ELU)
- Larger capacity: [256, 256, 128, 64]

### Better Training
- Cosine annealing scheduler
- Gradient clipping
- Stronger regularization

## Expected Results
- Forward RMSE: 0.156 → 0.12 m/s (target)
- Success rate: 100% maintained
- Oscillation: Low (maintained from v1)
README_EOF

echo "✅ Created ActuatorNet v2 README"

# ── Step 6: Print status summary ───────────────────────────────────────────
echo ""
echo "[6/7] Checking current status..."

if [ -f "actuator_net/models/go2/actuator_net_jit.pt" ]; then
    echo "✅ ActuatorNet v1 exists: $(ls -lh actuator_net/models/go2/actuator_net_jit.pt | awk '{print $5}')"
else
    echo "⚠️  ActuatorNet v1 not found (you'll train v2 from scratch)"
fi

if [ -d "experiments/go2/results/actuator_net" ]; then
    echo "✅ ActuatorNet v1 evaluation exists"
else
    echo "⚠️  ActuatorNet v1 evaluation not found (needed for comparison)"
fi

# ── Step 7: Print next steps ───────────────────────────────────────────────
echo ""
echo "[7/7] Preparation complete! ✅"
echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
echo ""
echo "1. Collect enhanced training data (~2 hours):"
echo "   python actuator_net/collect_training_data_v2.py \\"
echo "       --policy $MODERATE_POLICY \\"
echo "       --xml $XML_PATH \\"
echo "       --config scripts/go2/go2_isaaclab.yaml \\"
echo "       --num_episodes 1000 \\"
echo "       --episode_length 1000 \\"
echo "       --output actuator_net/data/go2_v2/training_data.csv \\"
echo "       --seed 42"
echo ""
echo "2. Train ActuatorNet v2 (~1 hour):"
echo "   python actuator_net/train_actuator_net_v2.py \\"
echo "       --data actuator_net/data/go2_v2/training_data.csv \\"
echo "       --output_dir actuator_net/models/go2_v2 \\"
echo "       --epochs 150 \\"
echo "       --batch_size 512 \\"
echo "       --lr 5e-4 \\"
echo "       --patience 20"
echo ""
echo "3. Evaluate and compare (~30 min):"
echo "   See actuator_net/README_V2.md for evaluation commands"
echo ""
echo "For full guide, see: experiments/go2/IMPROVEMENT_GUIDE.md"
echo "============================================================"
