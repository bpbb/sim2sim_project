#!/usr/bin/env bash
# Prepare RMA Phase 2 — Adaptation Module Training
#
# This script sets up everything needed for RMA Phase 2 adaptation module training.
# Run this BEFORE starting the actual training to verify prerequisites and create directories.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "RMA Phase 2 Preparation Script"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo ""

# ── Step 1: Verify prerequisites ────────────────────────────────────────────
echo "[1/6] Verifying prerequisites..."

# Check Python packages
python -c "import torch; import mujoco; import numpy" 2>/dev/null || {
    echo "❌ ERROR: Missing required packages"
    echo "Install: pip install torch mujoco numpy"
    exit 1
}

# Check RMA Phase 1 policy exists
RMA_POLICY=$(find logs/go2/rsl_rl/go2_rma -name "policy.pt" -path "*/exported/*" 2>/dev/null | head -1)
if [ -z "$RMA_POLICY" ]; then
    echo "❌ ERROR: RMA Phase 1 policy not found"
    echo "Expected: logs/go2/rsl_rl/go2_rma/*/exported/policy.pt"
    exit 1
fi

echo "✅ RMA Phase 1 policy found: $RMA_POLICY"

# Check Phase 1 checkpoint (for data collection in Isaac Lab)
RMA_CHECKPOINT=$(find logs/go2/rsl_rl/go2_rma -name "model_4999.pt" -o -name "model_*.pt" 2>/dev/null | tail -1)
if [ -z "$RMA_CHECKPOINT" ]; then
    echo "⚠️  Warning: RMA checkpoint not found (needed for Isaac Lab collection)"
else
    echo "✅ RMA checkpoint found: $RMA_CHECKPOINT"
fi

# ── Step 2: Create directory structure ─────────────────────────────────────
echo ""
echo "[2/6] Creating directory structure..."

mkdir -p experiments/go2/rma/data
mkdir -p experiments/go2/rma/models
mkdir -p experiments/go2/rma/logs
mkdir -p experiments/go2/results/rma_phase2

echo "✅ Directories created"

# ── Step 3: Create MuJoCo data collection script ───────────────────────────
echo ""
echo "[3/6] Creating MuJoCo adaptation data collection script..."

cat > experiments/go2/rma/collect_adaptation_data_mujoco.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Collect RMA adaptation training data from MuJoCo with randomized physics.

Collects (observation_history, privileged_info) pairs for adaptation module training.
"""

import argparse
import collections
import mujoco
import numpy as np
import torch
import yaml
from pathlib import Path
import sys
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "go2"))

from eval_mujoco import (
    quat_rotate_inverse,
    get_gravity_orientation,
    MUJOCO_TO_ISAACLAB,
    ISAACLAB_TO_MUJOCO,
)

def collect_episode(model, data, policy, config, privileged_params, history_length=50):
    """Collect one episode with fixed privileged parameters."""
    # Reset
    default_angles = np.array(config["default_angles"], dtype=np.float32)
    default_angles_il = default_angles[MUJOCO_TO_ISAACLAB]

    data.qpos[0:3] = [0, 0, 0.35]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = default_angles
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # Random command
    cmd = np.array([
        np.random.uniform(-0.5, 0.5),
        np.random.uniform(-0.3, 0.3),
        np.random.uniform(-0.3, 0.3),
    ], dtype=np.float32)

    # Apply randomized physics (only for ground friction and mass)
    # Note: MuJoCo doesn't support per-episode PD gain changes easily
    friction_scale, mass_offset, kp_scale, kd_scale = privileged_params

    # Update ground friction
    model.geom_friction[0, 0] = friction_scale * 1.0  # Nominal = 1.0

    # Update base mass
    nominal_mass = 15.0  # Go2 base mass (approximate)
    model.body_mass[1] = nominal_mass + mass_offset

    # PD gains (scaled from nominal)
    kps = np.array(config["kps"]) * kp_scale
    kds = np.array(config["kds"]) * kd_scale

    # History buffer
    obs_history = collections.deque(maxlen=history_length)
    samples = []

    action = np.zeros(12, dtype=np.float32)
    action_scale = config["action_scale"]
    control_decimation = config["control_decimation"]
    max_steps = 500  # 10 seconds

    for step in range(max_steps * control_decimation):
        # PD control
        target_dof_pos = action[ISAACLAB_TO_MUJOCO] * action_scale + default_angles
        qpos_cur = data.qpos[7:]
        qvel_cur = data.qvel[6:]
        tau = kps * (target_dof_pos - qpos_cur) - kds * qvel_cur

        # Apply control (with swap)
        data.ctrl[0:3] = tau[3:6]
        data.ctrl[3:6] = tau[0:3]
        data.ctrl[6:9] = tau[9:12]
        data.ctrl[9:12] = tau[6:9]

        mujoco.mj_step(model, data)

        # Record at control frequency
        if step % control_decimation == 0:
            # Build observation (48-dim)
            quat = data.qpos[3:7]
            base_lin_vel = quat_rotate_inverse(quat, data.qvel[0:3])
            base_ang_vel = quat_rotate_inverse(quat, data.qvel[3:6])
            gravity = get_gravity_orientation(quat)

            obs_il = data.qpos[7:][MUJOCO_TO_ISAACLAB]
            dqj_il = data.qvel[6:][MUJOCO_TO_ISAACLAB]

            obs = np.concatenate([
                base_lin_vel * 2.0,
                base_ang_vel * 0.25,
                gravity,
                cmd * np.array([2.0, 2.0, 0.25]),
                (obs_il - default_angles_il) * 1.0,
                dqj_il * 0.05,
                action,
            ])

            obs_history.append(obs)

            # Record sample if history is full
            if len(obs_history) == history_length:
                sample = {
                    'obs_history': np.array(obs_history),  # (50, 48)
                    'privileged': privileged_params,       # (4,)
                    'command': cmd,
                }
                samples.append(sample)

            # Get next action (with dummy privileged obs for consistency)
            obs_with_priv = np.concatenate([obs, [1.0, 0.0, 1.0, 1.0]])
            with torch.no_grad():
                action = policy(torch.from_numpy(obs_with_priv).unsqueeze(0).float()).squeeze().numpy()

        # Check if fallen
        if data.qpos[2] < 0.15:
            break

    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True, help='RMA Phase 1 policy.pt')
    parser.add_argument('--config', required=True, help='Config YAML')
    parser.add_argument('--num_samples', type=int, default=50000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output', required=True, help='Output .npz path')
    parser.add_argument('--randomize_physics', action='store_true')
    parser.add_argument('--friction_range', nargs=2, type=float, default=[0.5, 1.5])
    parser.add_argument('--mass_range', nargs=2, type=float, default=[-0.5, 1.0])
    parser.add_argument('--kp_range', nargs=2, type=float, default=[0.8, 1.2])
    parser.add_argument('--kd_range', nargs=2, type=float, default=[0.8, 1.2])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load
    with open(args.config) as f:
        config = yaml.safe_load(f)

    xml_path = Path(config["xml_path"])
    if not xml_path.is_absolute():
        xml_path = PROJECT_ROOT / xml_path

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = config["simulation_dt"]

    policy = torch.jit.load(args.policy)
    policy.eval()

    print(f"Collecting {args.num_samples} samples with randomized physics...")
    print(f"  Friction range: {args.friction_range}")
    print(f"  Mass range: {args.mass_range}")
    print(f"  Kp range: {args.kp_range}")
    print(f"  Kd range: {args.kd_range}")

    all_samples = []
    episodes_needed = args.num_samples // 200  # ~200 samples per episode

    for ep in tqdm(range(episodes_needed)):
        # Sample privileged parameters
        privileged = np.array([
            np.random.uniform(*args.friction_range),
            np.random.uniform(*args.mass_range),
            np.random.uniform(*args.kp_range),
            np.random.uniform(*args.kd_range),
        ], dtype=np.float32)

        data = mujoco.MjData(model)
        samples = collect_episode(model, data, policy, config, privileged)
        all_samples.extend(samples)

        if len(all_samples) >= args.num_samples:
            break

    # Convert to arrays
    all_samples = all_samples[:args.num_samples]
    obs_histories = np.array([s['obs_history'] for s in all_samples])  # (N, 50, 48)
    privileged = np.array([s['privileged'] for s in all_samples])      # (N, 4)
    commands = np.array([s['command'] for s in all_samples])           # (N, 3)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        obs_history=obs_histories,
        privileged=privileged,
        commands=commands,
    )

    print(f"\n✅ Saved {len(all_samples)} samples to {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"   Shape: obs_history={obs_histories.shape}, privileged={privileged.shape}")

if __name__ == '__main__':
    main()
PYTHON_EOF

chmod +x experiments/go2/rma/collect_adaptation_data_mujoco.py
echo "✅ Created collect_adaptation_data_mujoco.py"

# ── Step 4: Create adaptation module training script ───────────────────────
echo ""
echo "[4/6] Creating adaptation module training script..."

cat > experiments/go2/rma/train_adaptation_module.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Train RMA Phase 2 adaptation module."""

import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

class AdaptationModuleTransformer(nn.Module):
    """Transformer-based adaptation module."""

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
        x = x.mean(dim=1)                 # Global average pooling
        x = self.decoder(x)               # (B, latent_dim)
        return x

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to .npz data file')
    parser.add_argument('--architecture', default='transformer', choices=['transformer', 'cnn'])
    parser.add_argument('--history_length', type=int, default=50)
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.data}...")
    data = np.load(args.data)
    obs_history = data['obs_history']  # (N, 50, 48)
    privileged = data['privileged']    # (N, 4)

    print(f"  obs_history shape: {obs_history.shape}")
    print(f"  privileged shape: {privileged.shape}")

    # Train/val split
    n_train = int(0.8 * len(obs_history))
    indices = np.random.permutation(len(obs_history))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    X_train = torch.FloatTensor(obs_history[train_idx])
    y_train = torch.FloatTensor(privileged[train_idx])
    X_val = torch.FloatTensor(obs_history[val_idx])
    y_val = torch.FloatTensor(privileged[val_idx])

    # Normalize privileged info to [-1, 1]
    priv_min = y_train.min(dim=0)[0]
    priv_max = y_train.max(dim=0)[0]
    y_train = 2 * (y_train - priv_min) / (priv_max - priv_min + 1e-8) - 1
    y_val = 2 * (y_val - priv_min) / (priv_max - priv_min + 1e-8) - 1

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaptationModuleTransformer(
        obs_dim=48,
        history_len=args.history_length,
        latent_dim=args.latent_dim
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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

        print(f"Epoch {epoch+1:3d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'adaptation_module.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Export TorchScript
    model.load_state_dict(torch.load(output_dir / 'adaptation_module.pt'))
    model.eval()
    example_input = torch.randn(1, args.history_length, 48).to(device)
    traced = torch.jit.trace(model, example_input)
    torch.jit.save(traced, output_dir / 'adaptation_jit.pt')

    # Save normalization parameters
    torch.save({
        'priv_min': priv_min,
        'priv_max': priv_max
    }, output_dir / 'normalization.pt')

    print(f"\n✅ Training complete!")
    print(f"   Best val loss: {best_val_loss:.6f}")
    print(f"   Saved to: {output_dir}")

if __name__ == '__main__':
    train()
PYTHON_EOF

chmod +x experiments/go2/rma/train_adaptation_module.py
echo "✅ Created train_adaptation_module.py"

# ── Step 5: Create README ───────────────────────────────────────────────────
echo ""
echo "[5/6] Creating RMA Phase 2 README..."

cat > experiments/go2/rma/README_PHASE2.md << 'README_EOF'
# RMA Phase 2 — Adaptation Module Training Guide

## Quick Start

### 1. Collect Adaptation Data from MuJoCo (2 hours)
```bash
python experiments/go2/rma/collect_adaptation_data_mujoco.py \
    --policy logs/go2/rsl_rl/go2_rma/*/exported/policy.pt \
    --config scripts/go2/go2_isaaclab.yaml \
    --num_samples 50000 \
    --num_workers 8 \
    --output experiments/go2/rma/adaptation_data.npz \
    --randomize_physics \
    --friction_range 0.5 1.5 \
    --mass_range -0.5 1.0 \
    --kp_range 0.8 1.2 \
    --kd_range 0.8 1.2 \
    --seed 42
```

### 2. Train Adaptation Module (2 hours)
```bash
python experiments/go2/rma/train_adaptation_module.py \
    --data experiments/go2/rma/adaptation_data.npz \
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

### 3. Evaluate RMA Phase 2
```bash
# Note: Requires deployment script with adaptation module integration
# See experiments/go2/IMPROVEMENT_GUIDE.md for deployment details
```

## Architecture Details

### Transformer (Recommended)
- Input: (batch, 50, 48) observation history
- Transformer: 2 layers, 4 heads, 128-dim embedding
- Output: (batch, 4) predicted latent z

### Training Configuration
- Loss: MSE between predicted and ground-truth latent
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine annealing
- Early stopping: patience=15

## Expected Results
- Validation MSE: < 0.01 per dimension
- Success rate: ≥ 95% (vs 62.5% Phase 1)
- Forward RMSE: ≤ 0.15 m/s
- Adaptation convergence: < 0.5s

## Debugging

### If validation loss is high (> 0.05):
1. Check data quality: visualize obs_history samples
2. Increase model capacity: layers 2 → 4
3. Reduce learning rate: 1e-3 → 5e-4

### If deployed policy doesn't work:
1. Verify z_hat is non-zero during deployment
2. Check normalization is applied correctly
3. Ensure Phase 1 policy properly uses privileged info
README_EOF

echo "✅ Created RMA Phase 2 README"

# ── Step 6: Print next steps ───────────────────────────────────────────────
echo ""
echo "[6/6] Preparation complete! ✅"
echo ""
echo "============================================================"
echo "NEXT STEPS"
echo "============================================================"
echo ""
echo "1. Collect adaptation training data (~2 hours):"
echo "   python experiments/go2/rma/collect_adaptation_data_mujoco.py \\"
echo "       --policy $RMA_POLICY \\"
echo "       --config scripts/go2/go2_isaaclab.yaml \\"
echo "       --num_samples 50000 \\"
echo "       --num_workers 8 \\"
echo "       --output experiments/go2/rma/adaptation_data.npz \\"
echo "       --randomize_physics \\"
echo "       --friction_range 0.5 1.5 \\"
echo "       --mass_range -0.5 1.0 \\"
echo "       --kp_range 0.8 1.2 \\"
echo "       --kd_range 0.8 1.2"
echo ""
echo "2. Train adaptation module (~2 hours):"
echo "   python experiments/go2/rma/train_adaptation_module.py \\"
echo "       --data experiments/go2/rma/adaptation_data.npz \\"
echo "       --architecture transformer \\"
echo "       --output_dir experiments/go2/rma/adaptation_module"
echo ""
echo "3. Deploy and evaluate (~1 hour):"
echo "   See experiments/go2/rma/README_PHASE2.md for deployment integration"
echo ""
echo "For full guide, see: experiments/go2/IMPROVEMENT_GUIDE.md"
echo "============================================================"
