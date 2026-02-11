#!/usr/bin/env python3
"""Train RMA Phase 2 Adaptation Module.

This script trains the adaptation module that predicts privileged observations
from observation history. The trained module enables deployment without
access to ground-truth environment parameters.

RMA Architecture:
- Phase 1: Policy trained with privileged encoder (encoder sees ground truth)
- Phase 2: Adaptation module learns to predict encoder output from obs history

Usage:
    python rma/train_adaptation.py \
        --data rma/data/go2_rma_phase2.npz \
        --output rma/models/go2/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


class RMADataset(Dataset):
    """Dataset for RMA adaptation module training."""

    def __init__(self, obs_history: np.ndarray, privileged: np.ndarray):
        """Initialize dataset.

        Args:
            obs_history: Observation history [N, history_length, obs_dim]
            privileged: Privileged observations [N, privileged_dim]
        """
        # Flatten history for MLP input
        self.history_flat = torch.FloatTensor(
            obs_history.reshape(obs_history.shape[0], -1)
        )
        self.privileged = torch.FloatTensor(privileged)

        print(f"Dataset: {len(self)} samples")
        print(f"  History shape: {obs_history.shape} -> {self.history_flat.shape}")
        print(f"  Privileged shape: {self.privileged.shape}")

    def __len__(self):
        return len(self.history_flat)

    def __getitem__(self, idx):
        return self.history_flat[idx], self.privileged[idx]


class AdaptationModule(nn.Module):
    """MLP that predicts privileged observations from observation history.

    Input: Flattened observation history [batch, history_length * obs_dim]
    Output: Predicted privileged observations [batch, privileged_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list = [256, 128],
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        print(f"AdaptationModule: {input_dim} -> {hidden_dims} -> {output_dim}")

    def forward(self, x):
        return self.network(x)


class TCNAdaptationModule(nn.Module):
    """Temporal Convolutional Network for adaptation.

    Processes observation history with 1D convolutions, better capturing
    temporal patterns than MLP.

    Input: Observation history [batch, history_length, obs_dim]
    Output: Predicted privileged observations [batch, privileged_dim]
    """

    def __init__(
        self,
        obs_dim: int,
        history_length: int,
        output_dim: int,
        channels: list = [64, 128, 64],
    ):
        super().__init__()

        # 1D convolutions over time
        conv_layers = []
        prev_channels = obs_dim

        for i, ch in enumerate(channels):
            conv_layers.extend([
                nn.Conv1d(prev_channels, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ELU(),
                nn.MaxPool1d(2) if i < len(channels) - 1 else nn.Identity(),
            ])
            prev_channels = ch

        self.conv = nn.Sequential(*conv_layers)

        # Calculate output size after convolutions
        with torch.no_grad():
            dummy = torch.zeros(1, obs_dim, history_length)
            conv_out = self.conv(dummy)
            conv_flat = conv_out.view(1, -1).shape[1]

        # MLP head
        self.head = nn.Sequential(
            nn.Linear(conv_flat, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

        print(f"TCNAdaptationModule: [{obs_dim} x {history_length}] -> {channels} -> {output_dim}")

    def forward(self, x):
        # x: [batch, history_length, obs_dim]
        # Conv1d expects: [batch, channels, length]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


def train_adaptation_module(
    data_path: str,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    use_tcn: bool = False,
    device: str = "cuda",
):
    """Train the adaptation module.

    Args:
        data_path: Path to collected RMA data (.npz file)
        output_dir: Directory to save trained models
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        use_tcn: Use TCN instead of MLP
        device: Device to train on
    """
    print("=" * 60)
    print("RMA Phase 2: Adaptation Module Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from: {data_path}")
    data = np.load(data_path)
    obs_history = data["obs_history"]  # [N, history_length, obs_dim]
    privileged = data["privileged"]    # [N, privileged_dim]

    history_length = obs_history.shape[1]
    obs_dim = obs_history.shape[2]
    privileged_dim = privileged.shape[1]

    print(f"Observation history: {obs_history.shape}")
    print(f"Privileged observations: {privileged.shape}")

    # Create dataset
    if use_tcn:
        # TCN needs original shape
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(obs_history),
            torch.FloatTensor(privileged),
        )
    else:
        dataset = RMADataset(obs_history, privileged)

    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create model
    if use_tcn:
        model = TCNAdaptationModule(
            obs_dim=obs_dim,
            history_length=history_length,
            output_dim=privileged_dim,
        )
    else:
        input_dim = history_length * obs_dim
        model = AdaptationModule(
            input_dim=input_dim,
            output_dim=privileged_dim,
            hidden_dims=[256, 128],  # matches rma/configs/go2.yaml
        )

    model = model.to(device)
    print(f"\nTraining on: {device}")

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "adaptation_best.pt"))

            # Save as TorchScript for deployment
            model.eval()
            scripted = torch.jit.script(model.cpu())
            scripted.save(os.path.join(output_dir, "adaptation_jit.pt"))
            model = model.to(device)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")

    # Final save
    torch.save(model.state_dict(), os.path.join(output_dir, "adaptation_final.pt"))

    # Save model info
    info = {
        "input_type": "tcn" if use_tcn else "mlp",
        "history_length": history_length,
        "obs_dim": obs_dim,
        "privileged_dim": privileged_dim,
        "best_val_loss": best_val_loss,
    }
    torch.save(info, os.path.join(output_dir, "adaptation_info.pt"))

    print(f"\nModels saved to: {output_dir}/")
    print("  - adaptation_best.pt (best validation)")
    print("  - adaptation_jit.pt (TorchScript for deployment)")
    print("  - adaptation_info.pt (model metadata)")


def main():
    parser = argparse.ArgumentParser(description="Train RMA Adaptation Module")
    parser.add_argument("--data", type=str, default="rma/data/go2_rma_phase2.npz",
                        help="Path to collected RMA data")
    parser.add_argument("--output", type=str, default="rma/models/go2/",
                        help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--tcn", action="store_true",
                        help="Use TCN instead of MLP")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        print("\nFirst collect RMA data with one of:")
        print("  # Isaac Lab:  python rma/collect_rma_data.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-v0 --headless")
        print("  # MuJoCo:     python rma/collect_mujoco_rma_data.py --policy <path_to_policy.pt>")
        return

    train_adaptation_module(
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_tcn=args.tcn,
        device=args.device,
    )


if __name__ == "__main__":
    main()
