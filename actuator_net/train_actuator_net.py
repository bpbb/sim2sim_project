#!/usr/bin/env python3
"""Train Actuator Net MLP model from collected data.

This trains an MLP to predict torque from (pos_desired, pos_current, velocity).
Based on the actuator_net project: https://github.com/sunzhon/actuator_net

Usage:
    python actuator_net/train_actuator_net.py --data actuator_net/data/actuator_data.csv
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib


class ActuatorDataset(Dataset):
    """Dataset for actuator training."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class ActuatorNet(nn.Module):
    """MLP to model actuator dynamics.
    
    Input: [pos_desired, pos_current, velocity, joint_idx_onehot]
    Output: torque
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Output: single torque value
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze(-1)


def load_data(csv_path: str, num_joints: int = 12) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Load and preprocess actuator data."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Create features: [pos_desired, pos_current, velocity, joint_idx_onehot]
    # One-hot encode joint index for joint-specific learning
    joint_onehot = np.eye(num_joints)[df['joint_idx'].values.astype(int) % num_joints]
    
    raw_features = df[['pos_desired', 'pos_current', 'velocity']].values
    features = np.hstack([raw_features, joint_onehot])
    
    targets = df['torque'].values
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"Feature shape: {features_scaled.shape}")
    print(f"Target shape: {targets.shape}")
    
    return features_scaled, targets, scaler


def train_model(features: np.ndarray, targets: np.ndarray, 
                output_dir: str, epochs: int = 100, batch_size: int = 256):
    """Train the actuator net model."""
    
    # Create dataset
    dataset = ActuatorDataset(features, targets)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    input_dim = features.shape[1]
    model = ActuatorNet(input_dim=input_dim, hidden_dims=[128, 64, 32])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on: {device}")
    print(f"Model: {model}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_features)
                val_loss += criterion(predictions, batch_targets).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "actuator_net_best.pt"))
            
            # Also save as JIT for deployment
            model.eval()
            scripted = torch.jit.script(model.cpu())
            scripted.save(os.path.join(output_dir, "actuator_net_jit.pt"))
            model = model.to(device)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
    
    # Final save
    torch.save(model.state_dict(), os.path.join(output_dir, "actuator_net_final.pt"))
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Actuator Net")
    parser.add_argument("--data", type=str, default="actuator_net/data/actuator_data.csv",
                        help="Path to training data CSV")
    parser.add_argument("--output", type=str, default="actuator_net/models",
                        help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_joints", type=int, default=12, help="Number of joints")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    features, targets, scaler = load_data(args.data, args.num_joints)
    
    # Save scaler for inference
    joblib.dump(scaler, os.path.join(args.output, "feature_scaler.pkl"))
    print(f"Saved feature scaler to: {args.output}/feature_scaler.pkl")
    
    # Train model
    model = train_model(features, targets, args.output, args.epochs, args.batch_size)
    
    print(f"\nModels saved to: {args.output}/")
    print("  - actuator_net_best.pt (PyTorch state dict)")
    print("  - actuator_net_jit.pt (TorchScript for deployment)")
    print("  - feature_scaler.pkl (Scikit-learn scaler)")


if __name__ == "__main__":
    main()
