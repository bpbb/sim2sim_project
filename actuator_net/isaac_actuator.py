#!/usr/bin/env python3
"""Isaac Lab actuator model using trained Actuator Net.

This module provides an actuator model that uses a trained neural network
to predict joint torques, replacing the analytical PD controller with
learned dynamics from MuJoCo.

Usage in Isaac Lab env config:
    from actuator_net.isaac_actuator import ActuatorNetCfg

    robot = UNITREE_GO2_CFG.replace(
        actuators={
            "base_legs": ActuatorNetCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                model_path="actuator_net/models/go2/actuator_net_jit.pt",
                scaler_path="actuator_net/models/go2/feature_scaler.pkl",
                num_model_joints=12,
            ),
        },
    )
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import numpy as np

from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

# Try to import joblib, fallback to pickle if not available
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    import pickle
    HAS_JOBLIB = False


# ═══════════════════════════════════════════════════════════════════════════════
# ACTUATOR MODEL CLASS (defined first so Config can reference it)
# ═══════════════════════════════════════════════════════════════════════════════

class ActuatorNetModel(ActuatorBase):
    """Neural network actuator that predicts torques using trained MLP.

    This replaces the analytical PD controller with a learned model that
    better captures the actuator dynamics of the target simulator (MuJoCo).

    The model was trained on data collected from MuJoCo:
    (pos_desired, pos_current, velocity) -> torque
    """

    cfg: "ActuatorNetCfg"
    """The configuration for the actuator model."""

    def __init__(self, cfg: "ActuatorNetCfg", *args, **kwargs):
        """Initialize the ActuatorNet model.

        Args:
            cfg: The configuration for the actuator model.
            *args: Additional positional arguments for ActuatorBase.
            **kwargs: Additional keyword arguments for ActuatorBase.
        """
        # Call parent constructor first
        super().__init__(cfg, *args, **kwargs)

        # Store configuration
        self._num_model_joints = cfg.num_model_joints
        self._fallback_kp = cfg.fallback_stiffness
        self._fallback_kd = cfg.fallback_damping

        # Load trained model
        self._load_model(cfg.model_path)

        # Load feature scaler
        self._load_scaler(cfg.scaler_path)

        # Create buffer for one-hot encoding (reused each step)
        self._joint_onehot_buffer = None

    def _load_model(self, model_path: str):
        """Load the trained TorchScript model."""
        if os.path.exists(model_path):
            try:
                self._model = torch.jit.load(model_path, map_location=self._device)
                self._model.eval()
                print(f"[ActuatorNet] Loaded model from: {model_path}")
            except Exception as e:
                print(f"[ActuatorNet] ERROR loading model: {e}")
                print("[ActuatorNet] Using fallback PD controller")
                self._model = None
        else:
            print(f"[ActuatorNet] WARNING: Model not found at {model_path}")
            print("[ActuatorNet] Using fallback PD controller")
            self._model = None

    def _load_scaler(self, scaler_path: str):
        """Load the feature scaler for normalization."""
        if os.path.exists(scaler_path):
            try:
                if HAS_JOBLIB:
                    self._scaler = joblib.load(scaler_path)
                else:
                    with open(scaler_path, 'rb') as f:
                        self._scaler = pickle.load(f)
                print(f"[ActuatorNet] Loaded scaler from: {scaler_path}")
            except Exception as e:
                print(f"[ActuatorNet] ERROR loading scaler: {e}")
                self._scaler = None
        else:
            print(f"[ActuatorNet] WARNING: Scaler not found at {scaler_path}")
            self._scaler = None

    def reset(self, env_ids: Sequence[int]):
        """Reset the actuator state for specified environments.

        Args:
            env_ids: List of environment IDs to reset.
        """
        # No internal state to reset for this actuator
        pass

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute actuator torques using neural network.

        Args:
            control_action: The joint action instance with desired positions/velocities/efforts.
            joint_pos: Current joint positions [num_envs, num_joints].
            joint_vel: Current joint velocities [num_envs, num_joints].

        Returns:
            The computed articulation actions with joint efforts set.
        """
        # Get desired positions from control action
        pos_desired = control_action.joint_positions

        # If no desired positions provided, use current positions (zero error)
        if pos_desired is None:
            pos_desired = joint_pos.clone()

        batch_size = pos_desired.shape[0]
        num_joints_actual = pos_desired.shape[1]

        if self._model is None:
            # Fallback to analytical PD controller
            torques = self._compute_pd_fallback(pos_desired, joint_pos, joint_vel)
        else:
            # Use neural network to predict torques
            torques = self._compute_neural_net(pos_desired, joint_pos, joint_vel, batch_size, num_joints_actual)

        # Add feed-forward effort if provided
        if control_action.joint_efforts is not None:
            torques = torques + control_action.joint_efforts

        # Store computed effort and clip to limits
        self.computed_effort = torques
        self.applied_effort = self._clip_effort(torques)

        # Return action with efforts only (explicit actuator model)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action

    def _compute_pd_fallback(
        self,
        pos_desired: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Compute torques using fallback PD controller.

        Args:
            pos_desired: Desired joint positions [num_envs, num_joints].
            joint_pos: Current joint positions [num_envs, num_joints].
            joint_vel: Current joint velocities [num_envs, num_joints].

        Returns:
            Computed torques [num_envs, num_joints].
        """
        error_pos = pos_desired - joint_pos
        torques = self._fallback_kp * error_pos - self._fallback_kd * joint_vel
        return torques

    def _compute_neural_net(
        self,
        pos_desired: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        batch_size: int,
        num_joints: int,
    ) -> torch.Tensor:
        """Compute torques using trained neural network.

        Args:
            pos_desired: Desired joint positions [num_envs, num_joints].
            joint_pos: Current joint positions [num_envs, num_joints].
            joint_vel: Current joint velocities [num_envs, num_joints].
            batch_size: Number of environments.
            num_joints: Number of joints.

        Returns:
            Computed torques [num_envs, num_joints].
        """
        # Flatten inputs: [batch * num_joints, 1]
        pos_des_flat = pos_desired.reshape(-1, 1)
        pos_cur_flat = joint_pos.reshape(-1, 1)
        vel_flat = joint_vel.reshape(-1, 1)

        # Create joint one-hot encoding [batch * num_joints, num_model_joints]
        # Lazily create buffer on first use or if batch size changed
        if self._joint_onehot_buffer is None or self._joint_onehot_buffer.shape[0] != batch_size * num_joints:
            joint_indices = torch.arange(num_joints, device=self._device)
            joint_indices = joint_indices.repeat(batch_size)
            # Use min to handle case where actual joints < model joints
            joint_indices_clamped = torch.clamp(joint_indices, max=self._num_model_joints - 1)
            self._joint_onehot_buffer = torch.nn.functional.one_hot(
                joint_indices_clamped, num_classes=self._num_model_joints
            ).float()

        # Concatenate features: [batch * num_joints, 3 + num_model_joints]
        raw_features = torch.cat([pos_des_flat, pos_cur_flat, vel_flat], dim=1)
        features = torch.cat([raw_features, self._joint_onehot_buffer], dim=1)

        # Apply feature scaling if available
        if self._scaler is not None:
            # Move to CPU for sklearn scaler
            features_np = features.cpu().numpy()
            features_scaled = self._scaler.transform(features_np)
            features = torch.FloatTensor(features_scaled).to(self._device)

        # Predict torques with neural network
        with torch.no_grad():
            torques_flat = self._model(features)

        # Reshape back to [batch, num_joints]
        torques = torques_flat.reshape(batch_size, num_joints)

        return torques


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLASS (references ActuatorNetModel directly)
# ═══════════════════════════════════════════════════════════════════════════════

@configclass
class ActuatorNetCfg(ActuatorBaseCfg):
    """Configuration for neural network actuator model.

    This actuator uses a trained MLP to predict torques from:
    - Desired joint position
    - Current joint position
    - Current joint velocity
    - Joint index (one-hot encoded)

    The model is trained on data collected from MuJoCo simulation.
    """

    # Reference to the actuator class - set directly to avoid None issues
    class_type: type = ActuatorNetModel

    # Path to trained TorchScript model
    model_path: str = "actuator_net/models/go2/actuator_net_jit.pt"

    # Path to feature scaler (sklearn StandardScaler)
    scaler_path: str = "actuator_net/models/go2/feature_scaler.pkl"

    # Number of joints the model was trained on
    num_model_joints: int = 12

    # Fallback PD gains (used if model file not found)
    fallback_stiffness: float = 20.0
    fallback_damping: float = 0.5

    # Override stiffness/damping from base class
    # Set to 0 since neural net doesn't use PD gains directly
    stiffness: float = 0.0
    damping: float = 0.0
