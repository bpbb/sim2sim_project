"""Custom observation terms for Go2 RMA training.

These functions expose privileged physics parameters as observations for
the RMA critic during training. They read the ACTUAL randomized values
from the simulator (not nominal values).

Privileged info vector (4 dims):
  [0] ground_friction  — mean static friction coefficient
  [1] base_mass_offset — mass added to base body (kg)
  [2] kp_scale         — actuator stiffness / nominal (20.0)
  [3] kd_scale         — actuator damping / nominal (0.5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def ground_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Mean static friction coefficient across robot body shapes.

    Reads the current (randomized) material properties from PhysX.

    Returns:
        [num_envs, 1] mean static friction coefficient.
    """
    asset = env.scene[asset_cfg.name]
    # get_material_properties() → [num_envs, num_shapes, 3]
    # where 3 = (static_friction, dynamic_friction, restitution)
    materials = asset.root_physx_view.get_material_properties()
    # Average static friction across all shapes
    mean_friction = materials[:, :, 0].mean(dim=-1, keepdim=True)
    return mean_friction.to(env.device)


def body_mass_offset(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nominal_mass: float = 6.921,
) -> torch.Tensor:
    """Base body mass offset from nominal value.

    Reads the current (randomized) mass from PhysX and subtracts nominal.

    Args:
        nominal_mass: Go2 base body nominal mass in kg.

    Returns:
        [num_envs, 1] mass offset in kg.
    """
    asset = env.scene[asset_cfg.name]
    # get_masses() → [num_envs, num_bodies]
    masses = asset.root_physx_view.get_masses()
    base_mass = masses[:, 0:1]  # base body is index 0
    return (base_mass - nominal_mass).to(env.device)


def actuator_stiffness_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nominal_kp: float = 20.0,
) -> torch.Tensor:
    """Mean actuator stiffness as scale factor of nominal.

    Reads the current (randomized) stiffness from the actuator model.

    Args:
        nominal_kp: Nominal PD stiffness gain.

    Returns:
        [num_envs, 1] stiffness scale factor (1.0 = nominal).
    """
    asset = env.scene[asset_cfg.name]
    # actuator stiffness: [num_envs, num_joints]
    kp = asset.actuators["base_legs"].stiffness
    mean_kp = kp.mean(dim=-1, keepdim=True)
    return (mean_kp / nominal_kp).to(env.device)


def actuator_damping_scale(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    nominal_kd: float = 0.5,
) -> torch.Tensor:
    """Mean actuator damping as scale factor of nominal.

    Reads the current (randomized) damping from the actuator model.

    Args:
        nominal_kd: Nominal PD damping gain.

    Returns:
        [num_envs, 1] damping scale factor (1.0 = nominal).
    """
    asset = env.scene[asset_cfg.name]
    # actuator damping: [num_envs, num_joints]
    kd = asset.actuators["base_legs"].damping
    mean_kd = kd.mean(dim=-1, keepdim=True)
    return (mean_kd / nominal_kd).to(env.device)
