# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cassie Direct RL environment for bipedal locomotion.

Task: Velocity tracking on flat ground
  - Track commanded linear velocity (x, y)
  - Track commanded angular velocity (yaw)

Observations (48 dim):
    base_lin_vel_b (3) + base_ang_vel_b (3) + projected_gravity (3)
    + commands (3) + joint_pos_offset (12) + joint_vel (12) + actions (12)

Actions (12 dim):
    Joint position targets (offset from default standing pose)

Changes from the original (in isaaclab_tasks/direct/cassie/):
  - Added: dof_pos_limits penalty (was declared in cfg but never computed)
  - Added: feet_slide penalty (from Digit biped config)
  - Fixed: command resampling moved from step() to _pre_physics_step()
  - Simplified: feet_air_time reward to match manager-based pattern
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .cassie_sim2sim_cfg import CassieSim2SimEnvCfg


class CassieEnv(DirectRLEnv):
    """Direct RL environment for Cassie — matching manager-based behavior."""

    cfg: CassieSim2SimEnvCfg

    def __init__(self, cfg: CassieSim2SimEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Command buffer: [lin_vel_x, lin_vel_y, ang_vel_z]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Command resampling timer
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)

        # Find joint indices for reward terms
        self._setup_joint_indices()

        # Find body indices for contact sensor
        self._setup_body_indices()

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "feet_air_time",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "flat_orientation_l2",
                "joint_deviation_hip",
                "joint_deviation_toes",
                "dof_pos_limits",
                "feet_slide",
                "termination_penalty",
            ]
        }

        print(f"[CassieEnv] Initialized with {self.num_envs} environments")
        print(f"[CassieEnv] Action scale: {self.cfg.action_scale}")
        print(f"[CassieEnv] Observation space: {self.cfg.observation_space}")

    # ─────────────────────────────────────────────────────────────────────────
    # Setup helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_joint_indices(self):
        """Find indices for hip and toe joints (reward terms)."""
        joint_names = self._robot.joint_names
        print(f"[CassieEnv] Joint names: {joint_names}")

        # Hip joints (abduction + rotation) — deviation penalty
        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(joint_names) if "hip_abduction" in n or "hip_rotation" in n],
            device=self.device, dtype=torch.long,
        )

        # Toe joints — deviation penalty
        self._toe_joint_ids = torch.tensor(
            [i for i, n in enumerate(joint_names) if "toe_joint" in n],
            device=self.device, dtype=torch.long,
        )

        print(f"[CassieEnv] Hip joint ids: {self._hip_joint_ids.tolist()}")
        print(f"[CassieEnv] Toe joint ids: {self._toe_joint_ids.tolist()}")

        # Joint position limits (for dof_pos_limits penalty)
        joint_limits = self._robot.root_physx_view.get_dof_limits().to(self.device)
        self._joint_pos_limits_lower = joint_limits[0, :, 0]
        self._joint_pos_limits_upper = joint_limits[0, :, 1]

        # Soft limits: start penalizing at soft_ratio of the full range
        limit_range = self._joint_pos_limits_upper - self._joint_pos_limits_lower
        soft_margin = limit_range * (1.0 - self.cfg.dof_pos_limits_soft_ratio) / 2.0
        self._soft_lower = self._joint_pos_limits_lower + soft_margin
        self._soft_upper = self._joint_pos_limits_upper - soft_margin

    def _setup_body_indices(self):
        """Find body indices for contact-based rewards and termination."""
        # Feet bodies (for air time reward and slide penalty)
        self._feet_ids, feet_names = self._contact_sensor.find_bodies(".*toe.*")
        print(f"[CassieEnv] Feet body ids: {self._feet_ids}, names: {feet_names}")

        # Pelvis body (for termination)
        self._pelvis_ids, pelvis_names = self._contact_sensor.find_bodies(".*pelvis.*")
        if len(self._pelvis_ids) == 0:
            self._pelvis_ids, pelvis_names = self._contact_sensor.find_bodies(".*base.*")
        print(f"[CassieEnv] Pelvis body ids: {self._pelvis_ids}, names: {pelvis_names}")

    # ─────────────────────────────────────────────────────────────────────────
    # Scene
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        """Setup the scene with robot, terrain, and sensors."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # Step pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions and handle command resampling."""
        self._actions = actions.clone()
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        )

        # Resample commands periodically (moved here from step() override)
        self._command_time_left -= self.step_dt
        resample_ids = torch.where(self._command_time_left <= 0.0)[0]
        if len(resample_ids) > 0:
            self._resample_commands(resample_ids)
            self._command_time_left[resample_ids] = self.cfg.command_resampling_time

    def _apply_action(self):
        """Apply joint position targets to the robot."""
        self._robot.set_joint_position_target(self._processed_actions)

    # ─────────────────────────────────────────────────────────────────────────
    # Observations
    # ─────────────────────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        """Compute 48-dim observation vector with optional noise."""
        self._previous_actions = self._actions.clone()

        # Raw sensor readings
        lin_vel = self._robot.data.root_lin_vel_b.clone()
        ang_vel = self._robot.data.root_ang_vel_b.clone()
        projected_gravity = self._robot.data.projected_gravity_b.clone()
        joint_pos = (self._robot.data.joint_pos - self._robot.data.default_joint_pos).clone()
        joint_vel = self._robot.data.joint_vel.clone()

        # Add noise if enabled
        if self.cfg.add_noise:
            lin_vel += torch.randn_like(lin_vel) * self.cfg.noise_scales["lin_vel"]
            ang_vel += torch.randn_like(ang_vel) * self.cfg.noise_scales["ang_vel"]
            projected_gravity += (
                torch.randn_like(projected_gravity) * self.cfg.noise_scales["projected_gravity"]
            )
            joint_pos += torch.randn_like(joint_pos) * self.cfg.noise_scales["joint_pos"]
            joint_vel += torch.randn_like(joint_vel) * self.cfg.noise_scales["joint_vel"]

        obs = torch.cat(
            [lin_vel, ang_vel, projected_gravity, self._commands,
             joint_pos, joint_vel, self._actions],
            dim=-1,
        )
        return {"policy": obs}


    # ─────────────────────────────────────────────────────────────────────────
    # Rewards
    # ─────────────────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards matching manager-based implementation."""

        # ── Tracking rewards ─────────────────────────────────────────────────
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]),
            dim=1,
        )
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)

        ang_vel_error = torch.square(
            self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]
        )
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)

        # ── Biped gait reward ────────────────────────────────────────────────
        feet_air_time = self._compute_feet_air_time_biped()

        # ── Penalties ────────────────────────────────────────────────────────
        lin_vel_z_l2 = torch.square(self._robot.data.root_lin_vel_b[:, 2])

        ang_vel_xy_l2 = torch.sum(
            torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1
        )

        dof_torques_l2 = torch.sum(
            torch.square(self._robot.data.applied_torque), dim=1
        )

        dof_acc_l2 = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)

        action_rate_l2 = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )

        flat_orientation_l2 = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        # Joint deviation from default pose
        joint_pos_error = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        joint_deviation_hip = torch.sum(
            torch.abs(joint_pos_error[:, self._hip_joint_ids]), dim=1
        )
        joint_deviation_toes = torch.sum(
            torch.abs(joint_pos_error[:, self._toe_joint_ids]), dim=1
        )

        # FIX: dof_pos_limits — penalize joints near their limits
        dof_pos_limits = self._compute_dof_pos_limits_penalty()

        # Feet slide penalty — penalize foot velocity while in contact
        feet_slide = self._compute_feet_slide_penalty()

        # ── Assemble rewards ─────────────────────────────────────────────────
        rewards = {
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp * self.cfg.track_lin_vel_xy_exp_weight * self.step_dt,
            "track_ang_vel_z_exp": track_ang_vel_z_exp * self.cfg.track_ang_vel_z_exp_weight * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_weight * self.step_dt,
            "lin_vel_z_l2": lin_vel_z_l2 * self.cfg.lin_vel_z_l2_weight * self.step_dt,
            "ang_vel_xy_l2": ang_vel_xy_l2 * self.cfg.ang_vel_xy_l2_weight * self.step_dt,
            "dof_torques_l2": dof_torques_l2 * self.cfg.dof_torques_l2_weight * self.step_dt,
            "dof_acc_l2": dof_acc_l2 * self.cfg.dof_acc_l2_weight * self.step_dt,
            "action_rate_l2": action_rate_l2 * self.cfg.action_rate_l2_weight * self.step_dt,
            "flat_orientation_l2": flat_orientation_l2 * self.cfg.flat_orientation_l2_weight * self.step_dt,
            "joint_deviation_hip": joint_deviation_hip * self.cfg.joint_deviation_hip_weight * self.step_dt,
            "joint_deviation_toes": joint_deviation_toes * self.cfg.joint_deviation_toes_weight * self.step_dt,
            "dof_pos_limits": dof_pos_limits * self.cfg.dof_pos_limits_weight * self.step_dt,
            "feet_slide": feet_slide * self.cfg.feet_slide_weight * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Accumulate for logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _compute_feet_air_time_biped(self) -> torch.Tensor:
        """Compute biped feet air time reward (encourages alternating single stance)."""
        air_time = self._contact_sensor.data.current_air_time[:, self._feet_ids]
        contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]

        in_contact = contact_time > 0.0
        in_mode_time = torch.where(in_contact, contact_time, air_time)

        # Single stance: exactly one foot in contact (walking gait)
        single_stance = torch.sum(in_contact.int(), dim=1) == 1

        # Reward: minimum time in current mode during single stance
        reward = torch.min(
            torch.where(
                single_stance.unsqueeze(-1),
                in_mode_time,
                torch.zeros_like(in_mode_time),
            ),
            dim=1,
        )[0]
        reward = torch.clamp(reward, max=self.cfg.feet_air_time_threshold)

        # No reward when standing still
        reward *= torch.norm(self._commands[:, :2], dim=1) > 0.1

        return reward

    def _compute_dof_pos_limits_penalty(self) -> torch.Tensor:
        """Penalize joints approaching their position limits.

        Uses soft limits: no penalty inside the soft range, linearly
        increasing penalty as joint approaches the hard limit.
        """
        joint_pos = self._robot.data.joint_pos

        # How far below the soft lower limit (positive = violation)
        below_lower = torch.clamp(self._soft_lower - joint_pos, min=0.0)
        # How far above the soft upper limit (positive = violation)
        above_upper = torch.clamp(joint_pos - self._soft_upper, min=0.0)

        # Sum of violations across all joints
        penalty = torch.sum(below_lower + above_upper, dim=1)
        return penalty

    def _compute_feet_slide_penalty(self) -> torch.Tensor:
        """Penalize foot sliding (lateral/longitudinal velocity while in contact)."""
        contact_time = self._contact_sensor.data.current_contact_time[:, self._feet_ids]
        in_contact = (contact_time > 0.0).float()

        # Get foot body velocities in world frame
        # Use the body linear velocity for the feet
        foot_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids, :2]  # xy only

        # Penalize velocity magnitude while in contact
        foot_speed = torch.sum(torch.square(foot_vel), dim=-1)  # (num_envs, num_feet)
        penalty = torch.sum(foot_speed * in_contact, dim=1)

        return penalty

    # ─────────────────────────────────────────────────────────────────────────
    # Termination
    # ─────────────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Pelvis contact → robot fell
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        if len(self._pelvis_ids) > 0:
            pelvis_contact = torch.any(
                torch.max(
                    torch.norm(net_contact_forces[:, :, self._pelvis_ids], dim=-1),
                    dim=1,
                )[0] > 1.0,
                dim=1,
            )
        else:
            # Fallback: height-based termination
            pelvis_contact = self._robot.data.root_pos_w[:, 2] < 0.3

        return pelvis_contact, time_out

    # ─────────────────────────────────────────────────────────────────────────
    # Reset
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments with randomization and logging."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # Spread episode lengths to avoid synchronized resets
        if num_resets == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        # Reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Sample new commands
        self._resample_commands(env_ids)
        self._command_time_left[env_ids] = self.cfg.command_resampling_time

        # ── Robot state reset ────────────────────────────────────────────────
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()

        # Add terrain origin offset
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]

        # Randomize position (x, y)
        default_root_state[:, 0] += torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.reset_position_noise, self.cfg.reset_position_noise
        )
        default_root_state[:, 1] += torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.reset_position_noise, self.cfg.reset_position_noise
        )

        # Randomize yaw (quaternion multiplication)
        yaw_noise = torch.empty(num_resets, device=self.device).uniform_(
            -self.cfg.reset_yaw_noise, self.cfg.reset_yaw_noise
        )
        cos_yaw = torch.cos(yaw_noise / 2)
        sin_yaw = torch.sin(yaw_noise / 2)
        qw = default_root_state[:, 3]
        qx = default_root_state[:, 4]
        qy = default_root_state[:, 5]
        qz = default_root_state[:, 6]
        default_root_state[:, 3] = qw * cos_yaw - qz * sin_yaw
        default_root_state[:, 4] = qx * cos_yaw + qy * sin_yaw
        default_root_state[:, 5] = qy * cos_yaw - qx * sin_yaw
        default_root_state[:, 6] = qz * cos_yaw + qw * sin_yaw

        # Zero initial velocity
        default_root_state[:, 7:] = 0.0

        # Write to simulation
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ── Logging ──────────────────────────────────────────────────────────
        # Add termination penalty for died environments
        terminated_mask = self.reset_terminated[env_ids]
        if torch.any(terminated_mask):
            self._episode_sums["termination_penalty"][
                env_ids[terminated_mask]
            ] += self.cfg.termination_penalty

        # Episode reward logs
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        # Termination reason logs
        extras = dict()
        extras["Episode_Termination/pelvis_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

    # ─────────────────────────────────────────────────────────────────────────
    # Command sampling
    # ─────────────────────────────────────────────────────────────────────────

    def _resample_commands(self, env_ids: torch.Tensor):
        """Sample new velocity commands."""
        n = len(env_ids)
        self._commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            self.cfg.lin_vel_x_range[0], self.cfg.lin_vel_x_range[1]
        )
        self._commands[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            self.cfg.lin_vel_y_range[0], self.cfg.lin_vel_y_range[1]
        )
        self._commands[env_ids, 2] = torch.empty(n, device=self.device).uniform_(
            self.cfg.ang_vel_z_range[0], self.cfg.ang_vel_z_range[1]
        )