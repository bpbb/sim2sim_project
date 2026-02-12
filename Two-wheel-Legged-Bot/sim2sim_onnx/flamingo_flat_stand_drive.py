"""
Flamingo Sim2Sim Transfer Script - Final Fix (Gear Ratio Physics)

This script is designed to match the EXACT observation structure and actuator physics from:
  logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/params/env.yaml

Key Fixes:
1. Gear Ratio Physics: `GearDelayedPDActuator` with gear_ratio=-1.5 means:
   - Effective Stiffness (Kp) = config_Kp * gear_ratio^2
   - Effective Damping (Kd) = config_Kd * gear_ratio^2
   - Target Position (Joint) = Action (Motor) / gear_ratio
2. Observation Structure: Matches training config (28 dims per frame).
3. Command Scaling: Matches training config (2.0, 0.0, 0.25, 1.0).

Observation structure per frame (28 dims):
  1. hip_shoulder_joint_pos (4): [LH, RH, LS, RS] - no scaling
  2. leg_joint_pos (2): [LL, RL] * gear_ratio (-1.5)
  3. hip_shoulder_joint_vel (4): * 0.15
  4. leg_joint_vel (2): * gear_ratio (-1.5) * 0.15
  5. wheel_joint_vel (2): * 0.15
  6. base_ang_vel (3): * 0.25
  7. base_projected_gravity (3): no scaling
  8. last_action (8): no scaling
"""

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import mujoco
from scipy.spatial.transform import Rotation as R
import onnxruntime as ort
import torch
import math
from prettytable import PrettyTable
import pygame
import time
import argparse
import threading


class FlamingoFlatStandDrive(MujocoEnv, utils.EzPickle):
    """
    MuJoCo environment matching Flamingo_Flat_Stand_Drive IsaacLab training config.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(self, 
                 env_id='FlamingoFSD-v0',
                 model_path='./assets/flamingo_torque_velocity.xml',
                 frame_skip=4,  # Decimation 4 in training (50Hz)
                 render_mode='human',
                 num_stacks=3):
        
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.dt_ = 0.005  # Simulation timestep
        self.sim_duration = 60  # Simulation duration (seconds)
        self.sim_step = (self.sim_duration / self.dt_) / self.frame_skip
        self.id = env_id
        
        # Observation dimensions matching Flamingo_Flat_Stand_Drive config
        self.obs_per_frame = 28
        self.num_stacks = num_stacks  # Number of stacked frames
        self.num_commands = 4  # vx, vy, wz, pos_z
        self.obs_dim = self.obs_per_frame * self.num_stacks + self.num_commands
        
        print(f"Observation dimension: {self.obs_dim} = {self.obs_per_frame}×{self.num_stacks} + {self.num_commands}")
        
        self.act_dim = 8
        self.filtered_action = None
        
        # State history for stacking
        self.previous_states = []
        self.step_counter = 0
        self.state_clip = 100.0  # Increased clip range
        
        # Gear ratio for leg joints (from training config)
        self.leg_gear_ratio = -1.5
        
        # JOINT OFFSET CALIBRATION
        # After testing: Isaac Lab and MuJoCo have SAME joint zero (0 = standing).
        # The issue was HEIGHT offset: Isaac Lab uses 0.55m, MuJoCo XML default is 0.25m.
        # MuJoCo world frame origin is different from Isaac Lab.
        self.leg_joint_offset = 0.0  # No joint offset needed - same zero definition
        
        # Observation scaling factors (from training config env.yaml)
        self.joint_vel_scale = 0.15
        self.ang_vel_scale = 0.25
        
        # Command scaling (from generated_scaled_commands: scale=(2.0, 0.0, 0.25))
        self.cmd_scale_vx = 2.0
        self.cmd_scale_vy = 0.0  # Disabled for 2-wheel
        self.cmd_scale_wz = 0.25
        self.cmd_scale_pz = 1.0
        
        self.commands = np.zeros(4)
        self.commands[3] = 0.246  # Nominal height command for MuJoCo model (matches q=0)
        self.max_linear_speed = 1.0
        self.max_angular_speed = 2.5
        self.acceleration = 0.005
        self.deceleration = 0.01
        
        self.plot_log = True
        
        # MuJoCo joint indices
        # qpos order: base(0-6), LH(7), LS(8), LL(9), LW(10), RH(11), RS(12), RL(13), RW(14)
        # qvel order: base(0-5), LH(6), LS(7), LL(8), LW(9), RH(10), RS(11), RL(12), RW(13)
        
        # Joint position indices for observation
        self.hip_shoulder_qpos_idx = [7, 11, 8, 12]  # [LH, RH, LS, RS]
        self.leg_qpos_idx = [9, 13]  # [LL, RL]
        
        # Joint velocity indices for observation
        self.hip_shoulder_qvel_idx = [6, 10, 7, 11]  # [LH, RH, LS, RS]
        self.leg_qvel_idx = [8, 12]  # [LL, RL]
        self.wheel_qvel_idx = [9, 13]  # [LW, RW]
        
        # Sign constants (Matching XML Axis Definitions)
        # XML defines right side joints with axis="0 0 -1" (mirrored).
        # This allows symmetrical actions to produce symmetrical motion.
        self.hip_shoulder_pos_sign = np.ones(4)
        self.leg_pos_sign = np.ones(2)
        self.hip_shoulder_vel_sign = np.ones(4)
        self.leg_vel_sign = np.ones(2)
        self.wheel_vel_sign = np.ones(2)
        self.action_sign = np.ones(8)
        
        # Action sign (Symmetrical)
        # self.action_sign = np.ones(8)
        
        utils.EzPickle.__init__(self)
        
        MujocoEnv.__init__(
            self,
            model_path=self.model_path,
            frame_skip=self.frame_skip,
            observation_space=Box(
                low=-self.state_clip,
                high=self.state_clip,
                shape=(self.obs_dim,),
                dtype=np.float32
            ),
            render_mode=self.render_mode,
        )
        
        # Override torque limits in model (must be after MujocoEnv.__init__)
        # XML limits to 23Nm, but Env limits to 60Nm.
        self.model.actuator_ctrlrange[:] = np.array([[-60, 60]] * 8)
        self.model.actuator_forcerange[:] = np.array([[-60, 60]] * 8)
        
        # Set print options for cleaner logs
        np.set_printoptions(precision=3, suppress=True)

    def draw_keyboard(self, keys):
        self.screen.fill((0, 0, 0))
        key_positions = {
            pygame.K_UP: (75, 50, 50, 50),
            pygame.K_DOWN: (75, 150, 50, 50),
            pygame.K_LEFT: (25, 100, 50, 50),
            pygame.K_RIGHT: (125, 100, 50, 50)
        }
        for key, pos in key_positions.items():
            color = (255, 0, 0) if keys[key] else (0, 255, 0)
            pygame.draw.rect(self.screen, color, pos)
        
        # Display current commands
        font = pygame.font.Font(None, 24)
        cmd_text = font.render(f"vx:{self.commands[0]:.2f} wz:{self.commands[2]:.2f} h:{self.commands[3]:.2f}", True, (255, 255, 255))
        self.screen.blit(cmd_text, (180, 100))

    def update_commands(self):
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        # Linear velocity (up/down keys)
        if keys[pygame.K_UP]:
            self.commands[0] = min(self.commands[0] + self.acceleration, self.max_linear_speed)
        elif keys[pygame.K_DOWN]:
            self.commands[0] = max(self.commands[0] - self.acceleration, -self.max_linear_speed)
        else:
            if self.commands[0] > 0:
                self.commands[0] = max(self.commands[0] - self.deceleration, 0)
            elif self.commands[0] < 0:
                self.commands[0] = min(self.commands[0] + self.deceleration, 0)
        
        # Angular velocity (left/right keys)
        if keys[pygame.K_LEFT]:
            self.commands[2] = min(self.commands[2] + self.acceleration, self.max_angular_speed)
        elif keys[pygame.K_RIGHT]:
            self.commands[2] = max(self.commands[2] - self.acceleration, -self.max_angular_speed)
        else:
            if self.commands[2] > 0:
                self.commands[2] = max(self.commands[2] - self.deceleration, 0)
            elif self.commands[2] < 0:
                self.commands[2] = min(self.commands[2] + self.deceleration, 0)
        
        # vy is always 0 for 2-wheel robot
        self.commands[1] = 0.0
        # Height remains at default
        
        self.draw_keyboard(keys)

    def compute_projected_gravity(self, quat_wxyz):
        """
        Compute projected gravity in body frame.
        Same as IsaacLab's projected_gravity function.
        
        Args:
            quat_wxyz: quaternion in [w, x, y, z] format
        Returns:
            gravity vector in body frame (3,)
        """
        # World gravity vector
        gravity_world = np.array([0, 0, -1.0])
        
        # Convert quaternion to rotation (scipy expects xyzw)
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        r = R.from_quat(quat_xyzw)
        
        # Rotate gravity to body frame (inverse rotation)
        gravity_body = r.apply(gravity_world, inverse=True)
        
        return gravity_body

    def _get_obs(self, action):
        """
        Build observation matching Flamingo_Flat_Stand_Drive config.
        """
        # Get quaternion from sensor (MuJoCo returns [w, x, y, z])
        quat_wxyz = self.data.sensor('orientation').data.astype(np.double)
        if np.allclose(quat_wxyz, 0):
            quat_wxyz = np.array([1, 0, 0, 0])
        
        # 1. Hip-shoulder joint positions [LH, RH, LS, RS] (4 dims)
        # Apply sign correction for right side
        hip_shoulder_pos = self.data.qpos[self.hip_shoulder_qpos_idx].astype(np.double)
        hip_shoulder_pos = hip_shoulder_pos * self.hip_shoulder_pos_sign
        
        # 2. Leg joint positions [LL, RL] * gear_ratio (2 dims)
        # Apply sign correction AND gear_ratio (-1.5)
        # Apply offset (subtract from raw MuJoCo to get "Zero-centered" Isaac space)
        leg_pos = self.data.qpos[self.leg_qpos_idx].astype(np.double)
        leg_pos = leg_pos - self.leg_joint_offset
        leg_pos = leg_pos * self.leg_pos_sign * self.leg_gear_ratio
        
        # 3. Hip-shoulder joint velocities [LH, RH, LS, RS] * 0.15 (4 dims)
        hip_shoulder_vel = self.data.qvel[self.hip_shoulder_qvel_idx].astype(np.double)
        hip_shoulder_vel = hip_shoulder_vel * self.hip_shoulder_vel_sign * self.joint_vel_scale
        
        # 4. Leg joint velocities [LL, RL] * gear_ratio * 0.15 (2 dims)
        leg_vel = self.data.qvel[self.leg_qvel_idx].astype(np.double)
        leg_vel = leg_vel * self.leg_vel_sign * self.leg_gear_ratio * self.joint_vel_scale
        
        # 5. Wheel joint velocities [LW, RW] * 0.15 (2 dims)
        wheel_vel = self.data.qvel[self.wheel_qvel_idx].astype(np.double)
        wheel_vel = wheel_vel * self.wheel_vel_sign * self.joint_vel_scale
        
        # 6. Base angular velocity * 0.25 (3 dims)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        ang_vel = ang_vel * self.ang_vel_scale
        
        # 7. Projected gravity (3 dims)
        proj_gravity = self.compute_projected_gravity(quat_wxyz)
        
        # 8. Last action (8 dims)
        # Action order matches IsaacLab: [LH, RH, LS, RS, LL, RL, LW, RW]
        
        # Build current frame observation (28 dims)
        current_state = np.concatenate([
            hip_shoulder_pos,  # 4 dims
            leg_pos,           # 2 dims
            hip_shoulder_vel,  # 4 dims
            leg_vel,           # 2 dims
            wheel_vel,         # 2 dims
            ang_vel,           # 3 dims
            proj_gravity,      # 3 dims
            action,            # 8 dims
        ])
        
        # Stack observations (newest first)
        if len(self.previous_states) < self.num_stacks - 1:
            self.previous_states.insert(0, current_state)
            while len(self.previous_states) < self.num_stacks:
                self.previous_states.append(np.zeros_like(current_state))
        else:
            self.previous_states.pop(-1)
            self.previous_states.insert(0, current_state)
        
        # Scale commands (matching generated_scaled_commands)
        scaled_commands = np.array([
            self.commands[0] * self.cmd_scale_vx,
            self.commands[1] * self.cmd_scale_vy,
            self.commands[2] * self.cmd_scale_wz,
            self.commands[3] * self.cmd_scale_pz,
        ])
        
        # Concatenate: stacked_obs + scaled_commands
        stacked_obs = np.concatenate(self.previous_states)
        full_obs = np.concatenate([stacked_obs, scaled_commands])
        
        return np.clip(full_obs, -self.state_clip, self.state_clip).astype(np.float32)

    def pd_controller(self, kp, target, current, kd, target_vel, current_vel):
        """PD controller for position control."""
        return kp * (target - current) + kd * (target_vel - current_vel)

    def low_pass_filter(self, action, alpha=1.0):
        if self.filtered_action is None:
            self.filtered_action = action.copy()
        else:
            self.filtered_action = alpha * action + (1 - alpha) * self.filtered_action
        return self.filtered_action

    def step(self, action):
        # 1. Action preprocessing
        self.filtered_action = self.low_pass_filter(action)
        
        # Action order from IsaacLab: [LH, RH, LS, RS, LL, RL, LW, RW]
        # Scale: joints * 1.0, wheels * 20.0
        # Make explicit copy to avoid modifying original action
        action_copy = self.filtered_action.copy()
        
        # Wheel velocity scaling
        action_copy[6:8] *= 20.0 
        
        # Apply sign correction for right side joints
        action_total_sign = self.action_sign.copy()
        action_scaled = action_copy * action_total_sign
        
        # Get current joint states for PD control
        # Positions
        pos_lh = self.data.qpos[7]
        pos_rh = self.data.qpos[11]
        pos_ls = self.data.qpos[8]
        pos_rs = self.data.qpos[12]
        pos_ll = self.data.qpos[9]
        pos_rl = self.data.qpos[13]
        
        # Velocities
        vel_lh = self.data.qvel[6]
        vel_rh = self.data.qvel[10]
        vel_ls = self.data.qvel[7]
        vel_rs = self.data.qvel[11]
        vel_ll = self.data.qvel[8]
        vel_rl = self.data.qvel[12]
        vel_lw = self.data.qvel[9]
        vel_rw = self.data.qvel[13]
        
        # PD gains from IsaacLab config
        kp_hip = 100.0
        kp_shoulder = 100.0
        # Leg joint effective PD gains due to Gear Actuator
        # kp_eff = kp * gear^2
        # kd_eff = kd * gear^2
        kp_leg_base = 120.0
        kd = 1.5
        
        gear_sq = self.leg_gear_ratio**2
        kp_leg_eff = kp_leg_base * gear_sq
        kd_leg_eff = kd * gear_sq
        
        wheel_kd = 0.7
        
        # Compute torques via PD control
        torque_lh = np.clip(self.pd_controller(kp_hip, action_scaled[0], pos_lh, kd, 0, vel_lh), -60, 60)
        torque_rh = np.clip(self.pd_controller(kp_hip, action_scaled[1], pos_rh, kd, 0, vel_rh), -60, 60)
        torque_ls = np.clip(self.pd_controller(kp_shoulder, action_scaled[2], pos_ls, kd, 0, vel_ls), -60, 60)
        torque_rs = np.clip(self.pd_controller(kp_shoulder, action_scaled[3], pos_rs, kd, 0, vel_rs), -60, 60)
        
        # Leg joints: Target position needs to be scaled by 1/gear_ratio
        # because the policy outputs "motor position", but we need "joint position" target
        # AND offset applied: target_mj = (action / gear) + offset
        target_ll = (action_scaled[4] / self.leg_gear_ratio) + self.leg_joint_offset
        target_rl = (action_scaled[5] / self.leg_gear_ratio) + self.leg_joint_offset
        
        # Compute joint torques using base PD gains (Kp=120)
        torque_joint_ll = self.pd_controller(kp_leg_base, target_ll, pos_ll, kd, 0, vel_ll)
        torque_joint_rl = self.pd_controller(kp_leg_base, target_rl, pos_rl, kd, 0, vel_rl)
        
        # Convert joint torque to motor torque (ctrl) by dividing by gear_ratio
        # This handles the sign inversion and magnitude correctly (1/1.5)
        torque_ll = np.clip(torque_joint_ll / self.leg_gear_ratio, -60, 60)
        torque_rl = np.clip(torque_joint_rl / self.leg_gear_ratio, -60, 60)
        
        # Wheel velocity control (damping)
        torque_lw = np.clip(wheel_kd * (action_scaled[6] - vel_lw), -36, 36)
        torque_rw = np.clip(wheel_kd * (action_scaled[7] - vel_rw), -36, 36)
        
        # Actuator order in XML: [LH, RH, LS, RS, LL, RL, LW, RW]
        ctrl = np.array([torque_lh, torque_rh, torque_ls, torque_rs, torque_ll, torque_rl, torque_lw, torque_rw])
        
        # Apply to simulation
        self.do_simulation(ctrl, self.frame_skip)
        self.step_counter += 1
        
        # 2. Get NEW observation (one call per step to prevent history corruption)
        obs = self._get_obs(self.filtered_action)
        
        reward = 0.0
        done = self._is_done()
        term = self.step_counter >= self.sim_step
        
        return obs, reward, done, term, {}

    def _is_done(self):
        """Check if episode should terminate."""
        # Check if base link height is too low (crashed)
        # Nominal is 0.246. If it drops below 0.15, it's definitely a fail.
        base_height = self.data.qpos[2]
        if base_height < 0.15:
            return True
        
        # Also check for base contact force
        contact_forces = self.data.cfrc_ext[1:10]
        base_contact = np.any(np.abs(contact_forces[0]) > 10.0) # Increased threshold
        return base_contact

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.previous_states = []
        self.step_counter = 0
        self.filtered_action = None
        
        action = np.zeros(8)
        return self._get_obs(action)

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        # Standing height in MuJoCo model is 0.246m
        qpos[2] = 0.255 # Small clearance
        qpos[3:7] = np.array([1, 0, 0, 0])  # w, x, y, z quaternion
        return qpos

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        pygame.quit()

    def plot_logged_data(self, log_data):
        """Plot logged data for debugging."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            time_data = log_data['time']
            joint_names = ['LH', 'RH', 'LS', 'RS', 'LL (Gear Safe)', 'RL (Gear Safe)']
            wheel_names = ['LW', 'RW']
            
            # Ensure data is numpy array
            des_joint = np.array(log_data['desired_joint_positions'])
            act_joint = np.array(log_data['actual_joint_positions'])
            des_wheel = np.array(log_data['desired_wheel_velocities'])
            act_wheel = np.array(log_data['actual_wheel_velocities'])
            
            plt.figure("Joint Tracking", figsize=(16, 12))
            for i in range(6):
                plt.subplot(4, 2, i + 1)
                if i < des_joint.shape[1]:
                    plt.plot(time_data, des_joint[:, i], label=f'Des {joint_names[i]}')
                if i < act_joint.shape[1]:
                    plt.plot(time_data, act_joint[:, i], label=f'Act {joint_names[i]}')
                plt.xlabel('Time (s)')
                plt.ylabel('Position (rad)')
                plt.grid(True)
                plt.legend()
            
            for i in range(2):
                plt.subplot(4, 2, 7 + i)
                # Wheel velocities are already scaled in log_data['desired_wheel_velocities']
                if i < des_wheel.shape[1]:
                    plt.plot(time_data, des_wheel[:, i], label=f'Des {wheel_names[i]}')
                if i < act_wheel.shape[1]:
                    plt.plot(time_data, act_wheel[:, i], label=f'Act {wheel_names[i]}')
                plt.xlabel('Time (s)')
                plt.ylabel('Velocity (rad/s)')
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            output_file = 'sim2sim_tracking.png'
            plt.savefig(output_file)
            print(f"Plot saved to {output_file}")
            plt.close()
            
        except Exception as e:
            print(f"Error plotting data: {e}")
            import traceback
            traceback.print_exc()

    def run_mujoco(self, session, input_name, env, sim_hz=200, decimation=2):
        """Run simulation with ONNX policy."""
        log_data = defaultdict(list)
        dt = (1.0 / sim_hz) * self.frame_skip
        
        while True:
            obs, _ = env.reset()
            # ... rest of function remains simliar but we need to ensure plot_logged_data is called correctly
            
            for step in tqdm(range(int(self.sim_step)), desc="Running sim2sim..."):
                current_time = step * dt
                
                # Run ONNX inference
                obs_tensor = obs.astype(np.float32)
                action = session.run(None, {input_name: obs_tensor[np.newaxis, :]})[0][0]
                action_clipped = np.clip(action, -1, 1)
                
                obs, rewards, dones, term, info = env.step(action_clipped)
                env.render()
                
                # Log data
                log_data['time'].append(current_time)
                log_data['desired_joint_positions'].append(action_clipped[:6].copy())
                # For graph, show 'effective' motor desired vs actual motor pos (joint * gear)
                # to match what policy sees
                log_data['actual_joint_positions'].append(
                    np.array([
                        env.data.qpos[7], env.data.qpos[11], 
                        env.data.qpos[8], env.data.qpos[12], 
                        env.data.qpos[9] * -1.5, env.data.qpos[13] * -1.5  # Leg scaled by gear ratio
                    ])
                )
                
                # Log wheel velocities (this was already correct in run_mujoco, but verify)
                log_data['desired_wheel_velocities'].append(action_clipped[6:8] * 20.0)
                
                log_data['actual_wheel_velocities'].append(
                    np.array([env.data.qvel[9], env.data.qvel[13]])
                )
                log_data['commands'].append(self.commands.copy())

                if dones or term:
                    print(f"Episode ended at step {step} ({current_time:.2f}s). Reason: {'Max time' if term else 'Contact'}")
                    break
            
            if self.plot_log:
                self.plot_logged_data(log_data)
                
            # Only run once then exit to analyze data
            break 
            



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flamingo Sim2Sim Transfer - Flat Stand Drive Config')
    parser.add_argument('--load_model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--num_stacks', type=int, default=3, help='Number of observation stacks (default: 3)')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
    args = parser.parse_args()
    
    # Load ONNX model
    session = ort.InferenceSession(args.load_model)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Print model info
    print(f"Model input name: {input_name}")
    print(f"Model input shape: {input_shape}")
    
    expected_dim = 28 * args.num_stacks + 4
    print(f"Expected observation dim with {args.num_stacks} stacks: {expected_dim}")
    
    if input_shape[1] != expected_dim:
        print(f"WARNING: Model expects {input_shape[1]} dims but script configured for {expected_dim}")
        suggested_stacks = (input_shape[1] - 4) // 28
        print(f"Suggested --num_stacks: {suggested_stacks}")
    
    env = FlamingoFlatStandDrive(env_id="FlamingoFSD-v0", num_stacks=args.num_stacks)
    env.plot_log = not args.no_plot
    env.render_mode = "human"
    
    # Run simulation directly in main thread for visualization
    env.run_mujoco(session, input_name, env, 200, 2)
