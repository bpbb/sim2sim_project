"""
Flamingo Sim2Sim Transfer Script - Fixed Version

This script is designed to match the exact observation structure from:
  Isaac-Velocity-Flat-Flamingo-Sim2Sim-v1-ppo training environment

Key fixes:
1. Command dimension: 3 (vx, vy, wz) instead of 4
2. Observation structure: 28 dims per frame, 3 frames stacked = 84 + 3 commands = 87 total
3. Joint axis inversions for right side joints accounted for
4. Correct joint ordering matching IsaacLab
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
import random
import time
import argparse
import threading


class FlamingoSim2SimFixed(MujocoEnv, utils.EzPickle):
    """
    Fixed Flamingo environment for sim2sim transfer from IsaacLab.
    
    Observation structure per frame (28 dims):
    - joint_pos (6): [LH, RH, LS, RS, LL, RL] - raw positions
    - joint_vel (8): [LH, RH, LS, RS, LL, RL, LW, RW] - raw velocities
    - base_ang_vel (3): angular velocity in body frame
    - base_euler (3): roll, pitch, yaw
    - last_action (8): previous action
    
    Stacked: 3 frames = 84 dims
    Commands: 3 dims (vx, vy, wz)
    Total: 87 dims
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
    }

    def __init__(self, 
                 env_id='FlamingoSim2Sim-v0',
                 model_path='./assets/flamingo_torque_velocity.xml',
                 frame_skip=2, 
                 render_mode='human'):
        
        self.model_path = os.path.join(os.path.dirname(__file__), model_path)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.dt_ = 0.005  # Simulation timestep
        self.sim_duration = 30  # Simulation duration (seconds)
        self.sim_step = (self.sim_duration / self.dt_) / self.frame_skip
        self.id = env_id
        
        # Observation dimensions matching sim2sim training config
        self.obs_per_frame = 28  # joint_pos(6) + joint_vel(8) + ang_vel(3) + euler(3) + action(8)
        self.num_stacks = 3
        self.num_commands = 3  # vx, vy, wz (NOT 4!)
        self.obs_dim = self.obs_per_frame * self.num_stacks + self.num_commands  # 28*3 + 3 = 87
        
        self.act_dim = 8
        self.action_scaler = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0, 20.0])
        self.filtered_action = None
        
        # State history for stacking
        self.previous_states = []
        self.step_counter = 0
        self.state_clip = 25
        
        # Pygame for keyboard control
        pygame.init()
        self.screen = pygame.display.set_mode((480, 240))
        pygame.display.set_caption('Flamingo Sim2Sim Controller')
        
        # Commands: [vx, vy, wz] - only 3 dimensions!
        self.commands = np.zeros(3)
        self.max_linear_speed = 1.0
        self.max_angular_speed = 2.5
        self.acceleration = 0.1
        self.deceleration = 0.05
        self.automation_command = False
        
        self.plot_log = True
        
        # Joint indices in MuJoCo qpos/qvel
        # MuJoCo order: base(0-6), LH(7), LS(8), LL(9), LW(10), RH(11), RS(12), RL(13), RW(14)
        self.joint_pos_indices = [7, 11, 8, 12, 9, 13]  # [LH, RH, LS, RS, LL, RL]
        self.joint_vel_indices = [6, 10, 7, 11, 8, 12, 9, 13]  # [LH, RH, LS, RS, LL, RL, LW, RW]
        
        # Right side joints have inverted axes in MuJoCo XML
        # right_shoulder: axis="0 0 -1", right_leg: axis="0 0 -1", right_wheel: axis="0 0 -1"
        # We need to negate observations and actions for these
        self.obs_sign = np.array([1, 1, 1, -1, 1, -1])  # For joint_pos [LH, RH, LS, RS, LL, RL]
        self.vel_sign = np.array([1, 1, 1, -1, 1, -1, 1, -1])  # For joint_vel [all 8]
        self.action_sign = np.array([1, 1, 1, -1, 1, -1, 1, -1])  # For actions [6 pos + 2 vel]
        
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
        )

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
        cmd_text = font.render(f"vx:{self.commands[0]:.2f} wz:{self.commands[2]:.2f}", True, (255, 255, 255))
        self.screen.blit(cmd_text, (200, 100))

    def update_commands(self, automation=False):
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
        
        self.draw_keyboard(keys)

    def quaternion_to_euler(self, quat):
        """Convert quaternion [x, y, z, w] to euler angles [roll, pitch, yaw]."""
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)
        
        # Pitch (y-axis rotation)
        t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        pitch = np.arcsin(t2)
        
        # Yaw (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        
        return np.array([roll, pitch, yaw])

    def _get_obs(self, action):
        """
        Build observation matching Isaac Lab sim2sim training config.
        
        Structure per frame (28 dims):
        - joint_pos (6): [LH, RH, LS, RS, LL, RL] - NO scaling, NO gear ratio
        - joint_vel (8): [LH, RH, LS, RS, LL, RL, LW, RW] - NO scaling
        - base_ang_vel (3): NO scaling
        - base_euler (3): roll, pitch, yaw
        - last_action (8)
        """
        # Get quaternion from sensor (MuJoCo returns [w, x, y, z])
        quat_wxyz = self.data.sensor('orientation').data.astype(np.double)
        # Convert to [x, y, z, w] for scipy
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        
        if np.all(quat_xyzw == 0):
            quat_xyzw = np.array([0, 0, 0, 1])
        
        # Angular velocity in body frame
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.double)
        
        # Euler angles
        euler = self.quaternion_to_euler(quat_xyzw)
        
        # Joint positions [LH, RH, LS, RS, LL, RL] - apply sign correction for right side
        joint_pos = self.data.qpos[self.joint_pos_indices].astype(np.double)
        joint_pos = joint_pos * self.obs_sign
        
        # Joint velocities [LH, RH, LS, RS, LL, RL, LW, RW] - apply sign correction
        joint_vel = self.data.qvel[self.joint_vel_indices].astype(np.double)
        joint_vel = joint_vel * self.vel_sign
        
        # Build current frame observation (28 dims)
        current_state = np.concatenate([
            joint_pos,      # 6 dims - NO scaling (matches env.yaml where scale: null)
            joint_vel,      # 8 dims - NO scaling
            ang_vel,        # 3 dims - NO scaling
            euler,          # 3 dims
            action,         # 8 dims
        ])
        
        # Stack observations (newest first, as in IsaacLab)
        if len(self.previous_states) < self.num_stacks - 1:
            # Initialize with zeros
            self.previous_states.insert(0, current_state)
            while len(self.previous_states) < self.num_stacks:
                self.previous_states.append(np.zeros_like(current_state))
        else:
            # Shift and insert new
            self.previous_states.pop(-1)
            self.previous_states.insert(0, current_state)
        
        # Update commands from keyboard
        self.update_commands(automation=self.automation_command)
        
        # Concatenate: stacked_obs (84) + commands (3) = 87
        stacked_obs = np.concatenate(self.previous_states)
        full_obs = np.concatenate([stacked_obs, self.commands])
        
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
        obs = self._get_obs(action)
        
        # Apply low-pass filter
        filtered_action = self.low_pass_filter(action)
        
        # Scale actions and apply sign correction for right side joints
        action_scaled = filtered_action * self.action_scaler * self.action_sign
        
        # Extract joint positions from current observation for PD control
        pos_hip = self.data.qpos[[7, 11]]  # [LH, RH]
        pos_shoulder = self.data.qpos[[8, 12]]  # [LS, RS]
        pos_leg = self.data.qpos[[9, 13]]  # [LL, RL]
        
        vel_hip = self.data.qvel[[6, 10]]
        vel_shoulder = self.data.qvel[[7, 11]]
        vel_leg = self.data.qvel[[8, 12]]
        
        # PD gains from Isaac Lab config
        kp_hip = 100.0
        kp_shoulder = 100.0
        kp_leg = 120.0
        kd = 1.5
        
        # Compute torques via PD control
        hip_torque = np.clip(
            self.pd_controller(kp_hip, action_scaled[0:2], pos_hip, kd, 0.0, vel_hip),
            -60.0, 60.0
        )
        shoulder_torque = np.clip(
            self.pd_controller(kp_shoulder, action_scaled[2:4], pos_shoulder, kd, 0.0, vel_shoulder),
            -60.0, 60.0
        )
        leg_torque = np.clip(
            self.pd_controller(kp_leg, action_scaled[4:6], pos_leg, kd, 0.0, vel_leg),
            -60.0, 60.0
        )
        
        # Wheel velocity control (damping + velocity target)
        wheel_kd = 0.7
        wheel_vel_current = self.data.qvel[[9, 13]]
        wheel_torque = np.clip(
            wheel_kd * (action_scaled[6:8] - wheel_vel_current),
            -36.0, 36.0
        )
        
        # Combine torques for actuators
        # Actuator order in XML: [LH, RH, LS, RS, LL, RL, LW, RW]
        ctrl = np.concatenate([hip_torque, shoulder_torque, leg_torque, wheel_torque])
        
        # Apply to simulation
        self.do_simulation(ctrl, self.frame_skip)
        self.step_counter += 1
        
        # Get new observation
        obs = self._get_obs(filtered_action)
        
        reward = self._get_reward(obs)
        done = self._is_done()
        term = self.step_counter >= self.sim_step
        
        return obs, reward, done, term, {}

    def _get_reward(self, obs):
        """Placeholder reward function."""
        return 0.0

    def _is_done(self):
        """Check if episode should terminate."""
        # Check for body contact
        contact_forces = self.data.cfrc_ext[1:10]
        base_contact = np.any(contact_forces[0] > 1.0)
        
        return base_contact

    def reset_model(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos()
        self.data.qvel[:] = 0
        self.previous_states = []
        self.step_counter = 0
        self.filtered_action = None
        
        # Initial action
        action = np.zeros(8)
        return self._get_obs(action)

    def initial_qpos(self):
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.35  # Initial height
        qpos[3:7] = np.array([1, 0, 0, 0])  # w, x, y, z quaternion
        return qpos

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        pygame.quit()

    def run_mujoco(self, session, input_name, env, sim_hz=200, decimation=2):
        """Run simulation with ONNX policy."""
        log_data = defaultdict(list)
        dt = (1.0 / sim_hz) * self.frame_skip
        
        while True:
            obs, _ = env.reset()
            
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
                log_data['desired_wheel_velocities'].append(action_clipped[6:8] * 20.0)
                log_data['actual_joint_positions'].append(env.data.qpos[[7, 11, 8, 12, 9, 13]].copy())
                log_data['actual_wheel_velocities'].append(env.data.qvel[[9, 13]].copy())
                log_data['joint_torques'].append(env.data.actuator_force[:6].copy())
                log_data['wheel_torques'].append(env.data.actuator_force[6:8].copy())
                log_data['commands'].append(self.commands.copy())
                
                if dones or term:
                    break
            
            if self.plot_log:
                self.plot_logged_data(log_data)
            log_data = defaultdict(list)
            env.reset()

    def plot_logged_data(self, log_data):
        """Plot logged data for debugging."""
        time = log_data['time']
        joint_names = ['LH', 'RH', 'LS', 'RS', 'LL', 'RL']
        wheel_names = ['LW', 'RW']
        
        plt.figure("Joint Tracking", figsize=(16, 10))
        for i in range(6):
            plt.subplot(4, 2, i + 1)
            plt.plot(time, np.array(log_data['desired_joint_positions'])[:, i], label=f'Des {joint_names[i]}')
            plt.plot(time, np.array(log_data['actual_joint_positions'])[:, i], label=f'Act {joint_names[i]}')
            plt.xlabel('Time (s)')
            plt.ylabel('Position (rad)')
            plt.ylim(-1.5, 1.5)
            plt.grid(True)
            plt.legend()
        
        for i in range(2):
            plt.subplot(4, 2, 7 + i)
            plt.plot(time, np.array(log_data['desired_wheel_velocities'])[:, i], label=f'Des {wheel_names[i]}')
            plt.plot(time, np.array(log_data['actual_wheel_velocities'])[:, i], label=f'Act {wheel_names[i]}')
            plt.xlabel('Time (s)')
            plt.ylabel('Velocity (rad/s)')
            plt.ylim(-30, 30)
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flamingo Sim2Sim Transfer Script (Fixed)')
    parser.add_argument('--load_model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--no_plot', action='store_true', help='Disable plotting')
    args = parser.parse_args()
    
    # Load ONNX model
    session = ort.InferenceSession(args.load_model)
    input_name = session.get_inputs()[0].name
    
    # Print model info
    print(f"Model input name: {input_name}")
    print(f"Model input shape: {session.get_inputs()[0].shape}")
    print(f"Expected observation dim: 87 (28*3 + 3)")
    
    env = FlamingoSim2SimFixed(env_id="FlamingoSim2Sim-v0")
    env.plot_log = not args.no_plot
    env.render_mode = "human"
    
    mujoco_thread = threading.Thread(target=env.run_mujoco, args=(session, input_name, env, 200, 2))
    mujoco_thread.start()
    
    try:
        pygame.init()
        clock = pygame.time.Clock()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    mujoco_thread.join()
                    exit()
            pygame.display.flip()
            clock.tick(100)
    
    except KeyboardInterrupt:
        pygame.quit()
        mujoco_thread.join()
