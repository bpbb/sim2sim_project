#!/usr/bin/env python3
"""Deploy Isaac Lab trained Cassie policy in MuJoCo.

Loads a TorchScript policy (exported from Isaac Lab RSL-RL training) and
runs it in the MuJoCo simulator with proper joint mapping, observation
construction, and PD torque control.

Supports two policy variants:
  - Active ankles (12 actions, 48 obs): original training with all joints actuated
  - Passive ankles (10 actions, 46 obs): matches MuJoCo 4-bar linkage, no ankle actuation

Key Cassie-specific considerations:
  - MuJoCo has 10 actuators; Isaac Lab has 12 joints (ankle is passive via
    4-bar linkage in MuJoCo). For 12-action policies, ankle actions are dropped.
    For 10-action (passive) policies, actions map directly to active joints.
  - Observations are NOT scaled (Cassie DirectRL env uses raw sensor values).
  - MuJoCo free joint qvel[3:6] is angular velocity in BODY frame (no rotation).
    qvel[0:3] is linear velocity in WORLD frame (needs rotation to body frame).
  - Isaac Lab and MuJoCo have different joint angle zero points for the
    same physical pose. The deploy uses MuJoCo home keyframe values as
    defaults (not Isaac Lab defaults) for observation offsets and PD targets.
  - PD gains vary by joint type (hips=100, knees=200, toes=20).
  - MuJoCo actuators are motor-type: ctrl = torque / gear.

Joint order mapping:
  Isaac Lab (12):   [hip_abd_L, hip_rot_L, hip_flex_L, thigh_L, ankle_L, toe_L,
                     hip_abd_R, hip_rot_R, hip_flex_R, thigh_R, ankle_R, toe_R]
  MuJoCo (10):      [left-hip-roll, left-hip-yaw, left-hip-pitch, left-knee, left-foot,
                     right-hip-roll, right-hip-yaw, right-hip-pitch, right-knee, right-foot]

Usage:
    python scripts/cassie/deploy_mujoco.py scripts/cassie/cassie_passive.yaml
    python scripts/cassie/deploy_mujoco.py scripts/cassie/cassie_isaaclab.yaml --policy /path/to/policy.pt
"""

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml


# ── Isaac Lab joint definitions ────────────────────────────────────────────────
ISAAC_JOINT_NAMES = [
    "hip_abduction_left",   # 0
    "hip_abduction_right",  # 1
    "hip_rotation_left",    # 2
    "hip_rotation_right",   # 3
    "hip_flexion_left",     # 4
    "hip_flexion_right",    # 5
    "thigh_joint_left",     # 6
    "thigh_joint_right",    # 7
    "ankle_joint_left",     # 8  (PASSIVE in MuJoCo)
    "ankle_joint_right",    # 9  (PASSIVE in MuJoCo)
    "toe_joint_left",       # 10
    "toe_joint_right",      # 11
]

# Isaac Lab joint name -> MuJoCo joint name (10 matched pairs, 2 unmapped ankles)
ISAAC_TO_MUJOCO_JOINT = {
    "hip_abduction_left":  "left-hip-roll",
    "hip_rotation_left":   "left-hip-yaw",
    "hip_flexion_left":    "left-hip-pitch",
    "thigh_joint_left":    "left-knee",
    "toe_joint_left":      "left-foot",
    "hip_abduction_right": "right-hip-roll",
    "hip_rotation_right":  "right-hip-yaw",
    "hip_flexion_right":   "right-hip-pitch",
    "thigh_joint_right":   "right-knee",
    "toe_joint_right":     "right-foot",
}

# MuJoCo actuator name -> Isaac Lab gain key
ACTUATOR_TO_GAIN_KEY = {
    "left-hip-roll":   "hip_abduction",
    "left-hip-yaw":    "hip_rotation",
    "left-hip-pitch":  "hip_flexion",
    "left-knee":       "thigh_joint",
    "left-foot":       "toe_joint",
    "right-hip-roll":  "hip_abduction",
    "right-hip-yaw":   "hip_rotation",
    "right-hip-pitch": "hip_flexion",
    "right-knee":      "thigh_joint",
    "right-foot":      "toe_joint",
}


# Passive ankle: Isaac Lab indices that are passive in MuJoCo 4-bar linkage
PASSIVE_ANKLE_INDICES = [8, 9]  # ankle_joint_left, ankle_joint_right
# Active joint indices (everything except ankles) for 10-action passive policy
ACTIVE_ISAAC_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
# [abdL, abdR, rotL, rotR, flexL, flexR, thighL, thighR, toeL, toeR]


def get_gravity_orientation(body_xmat):
    """Project world gravity [0, 0, -1] into body frame using rotation matrix."""
    # R is world-to-body, so R = body_xmat.T
    # gravity_body = R * [0, 0, -1] = - (3rd row of body_xmat)
    # xmat is 3x3 flattened (9 entries)
    return -body_xmat[6:9]


def quat_rotate_inverse(q, v):
    """Rotate vector by inverse of quaternion (world -> body frame).

    Quaternion format: (w, x, y, z).
    Must match Go2 deploy: v - w*t + cross(q_vec, t) for INVERSE rotation.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    q_vec = np.array([x, y, z])
    t = 2.0 * np.cross(q_vec, v)
    return v - w * t + np.cross(q_vec, t)


class CassieDeployer:
    """Deploys an Isaac Lab trained Cassie policy in MuJoCo."""

    def __init__(self, config_path, policy_override=None, headless=False):
        self.headless = headless
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.policy_path = policy_override or config["policy_path"]
        xml_path = config["xml_path"]
        self.simulation_duration = config["simulation_duration"]
        self.simulation_dt = config["simulation_dt"]
        self.control_decimation = config["control_decimation"]
        self.action_scale = config["action_scale"]
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.cmd = np.array(config["cmd_init"], dtype=np.float32)
        self.initial_height = config.get("initial_height", 0.9)

        # Default joint positions (Isaac Lab order, 12 dims)
        self.default_angles_isaac = np.array(
            config["default_angles_isaac"], dtype=np.float32
        )

        # PD gain config by joint type
        self.pd_gain_config = config["pd_gains"]
        self.config = config # Save config for later use

        # Load MuJoCo model
        print(f"Loading MuJoCo model: {xml_path}")
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)

        # CRITICAL: Use MuJoCo's native timestep for Cassie.
        # The cassie.xml defines dt=0.0005s (2kHz) because the 4-bar linkage
        # equality constraints need solref_time/dt ≈ 10 solver steps.
        # At dt=0.005s (200Hz from YAML), constraints get only 1 step → unstable.
        # We keep native dt and adjust control_decimation to maintain 50Hz policy.
        native_dt = self.m.opt.timestep  # 0.0005s from XML
        control_period = self.simulation_dt * self.control_decimation  # 0.02s = 50Hz
        # Adjusted control_decimation: 40
        self.control_decimation = int(round(control_period / native_dt))
        
        # Action smoothing and obs filtering
        self.action_alpha = 0.5
        self.obs_alpha = 0.3
        self.action_prev = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_joint_vel_prev = np.zeros(12, dtype=np.float32)


        # Discover joint and actuator mapping
        self._build_mapping()

        # Build PD gain arrays.
        self.pd_gain_scale = config.get("pd_gain_scale", 15.0)
        self._build_pd_gains()

        # CRITICAL: Expand motor ctrlrange to match Isaac Lab effort limits.
        # The cassie.xml motor ctrlrange is too restrictive:
        #   hip-roll: gear=25 × ctrl±4.5 = ±112Nm (training allows ±200Nm)
        # This severely limits lateral balance torque. Fix: set ctrlrange
        # so that gear × ctrlrange = effort_limit.
        self._expand_ctrlrange()

        # Load policy
        print(f"Loading policy: {self.policy_path}")
        self.policy = torch.jit.load(self.policy_path, map_location="cpu")
        self.policy.eval()

        # Detect passive ankle mode from action dimension
        self.passive_ankles = (self.num_actions == 10)
        if self.passive_ankles:
            # 10-action policy: actions map to active Isaac Lab joints only
            self.active_isaac_indices = ACTIVE_ISAAC_INDICES
            print(f"\n  Passive ankle mode: 10 actions -> 10 active joints")
        else:
            # 12-action policy: actions map to all Isaac Lab joints
            self.active_isaac_indices = list(range(12))
            print(f"\n  Active ankle mode: 12 actions -> 12 joints (ankles dropped in MuJoCo)")

        # Buffers
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0

        # Feedforward bias correction (optional)
        ff_config = config.get("feedforward_bias", {})
        self.ff_enabled = ff_config.get("enabled", False)
        if self.ff_enabled:
            self.ff_offsets = np.array(ff_config.get("offsets", [0.0] * self.num_actions), dtype=np.float32)
            print(f"\n  Feedforward bias ENABLED:")
            print(f"    Offsets: {self.ff_offsets.tolist()}")
        else:
            self.ff_offsets = np.zeros(self.num_actions, dtype=np.float32)

        print(f"\nDeployment ready:")
        print(f"  Physics: {1/self.m.opt.timestep:.0f} Hz")
        print(f"  Control: {1/(self.m.opt.timestep * self.control_decimation):.0f} Hz")
        print(f"  Action scale: {self.action_scale}")
        print(f"  Obs dims: {self.num_obs}, Action dims: {self.num_actions}")
        print(f"  Command: vx={self.cmd[0]:.2f}, vy={self.cmd[1]:.2f}, wz={self.cmd[2]:.2f}")

    def _build_mapping(self):
        """Discover joint/actuator mapping from MuJoCo model."""
        # Find pelvis body
        self.pelvis_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, "cassie-pelvis"
        )
        assert self.pelvis_id >= 0, "Could not find 'cassie-pelvis' body"

        # Map Isaac Lab joints -> MuJoCo qpos/qvel indices
        self.isaac_to_qpos = np.full(12, -1, dtype=np.int64)
        self.isaac_to_qvel = np.full(12, -1, dtype=np.int64)

        for isaac_idx, isaac_name in enumerate(ISAAC_JOINT_NAMES):
            mj_name = ISAAC_TO_MUJOCO_JOINT.get(isaac_name)
            if mj_name is None:
                continue  # ankle joints have no MuJoCo counterpart
            jnt_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, mj_name)
            if jnt_id >= 0:
                self.isaac_to_qpos[isaac_idx] = self.m.jnt_qposadr[jnt_id]
                self.isaac_to_qvel[isaac_idx] = self.m.jnt_dofadr[jnt_id]

        # Find tarsus joints (proxy for passive ankle)
        self.tarsus_qpos = {}
        self.tarsus_qvel = {}
        for side in ["left", "right"]:
            jnt_id = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}-tarsus"
            )
            if jnt_id >= 0:
                self.tarsus_qpos[side] = self.m.jnt_qposadr[jnt_id]
                self.tarsus_qvel[side] = self.m.jnt_dofadr[jnt_id]

        # Map Isaac Lab 12-dim actions -> MuJoCo actuator indices
        # -1 means passive (no actuator)
        self.isaac_to_actuator = np.full(12, -1, dtype=np.int64)
        mj_act_names = [
            self.m.actuator(i).name for i in range(self.m.nu)
        ]
        for isaac_idx, isaac_name in enumerate(ISAAC_JOINT_NAMES):
            mj_name = ISAAC_TO_MUJOCO_JOINT.get(isaac_name)
            if mj_name is not None and mj_name in mj_act_names:
                self.isaac_to_actuator[isaac_idx] = mj_act_names.index(mj_name)

        # Actuator properties
        self.actuator_gear = np.array(
            [self.m.actuator_gear[i, 0] for i in range(self.m.nu)]
        )
        self.actuator_ctrlrange = np.array(
            [self.m.actuator_ctrlrange[i] for i in range(self.m.nu)]
        )

        # Actuator -> qpos/qvel mapping for PD control
        self.act_qpos_idx = np.zeros(self.m.nu, dtype=np.int64)
        self.act_qvel_idx = np.zeros(self.m.nu, dtype=np.int64)
        for a in range(self.m.nu):
            jnt_id = self.m.actuator_trnid[a, 0]
            self.act_qpos_idx[a] = self.m.jnt_qposadr[jnt_id]
            self.act_qvel_idx[a] = self.m.jnt_dofadr[jnt_id]

        # Print mapping
        print(f"\nJoint mapping ({self.m.nu} MuJoCo actuators):")
        for i, isaac_name in enumerate(ISAAC_JOINT_NAMES):
            mj_name = ISAAC_TO_MUJOCO_JOINT.get(isaac_name, "PASSIVE")
            act_idx = self.isaac_to_actuator[i]
            qpos_idx = self.isaac_to_qpos[i]
            marker = " (passive)" if act_idx < 0 else ""
            print(
                f"  [{i:2d}] {isaac_name:<25s} -> {mj_name:<20s} "
                f"act={act_idx:2d} qpos={qpos_idx:2d}{marker}"
            )

    def _build_pd_gains(self):
        """Build PD gain arrays in MuJoCo actuator order.

        Gains are scaled by pd_gain_scale to compensate for the difference
        between Isaac Lab's implicit PD (constraint-based, very stiff) and
        MuJoCo's explicit PD (force-based, compliance depends on gains).
        """
        self.mj_kp = np.zeros(self.m.nu)
        self.mj_kd = np.zeros(self.m.nu)
        self.mj_effort_limit = np.zeros(self.m.nu)

        mj_act_names = [self.m.actuator(i).name for i in range(self.m.nu)]
        scale = self.pd_gain_scale

        for i, act_name in enumerate(mj_act_names):
            gain_key = ACTUATOR_TO_GAIN_KEY.get(act_name)
            if gain_key and gain_key in self.pd_gain_config:
                kp, kd, effort = self.pd_gain_config[gain_key]
                self.mj_kp[i] = kp * scale
                self.mj_kd[i] = kd * scale
                self.mj_effort_limit[i] = effort

        print(f"\nPD gains (scale={scale:.0f}x for explicit PD):")
        for i, name in enumerate(mj_act_names):
            print(
                f"  [{i}] {name:<18s}  Kp={self.mj_kp[i]:.0f}  "
                f"Kd={self.mj_kd[i]:.1f}  limit={self.mj_effort_limit[i]:.0f}N  "
                f"gear={self.actuator_gear[i]:.0f}"
            )

    def _reset(self):
        """Reset to MuJoCo home keyframe and settle under PD control."""
        # Reset to baseline MuJoCo state
        mujoco.mj_resetData(self.m, self.d)

        # Load kinematically valid home keyframe
        key_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.m, self.d, key_id)

        # 3. Orient pelvis tilted BACK by 5.0 degrees to counteract forward drift
        # q = [cos(-2.5 deg), 0, sin(-2.5 deg), 0]
        # 5.0 degrees = 0.087 rad. half = 0.0436. cos=0.999, sin=-0.0436
        self.d.qpos[3:7] = [0.999, 0, -0.0436, 0]
        print(f"  Initial orientation (q): {self.d.qpos[3:7]}")

        home_height = self.d.qpos[2]
        print(f"\n  Home keyframe height: {home_height:.3f}m")
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

        # Use MuJoCo home keyframe angles as the reference for observations.
        # This is critical: the robot starts in MuJoCo's home pose, so the
        # policy must see joint_pos=0 when the robot is in that pose.
        # Even if Isaac Lab defaults differ, the *physical* standing pose is 
        # usually what matters, and the home keyframe is the best MuJoCo equivalent.
        self.mj_default_angles = np.zeros(12, dtype=np.float32)
        for i in range(12):
            qpos_idx = self.isaac_to_qpos[i]
            if qpos_idx >= 0:
                self.mj_default_angles[i] = self.d.qpos[qpos_idx]
            elif i in PASSIVE_ANKLE_INDICES:
                # Use tarsus angle as ankle reference
                side = "left" if i == 8 else "right"
                if side in self.tarsus_qpos:
                    self.mj_default_angles[i] = self.d.qpos[self.tarsus_qpos[side]]


        print(f"\nJoint Reference (Isaac Lab defaults):")
        for i, name in enumerate(ISAAC_JOINT_NAMES):
            mj_val = self.mj_default_angles[i]
            isaac_val = self.default_angles_isaac[i]
            diff = mj_val - isaac_val
            flag = " (convention diff)" if abs(diff) > 0.1 else ""
            print(
                f"  {name:<25s}  MJ_Ref={mj_val:+7.3f}  "
                f"Isaac={isaac_val:+7.3f}  diff={diff:+7.3f}{flag}"
            )


        # Reset buffers
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.counter = 0

        # Short settle: let the robot make ground contact under PD control.
        # Keep this brief (0.1s) to minimize gravity droop. The robot starts
        # at z=1.006m with feet ~6mm above ground. A short settle is enough
        # for foot contact without compressing the legs significantly.
        dt = self.m.opt.timestep
        settle_steps = 200  # 0.1s at 2kHz
        settle_time = settle_steps * dt
        print(f"\n  Pre-settling ({settle_steps} steps, {settle_time:.2f}s "
              f"at dt={dt}s)...")
        zero_action = np.zeros(self.num_actions, dtype=np.float32)
        for step in range(settle_steps):
            self._apply_action(zero_action)
            mujoco.mj_step(self.m, self.d)

        # Zero velocities for clean policy start
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)
        settled_height = self.d.qpos[2]
        print(f"  Settled: height={settled_height:.3f}m")

        # Check foot positions after settling
        for foot_name in ["left-foot", "right-foot"]:
            body_id = mujoco.mj_name2id(
                self.m, mujoco.mjtObj.mjOBJ_BODY, foot_name
            )
            if body_id >= 0:
                foot_pos = self.d.xpos[body_id]
                print(f"  {foot_name} settled pos: z={foot_pos[2]:.4f}")

        # Print joint deflections from home (shows gravity droop)
        print(f"\n  Joint deflections after settling:")
        for i, name in enumerate(ISAAC_JOINT_NAMES):
            qpos_idx = self.isaac_to_qpos[i]
            if qpos_idx >= 0:
                current = self.d.qpos[qpos_idx]
                default = self.mj_default_angles[i]
                delta = current - default
                if abs(delta) > 0.01:
                    print(f"    {name:<25s}  home={default:+.3f}  "
                          f"settled={current:+.3f}  delta={delta:+.3f}")

        if settled_height < 0.5:
            print("  WARNING: Robot collapsed during settling! "
                  "Check PD gains and constraint stability.")

    def _get_observation(self):
        """Build observation matching Isaac Lab CassieEnv.

        Passive mode (46-dim): [lin_vel(3), ang_vel(3), grav(3), cmd(3),
                                joint_pos(12), joint_vel(12), actions(10)]
        Active mode  (48-dim): [lin_vel(3), ang_vel(3), grav(3), cmd(3),
                                joint_pos(12), joint_vel(12), actions(12)]

        IMPORTANT:
          - No observation scaling (Cassie DirectRL env uses raw values)
          - Uses Isaac Lab defaults as reference offsets
          - Passive mode: ankle obs from tarsus proxy (matches passive training)
        """
        # Base quaternion (w, x, y, z) from body xquat
        base_quat = self.d.xquat[self.pelvis_id]

        # Linear velocity: world frame -> body frame (need rotation)
        lin_vel_world = self.d.qvel[0:3]
        lin_vel_body = quat_rotate_inverse(base_quat, lin_vel_world)

        # Angular velocity: MuJoCo free joint qvel[3:6] is in BODY frame (local).
        ang_vel_body = self.d.qvel[3:6]

        # Projected gravity using rotation matrix
        proj_gravity = get_gravity_orientation(self.d.xmat[self.pelvis_id])
        
        # SPOOFING: Mask the -7 degree backward tilt from the policy
        # The physical robot is tilted back to fight drift.
        # But this makes the policy think it's falling backward and try to correct forward.
        # We remove the tilt component from the observation.
        # -5.0 deg tilt => gravity_body.x approx -0.09
        # We add this to make the policy think it is upright.
        proj_gravity[0] += 0.09
        
        # Renormalize to ensure it's a valid direction vector (policy might expect unit norm)
        proj_gravity /= np.linalg.norm(proj_gravity) + 1e-6

        # Joint positions (12 dims, offset from MuJoCo defaults)
        joint_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            qpos_idx = self.isaac_to_qpos[i]
            if qpos_idx >= 0:
                joint_pos[i] = self.d.qpos[qpos_idx] - self.mj_default_angles[i]
            elif i in PASSIVE_ANKLE_INDICES:
                # Provide real tarsus feedback for passive ankles.
                # Isaac index 8 = left, 9 = right.
                side = "left" if i == 8 else "right"
                if side in self.tarsus_qpos:
                    joint_pos[i] = self.d.qpos[self.tarsus_qpos[side]] - self.mj_default_angles[i]
                else:
                    joint_pos[i] = 0.0

        # Joint velocities (12 dims)
        joint_vel = np.zeros(12, dtype=np.float32)
        for i in range(12):
            qvel_idx = self.isaac_to_qvel[i]
            if qvel_idx >= 0:
                joint_vel[i] = self.d.qvel[qvel_idx]
            elif i in PASSIVE_ANKLE_INDICES:
                # Provide real tarsus velocity for passive ankles.
                side = "left" if i == 8 else "right"
                if side in self.tarsus_qvel:
                    joint_vel[i] = self.d.qvel[self.tarsus_qvel[side]]
                else:
                    joint_vel[i] = 0.0

        # Assemble observation (no scaling — Cassie DirectRL env uses raw values)
        act_start = 36  # actions start after joint_vel
        act_end = act_start + self.num_actions

        # Filter joint velocities to remove MuJoCo high-freq noise
        self.obs_joint_vel_prev = (1 - self.obs_alpha) * self.obs_joint_vel_prev + self.obs_alpha * joint_vel
        joint_vel_filt = self.obs_joint_vel_prev

        # Assemble observation
        self.obs[0:3] = lin_vel_body
        self.obs[3:6] = ang_vel_body
        self.obs[6:9] = proj_gravity
        self.obs[9:12] = self.cmd
        self.obs[12:24] = joint_pos
        self.obs[24:36] = joint_vel_filt
        self.obs[act_start:act_end] = self.action  # use prev filtered action? Isaac uses prev raw action.


        # Safety clip: prevent extreme values from crashing the policy.
        self.obs = np.clip(self.obs, -100.0, 100.0)

    def _expand_ctrlrange(self):
        """Expand motor ctrlrange to match Isaac Lab effort limits.

        The cassie.xml motor ctrlrange limits effective torque:
          torque_max = gear × ctrlrange_max

        For hip-roll: 25 × 4.5 = 112Nm, but training allows 200Nm.
        This 44% torque reduction on the lateral balance joint causes tipping.

        Fix: set ctrlrange = effort_limit / gear for each actuator.
        """
        print(f"\nExpanding motor ctrlrange to match training effort limits:")
        mj_act_names = [self.m.actuator(i).name for i in range(self.m.nu)]

        for i in range(self.m.nu):
            gear = self.actuator_gear[i]
            effort = self.mj_effort_limit[i]
            old_range = self.m.actuator_ctrlrange[i].copy()
            old_max_torque = gear * max(abs(old_range[0]), abs(old_range[1]))

            # New ctrlrange: effort_limit / gear
            new_max_ctrl = effort / gear
            self.m.actuator_ctrlrange[i] = [-new_max_ctrl, new_max_ctrl]
            # Update saved copy too
            self.actuator_ctrlrange[i] = [-new_max_ctrl, new_max_ctrl]

            new_max_torque = gear * new_max_ctrl
            change = "EXPANDED" if new_max_torque > old_max_torque else "reduced"
            print(f"  {mj_act_names[i]:<18s}  "
                  f"{old_max_torque:.0f}Nm -> {new_max_torque:.0f}Nm  ({change})")

    def _apply_action(self, action):
        """Convert policy action to MuJoCo motor torques via external PD.

        Handles both 10-dim (passive ankles) and 12-dim (active ankles) actions.

        Steps:
          1. Expand action to 12-dim Isaac Lab space (insert 0 for passive ankles)
          2. Compute target position: mj_default + action_scale * action_12
          3. Map to MuJoCo actuator targets (10 dims)
          4. PD control: torque = Kp*(target - pos) - Kd*vel
          5. Clip torque to effort limit
          6. Motor control: ctrl = torque / gear (clipped to ctrlrange)
        """
        # Apply feedforward bias correction (before expansion)
        if self.ff_enabled:
            action = action + self.ff_offsets

        # Velocity-proportional correction (disabled - didn't improve stability)
        # The sim-to-sim gap causes drift that builds before correction can help.
        # Keep the code for future experimentation.
        #
        # fwd_vel = self.obs[0]  # m/s
        # vel_p_gain = 0.8  # rad per m/s
        # if abs(fwd_vel) > 0.1:
        #     vel_correction = -vel_p_gain * fwd_vel
        #     if self.num_actions == 10:
        #         action = action.copy()
        #         action[2] += vel_correction
        #         action[7] += vel_correction


        # Expand action to 12-dim Isaac Lab space
        action_12 = np.zeros(12, dtype=np.float32)
        for i, isaac_idx in enumerate(self.active_isaac_indices):
            action_12[isaac_idx] = action[i]

        target_isaac = self.mj_default_angles + self.action_scale * action_12


        # Mechanical coupling hack: Add portion of ankle action to knee target
        # Knee indices in Isaac Lab (interleaved): 6 (left) and 7 (right).
        # Ankle indices in Isaac Lab (interleaved): 8 (left) and 9 (right).
        if self.num_actions == 12:
            target_isaac[6] += 0.5 * self.action_scale * action_12[8]
            target_isaac[7] += 0.5 * self.action_scale * action_12[9]

        # Map to MuJoCo actuator targets (10 dims, skip unmapped ankles)
        mj_targets = np.zeros(self.m.nu)
        for isaac_idx in range(12):
            act_idx = self.isaac_to_actuator[isaac_idx]
            if act_idx >= 0:
                mj_targets[act_idx] = target_isaac[isaac_idx]

        # Current joint state at actuator positions
        current_pos = self.d.qpos[self.act_qpos_idx]
        current_vel = self.d.qvel[self.act_qvel_idx]

        # PD control
        torque = self.mj_kp * (mj_targets - current_pos) - self.mj_kd * current_vel

        # Clip to effort limits
        torque = np.clip(torque, -self.mj_effort_limit, self.mj_effort_limit)

        # Convert to motor control (ctrl = torque / gear)
        ctrl = torque / self.actuator_gear
        ctrl = np.clip(ctrl, self.actuator_ctrlrange[:, 0], self.actuator_ctrlrange[:, 1])

        np.copyto(self.d.ctrl, ctrl)

    def run(self):
        """Main deployment loop with MuJoCo viewer.

        Physics runs at native MuJoCo timestep (2kHz for Cassie) while
        the policy is queried at 50Hz (every control_decimation physics steps).
        Viewer syncs at the policy rate. Real-time pacing is per control cycle.
        """
        self._reset()

        dt = self.m.opt.timestep
        control_dt = dt * self.control_decimation  # 0.02s at 50Hz

        if not self.headless:
            # Viewer loop
            with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
                start = time.time()
                print(f"\nRunning for {self.simulation_duration}s (Viewer Mode)...")
                print(f"  Physics: {1/dt:.0f} Hz (dt={dt}s)")
                print(f"  Policy:  {1/control_dt:.0f} Hz (decimation={self.control_decimation})")
                print("Press Ctrl+C to stop.\n")

                policy_step = 0
                sim_time = 0.0

                while viewer.is_running() and sim_time < self.simulation_duration:
                    step_start = time.time()

                    # Physics substeps at native timestep
                    for _ in range(self.control_decimation):
                        self._apply_action(self.action)
                        mujoco.mj_step(self.m, self.d)

                    policy_step += 1
                    sim_time = policy_step * control_dt

                    # Policy inference
                    self._get_observation()

                    # Safety: reset obs if NaN (prevents policy from exploding)
                    if np.any(np.isnan(self.obs)) or np.any(np.isinf(self.obs)):
                        print(f"  WARNING: NaN/Inf in obs at t={sim_time:.3f}s, zeroing")
                        self.obs[:] = 0.0
                        self.action[:] = 0.0
                    else:
                        obs_slice = self.obs[:self.num_obs]
                        obs_tensor = torch.from_numpy(obs_slice).unsqueeze(0).float()
                        with torch.no_grad():
                            self.action = (
                                self.policy(obs_tensor).detach().numpy().squeeze()
                            )
                        self.action = np.clip(self.action, -1.0, 1.0)

                    # Detailed debug output for first 1.0s
                    if sim_time <= 1.0:
                        height = self.d.qpos[2]
                        act_max = np.max(np.abs(self.action))
                        obs_max = np.max(np.abs(self.obs))
                        print(
                            f"  t={sim_time:5.2f}s  h={height:.3f}m  "
                            f"|act|_max={act_max:.3f}  |obs|_max={obs_max:.3f}  "
                            f"ctrl_max={np.max(np.abs(self.d.ctrl)):.3f}"
                        )
                        # First few steps: print detailed breakdown
                        if policy_step <= 3:
                            print(f"    lin_vel_b: {np.round(self.obs[0:3], 3).tolist()}")
                            print(f"    ang_vel_b: {np.round(self.obs[3:6], 3).tolist()}")
                            print(f"    gravity:   {np.round(self.obs[6:9], 3).tolist()}")
                            print(f"    cmd:       {np.round(self.obs[9:12], 3).tolist()}")
                            print(f"    joint_pos: {np.round(self.obs[12:24], 3).tolist()}")
                            print(f"    joint_vel: {np.round(self.obs[24:36], 3).tolist()}")
                            print(f"    prev_act:  {np.round(self.obs[36:36+self.num_actions], 3).tolist()}")
                            # Show raw policy output
                            obs_slice = self.obs[:self.num_obs]
                            obs_tensor = torch.from_numpy(obs_slice).unsqueeze(0).float()
                            with torch.no_grad():
                                raw_action = self.policy(obs_tensor).detach().numpy().squeeze()
                            print(f"    raw_act:   {np.round(raw_action, 3).tolist()}")


                    # Print status periodically
                    elif policy_step % 250 == 0:  # every 5s
                        height = self.d.qpos[2]
                        fwd_vel = self.d.qvel[0]
                        print(
                            f"  t={sim_time:6.1f}s  "
                            f"height={height:.3f}m  "
                            f"fwd_vel={fwd_vel:+.3f}m/s  "
                            f"cmd=[{self.cmd[0]:.2f},{self.cmd[1]:.2f},{self.cmd[2]:.2f}]"
                        )

                    viewer.sync()

                    # Real-time pacing per control cycle
                    elapsed = time.time() - step_start
                    if elapsed < control_dt:
                        time.sleep(control_dt - elapsed)
        else:
            # Headless loop
            start = time.time()
            print(f"\nRunning for {self.simulation_duration}s (Headless Mode)...")
            print(f"  Physics: {1/dt:.0f} Hz (dt={dt}s)")
            print(f"  Policy:  {1/control_dt:.0f} Hz (decimation={self.control_decimation})")

            policy_step = 0
            sim_time = 0.0

            while sim_time < self.simulation_duration:
                # Physics substeps at native timestep
                for _ in range(self.control_decimation):
                    self._apply_action(self.action)
                    mujoco.mj_step(self.m, self.d)

                policy_step += 1
                sim_time = policy_step * control_dt

                # Policy inference
                self._get_observation()
                
                # Check for collapse/termination
                if self.d.qpos[2] < 0.3: # Fall detection
                     print(f"  FAILURE: Robot collapsed (height < 0.3m) at t={sim_time:.3f}s")
                     break

                # Safety: reset obs if NaN
                if np.any(np.isnan(self.obs)) or np.any(np.isinf(self.obs)):
                    print(f"  WARNING: NaN/Inf in obs at t={sim_time:.3f}s, zeroing")
                    self.obs[:] = 0.0
                    self.action[:] = 0.0
                else:
                    obs_slice = self.obs[:self.num_obs]
                    obs_tensor = torch.from_numpy(obs_slice).unsqueeze(0).float()
                    with torch.no_grad():
                        self.action = (
                            self.policy(obs_tensor).detach().numpy().squeeze()
                        )
                    self.action = np.clip(self.action, -1.0, 1.0)

                # ===== FEEDFORWARD & VELOCITY CORRECTION =====
                # Apply bias from YAML
                if self.config.get("feedforward_bias", {}).get("enabled", False):
                    bias = np.array(self.config["feedforward_bias"]["offsets"], dtype=np.float32)
                    if len(bias) == len(self.action):
                        self.action += bias

                # Anti-drift velocity correction
                # Lean back if moving forward (index 4,5 for hip flexion)
                fwd_vel = self.d.qvel[0] # world x vel
                vel_correction = -1.5 * fwd_vel  # Adjusted for lower action scale
                if self.num_actions == 10:
                    self.action[4] += vel_correction  # Left flexion
                    self.action[5] += vel_correction  # Right flexion

                self.action = np.clip(self.action, -1.0, 1.0)


                # Detailed debug output for first 2.0s
                if sim_time <= 2.0:
                    height = self.d.qpos[2]
                    act_max = np.max(np.abs(self.action))
                    obs_max = np.max(np.abs(self.obs))
                    print(
                        f"  t={sim_time:5.2f}s  h={height:.3f}m  "
                        f"|act|_max={act_max:.3f}  |obs|_max={obs_max:.3f}  "
                        f"ctrl_max={np.max(np.abs(self.d.ctrl)):.3f}"
                    )
                    # First few steps: print detailed breakdown
                    if policy_step <= 3:
                        print(f"    lin_vel_b: [{self.obs[0]:.3f}, {self.obs[1]:.3f}, {self.obs[2]:.3f}]")
                        print(f"    ang_vel_b: [{self.obs[3]:.3f}, {self.obs[4]:.3f}, {self.obs[5]:.3f}]")
                        print(f"    gravity:   [{self.obs[6]:.3f}, {self.obs[7]:.3f}, {self.obs[8]:.3f}]")
                        print(f"    cmd:       [{self.obs[9]:.3f}, {self.obs[10]:.3f}, {self.obs[11]:.3f}]")
                        print(f"    joint_pos: {np.round(self.obs[12:24], 3).tolist()}")
                        print(f"    joint_vel: {np.round(self.obs[24:36], 3).tolist()}")
                        print(f"    prev_act:  {np.round(self.obs[36:36+self.num_actions], 3).tolist()}")
                        # Show raw policy output before clipping
                        raw_action = self.policy(
                            torch.from_numpy(self.obs).unsqueeze(0).float()
                        ).detach().numpy().squeeze()
                        print(f"    raw_act:   {np.round(raw_action, 3).tolist()}")
                        print(f"    act names: [abd_L, rot_L, flex_L, thigh_L, ankle_L, toe_L, "
                              f"abd_R, rot_R, flex_R, thigh_R, ankle_R, toe_R]")

                # Periodic status
                if policy_step % 10 == 0:
                    height = self.d.qpos[2]
                    fwd_vel = self.d.qvel[0]
                    # Print hip flexion action
                    hip_act = self.action[4] if self.num_actions > 4 else 0.0
                    print(
                        f"  t={sim_time:6.2f}s  "
                        f"h={height:.3f}m  "
                        f"vel={fwd_vel:+.3f}m/s  "
                        f"hip={hip_act:+.3f}"
                    )

        total_elapsed = time.time() - start
        print(f"\nSimulation complete: {total_elapsed:.1f}s wall time, "
              f"{sim_time:.1f}s sim time")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy Isaac Lab Cassie policy in MuJoCo"
    )
    parser.add_argument(
        "config_file", type=str,
        help="Path to YAML config file (e.g. scripts/cassie/cassie_isaaclab.yaml)",
    )
    parser.add_argument(
        "--policy", type=str, default=None,
        help="Override policy path from config",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer (headless mode)",
    )
    args = parser.parse_args()

    # Resolve config path
    if not os.path.isabs(args.config_file) and not os.path.exists(args.config_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.config_file = os.path.join(script_dir, args.config_file)

    deployer = CassieDeployer(
        args.config_file, policy_override=args.policy, headless=args.headless
    )
    deployer.run()


if __name__ == "__main__":
    main()
