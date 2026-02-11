# Sim-to-Sim Transfer: Complete Technical Documentation

This document provides exhaustive technical details of the sim-to-sim transfer project, including all code implementations, mathematical derivations, debugging approaches, and lessons learned.

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Simulator Differences](#2-simulator-differences)
3. [Cassie Robot Experiments](#3-cassie-robot-experiments)
4. [Go2 Robot Experiments](#4-go2-robot-experiments)
5. [Joint Ordering Deep Dive](#5-joint-ordering-deep-dive)
6. [Observation Format Analysis](#6-observation-format-analysis)
7. [PD Control Implementation](#7-pd-control-implementation)
8. [Training Configuration](#8-training-configuration)
9. [Debugging Tools and Scripts](#9-debugging-tools-and-scripts)
10. [Complete Code Reference](#10-complete-code-reference)
11. [Successful Transfer Results](#11-successful-transfer-results) ✅

---

# 1. Project Overview

## 1.1 Goal

Transfer reinforcement learning locomotion policies trained in **Isaac Lab** (using NVIDIA PhysX) to **MuJoCo** physics simulator without retraining.

## 1.2 Why Sim-to-Sim Transfer?

1. **Deployment flexibility**: MuJoCo runs on more hardware (no GPU required)
2. **Validation**: Verify policy generalization across physics engines
3. **Real-world preparation**: If policy transfers between simulators, it's more likely to transfer to real hardware

## 1.3 Challenges

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIM-TO-SIM CHALLENGES                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Joint Ordering     │ Different conventions between sims    │
│  2. Physics Engine     │ PhysX vs MuJoCo contact/friction      │
│  3. Observation Format │ Scaling, frame conventions            │
│  4. Control Interface  │ Position vs Torque control            │
│  5. Built-in Damping   │ MuJoCo has joint damping, Isaac doesn't│
└─────────────────────────────────────────────────────────────────┘
```

---

# 2. Simulator Differences

## 2.1 Isaac Lab (PhysX) Characteristics

```python
# Isaac Lab Go2 configuration
class Go2EnvCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/200,  # 200 Hz physics
        render_interval=4,  # 50 Hz rendering
    )

    # Actuators - explicit PD gains
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        actuators={
            "base_legs": DCMotorCfg(
                stiffness=20.0,  # Kp
                damping=0.5,     # Kd
            ),
        },
    )
```

**Key properties:**
- No built-in joint damping
- Explicit PD gains in actuator config
- PhysX contact solver
- GPU-accelerated parallel simulation

## 2.2 MuJoCo Characteristics

```xml
<!-- From go2.xml MuJoCo model -->
<default>
  <joint armature="0.01" damping="2" />  <!-- Built-in damping! -->
  <motor ctrllimited="true" ctrlrange="-23.7 23.7"/>
</default>
```

**Key properties:**
- Built-in joint damping (damping=2 for all joints)
- Armature (rotor inertia) = 0.01
- Torque limits on actuators
- CPU-based, single environment

## 2.3 Impact on Transfer

```
Isaac Lab PD Control:
    torque = Kp * (target - current) - Kd * velocity

MuJoCo with built-in damping:
    torque_applied = Kp * (target - current) - Kd * velocity
    torque_effective = torque_applied - damping * velocity
                     = Kp * (target - current) - (Kd + damping) * velocity
                     = Kp * (target - current) - (Kd + 2) * velocity
```

This means with same Kp=20, Kd=0.5:
- Isaac Lab effective damping: 0.5
- MuJoCo effective damping: 0.5 + 2.0 = 2.5 (5x higher!)

---

# 3. Cassie Robot Experiments

## 3.1 Cassie Robot Structure

Cassie is a bipedal robot with complex leg mechanism:

```
Cassie Leg Structure (per leg):
├── Hip Roll    (actuated)
├── Hip Yaw     (actuated)
├── Hip Pitch   (actuated)
├── Knee        (actuated)
├── Shin        (PASSIVE - 4-bar linkage)
├── Tarsus      (PASSIVE - 4-bar linkage)
├── Heel Spring (PASSIVE - compliant)
└── Toe         (actuated)

Total: 5 actuated + 3 passive per leg = 16 joints
Active joints: 10 (5 per leg)
```

## 3.2 4-Bar Linkage Problem

```
    Hip ──────┬────── Knee
              │
              │  (4-bar linkage)
              │
    Shin ─────┴────── Tarsus

The shin and tarsus positions are COUPLED to hip and knee.
This coupling is modeled differently in PhysX vs MuJoCo.
```

### Isaac Lab Approach
```python
# Isaac Lab uses constraint solver
# Passive joints are implicitly coupled through physics
```

### MuJoCo Approach
```xml
<!-- MuJoCo requires explicit equality constraints -->
<equality>
    <connect body1="left-shin" body2="left-tarsus" anchor="..."/>
</equality>
```

## 3.3 Decision to Switch to Go2

After analyzing the complexity:

| Aspect | Cassie | Go2 |
|--------|--------|-----|
| Total joints | 16 | 12 |
| Actuated joints | 10 | 12 |
| Passive joints | 6 | 0 |
| 4-bar linkage | Yes | No |
| Complexity | High | Low |

**Decision**: Verify transfer algorithm on Go2 first, then return to Cassie.

---

# 4. Go2 Robot Experiments

## 4.1 Go2 Robot Structure

```
Go2 Quadruped (Unitree):
├── Front Left (FL)
│   ├── FL_hip_joint    (abduction/adduction)
│   ├── FL_thigh_joint  (hip flexion)
│   └── FL_calf_joint   (knee flexion)
├── Front Right (FR)
│   ├── FR_hip_joint
│   ├── FR_thigh_joint
│   └── FR_calf_joint
├── Rear Left (RL)
│   ├── RL_hip_joint
│   ├── RL_thigh_joint
│   └── RL_calf_joint
└── Rear Right (RR)
    ├── RR_hip_joint
    ├── RR_thigh_joint
    └── RR_calf_joint

Total: 12 directly actuated joints (no passive joints)
```

## 4.2 Default Joint Positions

```python
# Isaac Lab default pose (from UNITREE_GO2_CFG)
# Note: Different thigh angles for front vs rear legs
DEFAULT_JOINT_POS_ISAAC = np.array([
    # Hips: FL, FR, RL, RR (slight outward angle)
    0.1, -0.1, 0.1, -0.1,
    # Thighs: FL, FR (0.8 rad), RL, RR (1.0 rad - more bent)
    0.8, 0.8, 1.0, 1.0,
    # Calfs: all at -1.5 rad (bent backward)
    -1.5, -1.5, -1.5, -1.5,
])

# Same pose in MuJoCo order (by leg)
DEFAULT_JOINT_POS_MUJOCO = np.array([
    0.1, 0.8, -1.5,    # FL: hip, thigh, calf
    -0.1, 0.8, -1.5,   # FR: hip, thigh, calf
    0.1, 1.0, -1.5,    # RL: hip, thigh, calf
    -0.1, 1.0, -1.5,   # RR: hip, thigh, calf
])
```

## 4.3 Experiment Timeline

```
Experiment 1: Direct Transfer
    │
    ▼ Failed - robot collapsed

Experiment 2: Increase PD Gains
    │
    ▼ Robot holds position, but policy still fails

Experiment 3: Fix Joint Ordering
    │
    ▼ Observations now correct, but actions still wrong

Experiment 4: Fix Action Ordering
    │
    ▼ Both correct, but extreme actions

Experiment 5: Tune PD + Action Clipping
    │
    ▼ Robot survives but doesn't walk

Current: Investigating observation/physics mismatch
```

---

# 5. Joint Ordering Deep Dive

## 5.1 The Ordering Problem

Isaac Lab and MuJoCo use **different conventions** for joint ordering:

```
ISAAC LAB ORDER (by joint type):
Index:  0    1    2    3    4    5    6    7    8    9   10   11
Joint: FL   FR   RL   RR   FL   FR   RL   RR   FL   FR   RL   RR
       hip  hip  hip  hip thigh thigh thigh thigh calf calf calf calf

MUJOCO ORDER (by leg):
Index:  0    1    2    3    4    5    6    7    8    9   10   11
Joint: FL   FL   FL   FR   FR   FR   RL   RL   RL   RR   RR   RR
       hip thigh calf hip thigh calf hip thigh calf hip thigh calf
```

## 5.2 Creating the Mapping

```python
# Step 1: Define joint names in each order
ISAAC_JOINT_NAMES = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]

MUJOCO_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Step 2: Build mapping by matching names
ISAAC_TO_MUJOCO = []
for isaac_idx, isaac_name in enumerate(ISAAC_JOINT_NAMES):
    mujoco_idx = MUJOCO_JOINT_NAMES.index(isaac_name)
    ISAAC_TO_MUJOCO.append(mujoco_idx)

# Result: ISAAC_TO_MUJOCO = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
# Meaning: Isaac index 0 (FL_hip) -> MuJoCo index 0
#          Isaac index 1 (FR_hip) -> MuJoCo index 3
#          Isaac index 4 (FL_thigh) -> MuJoCo index 1
#          etc.

# Step 3: Build inverse mapping
MUJOCO_TO_ISAAC = []
for mujoco_idx, mujoco_name in enumerate(MUJOCO_JOINT_NAMES):
    isaac_idx = ISAAC_JOINT_NAMES.index(mujoco_name)
    MUJOCO_TO_ISAAC.append(isaac_idx)

# Result: MUJOCO_TO_ISAAC = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
```

## 5.3 Using the Mapping Correctly

**CRITICAL**: The mapping usage depends on what you're converting:

```python
# Converting MuJoCo data TO Isaac order (for observations)
# "For each Isaac index i, get data from MuJoCo index ISAAC_TO_MUJOCO[i]"
isaac_data = mujoco_data[ISAAC_TO_MUJOCO]

# Converting Isaac data TO MuJoCo order (for actions)
# "For each MuJoCo index j, get data from Isaac index MUJOCO_TO_ISAAC[j]"
mujoco_data = isaac_data[MUJOCO_TO_ISAAC]
```

## 5.4 Verification Script

```python
#!/usr/bin/env python3
"""Verify joint order mapping is correct."""

import numpy as np

ISAAC_JOINT_NAMES = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]

MUJOCO_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

# Build mappings
ISAAC_TO_MUJOCO = [MUJOCO_JOINT_NAMES.index(name) for name in ISAAC_JOINT_NAMES]
MUJOCO_TO_ISAAC = [ISAAC_JOINT_NAMES.index(name) for name in MUJOCO_JOINT_NAMES]

print(f"ISAAC_TO_MUJOCO = {ISAAC_TO_MUJOCO}")
print(f"MUJOCO_TO_ISAAC = {MUJOCO_TO_ISAAC}")

# Test with labeled data
mujoco_data = np.array([f"M{i}" for i in range(12)])
isaac_data = mujoco_data[ISAAC_TO_MUJOCO]

print("\nVerification (MuJoCo -> Isaac):")
for i, name in enumerate(ISAAC_JOINT_NAMES):
    expected_mujoco_idx = MUJOCO_JOINT_NAMES.index(name)
    expected = f"M{expected_mujoco_idx}"
    actual = isaac_data[i]
    status = "✓" if expected == actual else "✗"
    print(f"  Isaac[{i}] {name}: expected={expected}, got={actual} {status}")
```

**Output:**
```
ISAAC_TO_MUJOCO = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
MUJOCO_TO_ISAAC = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

Verification (MuJoCo -> Isaac):
  Isaac[0] FL_hip: expected=M0, got=M0 ✓
  Isaac[1] FR_hip: expected=M3, got=M3 ✓
  Isaac[2] RL_hip: expected=M6, got=M6 ✓
  ... (all pass)
```

---

# 6. Observation Format Analysis

## 6.1 Observation Structure (48 dimensions)

```python
"""
Observation vector layout (matching Isaac Lab):

Index Range | Size | Component           | Description
------------|------|---------------------|----------------------------------
[0:3]       | 3    | lin_vel_b           | Base linear velocity in body frame
[3:6]       | 3    | ang_vel_b           | Base angular velocity in body frame
[6:9]       | 3    | projected_gravity   | Gravity vector in body frame
[9:12]      | 3    | commands            | Velocity commands (vx, vy, wz)
[12:24]     | 12   | joint_pos           | Joint positions (offset from default)
[24:36]     | 12   | joint_vel           | Joint velocities
[36:48]     | 12   | last_actions        | Previous actions

Total: 48 dimensions
"""
```

## 6.2 Velocity Transformation

```python
def get_body_frame_velocities(quat_wxyz, lin_vel_world, ang_vel_world):
    """
    Transform world-frame velocities to body frame.

    MuJoCo provides velocities in world frame.
    Isaac Lab provides velocities in body frame.
    We need to transform MuJoCo's world velocities to body frame.

    Args:
        quat_wxyz: Quaternion [w, x, y, z] (MuJoCo convention)
        lin_vel_world: Linear velocity in world frame [vx, vy, vz]
        ang_vel_world: Angular velocity in world frame [wx, wy, wz]

    Returns:
        lin_vel_body, ang_vel_body: Velocities in body frame
    """
    # Convert quaternion to rotation matrix
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat_wxyz)
    rot_mat = rot_mat.reshape(3, 3)

    # Transform to body frame using transpose (inverse rotation)
    # R^T * v_world = v_body
    lin_vel_body = rot_mat.T @ lin_vel_world
    ang_vel_body = rot_mat.T @ ang_vel_world

    return lin_vel_body, ang_vel_body
```

## 6.3 Gravity Orientation Computation

Two methods were tested:

### Method 1: Rotation Matrix (original)
```python
def get_gravity_rotation_matrix(quat_wxyz):
    """Using rotation matrix to project gravity."""
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat_wxyz)
    rot_mat = rot_mat.reshape(3, 3)

    # World gravity points down: [0, 0, -1]
    gravity_world = np.array([0, 0, -1])

    # Project to body frame
    gravity_body = rot_mat.T @ gravity_world
    return gravity_body
```

### Method 2: Unitree Formula (working reference)
```python
def get_gravity_orientation(quaternion):
    """
    Compute gravity vector in body frame from quaternion.
    This is the formula used in Unitree's working deployment script.

    Mathematical derivation:
    Given quaternion q = [w, x, y, z] representing body orientation,
    the gravity vector [0, 0, -1] in body frame is:

    g_body = R^T * [0, 0, -1]

    Using quaternion rotation formula, this simplifies to:
    g_x = 2 * (-z*x + w*y)
    g_y = -2 * (z*y + w*x)
    g_z = 1 - 2*(w^2 + z^2)

    Note: The sign conventions may differ from standard formulas
    due to different axis conventions.
    """
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation
```

### Verification
```python
# For identity quaternion [1, 0, 0, 0] (upright robot):
quat = [1, 0, 0, 0]

# Method 1 result:
gravity_rm = get_gravity_rotation_matrix(quat)  # [0, 0, -1]

# Method 2 result:
gravity_ut = get_gravity_orientation(quat)       # [0, 0, -1]

# Both give same result for upright robot ✓
```

## 6.4 Joint Position Observation

```python
def get_joint_position_observation(qpos, default_pos, mujoco_to_isaac):
    """
    Get joint position observation as offset from default.

    Args:
        qpos: Current joint positions [7:19] from MuJoCo (in MuJoCo order)
        default_pos: Default joint positions (in MuJoCo order)
        mujoco_to_isaac: Index mapping for reordering

    Returns:
        Joint position offsets in Isaac Lab order
    """
    # Compute offset from default
    joint_pos_offset_mujoco = qpos - default_pos

    # Reorder to Isaac Lab order
    joint_pos_offset_isaac = joint_pos_offset_mujoco[ISAAC_TO_MUJOCO]

    return joint_pos_offset_isaac
```

## 6.5 Complete Observation Function

```python
def _get_observation(self) -> np.ndarray:
    """Get 48-dim observation matching Isaac Lab format."""

    # 1. Get quaternion
    quat_wxyz = self.data.qpos[3:7]

    # 2. Gravity orientation
    proj_gravity = get_gravity_orientation(quat_wxyz)

    # 3. Velocity transformation
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat_wxyz)
    rot_mat = rot_mat.reshape(3, 3)

    lin_vel_w = self.data.qvel[0:3]  # World frame
    ang_vel_w = self.data.qvel[3:6]  # World frame

    lin_vel_b = rot_mat.T @ lin_vel_w  # Body frame
    ang_vel_b = rot_mat.T @ ang_vel_w  # Body frame

    # 4. Joint states
    joint_pos_mujoco = self.data.qpos[7:19] - self.DEFAULT_JOINT_POS_MUJOCO
    joint_vel_mujoco = self.data.qvel[6:18]

    # 5. Reorder to Isaac Lab order
    joint_pos_isaac = joint_pos_mujoco[self.ISAAC_TO_MUJOCO]
    joint_vel_isaac = joint_vel_mujoco[self.ISAAC_TO_MUJOCO]

    # 6. Assemble observation (NO SCALING)
    obs = np.concatenate([
        lin_vel_b,            # [0:3]   - 3D
        ang_vel_b,            # [3:6]   - 3D
        proj_gravity,         # [6:9]   - 3D
        self._commands,       # [9:12]  - 3D
        joint_pos_isaac,      # [12:24] - 12D
        joint_vel_isaac,      # [24:36] - 12D
        self._last_actions,   # [36:48] - 12D
    ])

    return obs.astype(np.float32)
```

---

# 7. PD Control Implementation

## 7.1 PD Control Theory

```
Position-Derivative (PD) Control:

    τ = Kp * (q_target - q_current) - Kd * q̇_current

    where:
    - τ: torque command
    - Kp: position gain (stiffness)
    - Kd: velocity gain (damping)
    - q_target: desired joint position
    - q_current: current joint position
    - q̇_current: current joint velocity
```

## 7.2 Isaac Lab Implementation

```python
# In Isaac Lab, PD is handled by DCMotorCfg
robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,      # Max torque
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=20.0,         # Kp
            damping=0.5,            # Kd
            friction=0.0,
        ),
    },
)

# When action is applied:
# processed_action = action_scale * raw_action + default_pos
# torque = Kp * (processed_action - current_pos) - Kd * current_vel
```

## 7.3 MuJoCo Implementation

```python
def step(self, action: np.ndarray):
    """Apply action and step simulation."""

    # 1. Clip actions
    action = np.clip(action, -2.0, 2.0)

    # 2. Compute target position (Isaac order)
    target_pos_isaac = self.DEFAULT_JOINT_POS_ISAAC + self.action_scale * action

    # 3. Convert to MuJoCo order
    target_pos_mujoco = target_pos_isaac[self.MUJOCO_TO_ISAAC]

    # 4. Clip to joint limits
    target_pos_mujoco = np.clip(
        target_pos_mujoco,
        self.joint_pos_lower,
        self.joint_pos_upper
    )

    # 5. Run physics with PD control
    for _ in range(self.decimation):  # 4 physics steps per control step
        current_pos = self.data.qpos[7:19]
        current_vel = self.data.qvel[6:18]

        # PD control
        tau = self.KP * (target_pos_mujoco - current_pos) - self.KD * current_vel

        # Apply torque (MuJoCo will add its built-in damping)
        self.data.ctrl[:] = tau

        # Step physics
        mujoco.mj_step(self.model, self.data)
```

## 7.4 Gain Tuning Experiments

```python
# Experiment results table
"""
| Kp  | Kd  | Built-in | Effective Kd | Result                    |
|-----|-----|----------|--------------|---------------------------|
| 20  | 0.5 | 2.0      | 2.5          | Robot collapses           |
| 35  | 0.8 | 2.0      | 2.8          | Holds position, no walk   |
| 50  | 1.0 | 2.0      | 3.0          | Better tracking, unstable |
| 50  | 2.0 | 2.0      | 4.0          | Over-damped, survives     |
"""

# Final chosen values:
KP = 35.0  # Good tracking without oscillation
KD = 0.8   # Additional damping on top of MuJoCo's 2.0
```

## 7.5 Why MuJoCo Needs Higher Gains

```python
"""
Analysis of gain requirements:

Isaac Lab equation:
    τ_isaac = Kp * error - Kd * vel
    τ_isaac = 20 * error - 0.5 * vel

MuJoCo equation (with built-in damping):
    τ_applied = Kp * error - Kd * vel
    τ_effective = τ_applied - damping * vel
    τ_effective = Kp * error - (Kd + 2) * vel

To match Isaac Lab behavior in MuJoCo:
    Kp_mujoco = Kp_isaac = 20 (same)
    Kd_mujoco = Kd_isaac - 2 = 0.5 - 2 = -1.5 (negative! unstable)

Since we can't use negative Kd, we need higher Kp to compensate:
    Kp_mujoco > Kp_isaac to overcome the extra damping

Empirically, Kp=35 with Kd=0.8 works reasonably well.
"""
```

---

# 8. Training Configuration

## 8.1 Environment Configuration

### Before (Custom config)
```python
@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    # Task rewards
    track_lin_vel_xy_exp_weight = 1.0
    track_ang_vel_z_exp_weight = 0.5

    # Gait reward - TOO HIGH
    feet_air_time_weight = 0.25  # Should be 0.01

    # Penalties - TOO LOW
    dof_torques_l2_weight = -1e-5  # Should be -0.0002
    flat_orientation_l2_weight = -1.0  # Should be -2.5

    # Termination
    termination_penalty = -200.0  # Should be ~-50
```

### After (Matching official Isaac Lab)
```python
@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    # Task rewards (official values)
    track_lin_vel_xy_exp_weight = 1.5
    track_ang_vel_z_exp_weight = 0.75

    # Gait reward - REDUCED to official
    feet_air_time_weight = 0.01

    # Penalties - INCREASED to official
    dof_torques_l2_weight = -0.0002  # 20x stronger
    flat_orientation_l2_weight = -2.5  # 2.5x stronger

    # Termination
    termination_penalty = -50.0
```

## 8.2 PPO Configuration

### Before
```python
@configclass
class Go2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    max_iterations = 2000

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],  # Uniform
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=3e-4,  # Too low
        # ...
    )
```

### After
```python
@configclass
class Go2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    max_iterations = 1500  # Official value

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],  # Tapered
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        learning_rate=1e-3,  # Official value
        # ...
    )
```

## 8.3 Domain Randomization

```python
@configclass
class EventCfg:
    """Domain randomization events."""

    # Friction randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Base mass randomization (simulates payloads)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),  # -1kg to +2kg
            "operation": "add",
        },
    )

    # Actuator gains randomization
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),  # ±20%
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # External push perturbations
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )

    # Joint position randomization at reset
    randomize_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.25, 0.25),  # ±14°
            "velocity_range": (0.0, 0.0),
        },
    )
```

## 8.4 Training Results

```
Iteration 0:
    Mean reward: -0.65
    Episode length: 13 steps
    Robot falls immediately

Iteration 100:
    Mean reward: 4.25
    Episode length: 999 steps
    Robot survives full episode

Iteration 500:
    Mean reward: 7.33
    Episode length: 999 steps
    Good velocity tracking

Iteration 1000:
    Mean reward: 7.85
    Episode length: 999 steps
    Stable performance
```

---

# 9. Debugging Tools and Scripts

## 9.1 Policy Action Debugger

```python
#!/usr/bin/env python3
"""debug_policy_actions.py - Analyze policy outputs step by step."""

import torch
import numpy as np
from go2.mujoco_env import Go2MuJoCoEnv

def load_policy(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    class MLPActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = torch.nn.Sequential(
                torch.nn.Linear(48, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 64),
                torch.nn.ELU(),
                torch.nn.Linear(64, 12),
            )
        def forward(self, x):
            return self.actor(x)

    actor = MLPActor()
    actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state)
    actor.eval()
    return actor

def debug_policy():
    actor = load_policy("model_1000.pt")
    env = Go2MuJoCoEnv()

    JOINT_NAMES = ["FL_hip", "FR_hip", "RL_hip", "RR_hip",
                   "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                   "FL_calf", "FR_calf", "RL_calf", "RR_calf"]

    obs = env.reset(commands=[0.5, 0.0, 0.0])

    for step in range(10):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            action = actor(obs_tensor).numpy().squeeze()

        print(f"\n=== Step {step} ===")
        print(f"Height: {env.data.qpos[2]:.4f}")

        # Print action details
        print("Actions (Isaac order):")
        for i, name in enumerate(JOINT_NAMES):
            default = env.DEFAULT_JOINT_POS_ISAAC[i]
            target = default + env.action_scale * action[i]
            print(f"  {name:10s}: action={action[i]:+.3f}, target={target:+.3f}")

        obs, reward, done, info = env.step(action)

        if done:
            print("\n[TERMINATED]")
            break

if __name__ == "__main__":
    debug_policy()
```

## 9.2 Observation Comparison Script

```python
#!/usr/bin/env python3
"""compare_observations.py - Compare observations between simulators."""

import numpy as np

def analyze_observation(obs, label):
    """Break down observation vector."""
    print(f"\n=== {label} ===")
    print(f"Lin vel (body):     [{obs[0]:+.4f}, {obs[1]:+.4f}, {obs[2]:+.4f}]")
    print(f"Ang vel (body):     [{obs[3]:+.4f}, {obs[4]:+.4f}, {obs[5]:+.4f}]")
    print(f"Gravity:            [{obs[6]:+.4f}, {obs[7]:+.4f}, {obs[8]:+.4f}]")
    print(f"Commands:           [{obs[9]:+.4f}, {obs[10]:+.4f}, {obs[11]:+.4f}]")
    print(f"Joint pos [0:4]:    {obs[12:16]}")
    print(f"Joint vel [0:4]:    {obs[24:28]}")
    print(f"Last actions [0:4]: {obs[36:40]}")

# Example usage:
# obs_isaac = ... (from Isaac Lab)
# obs_mujoco = ... (from MuJoCo)
# analyze_observation(obs_isaac, "Isaac Lab")
# analyze_observation(obs_mujoco, "MuJoCo")
#
# diff = np.abs(obs_isaac - obs_mujoco)
# print(f"\nMax difference: {diff.max():.6f} at index {diff.argmax()}")
```

## 9.3 Joint Limit Checker

```python
#!/usr/bin/env python3
"""check_joint_limits.py - Verify joint limits match between simulators."""

import mujoco
import numpy as np

def check_mujoco_limits(model_path):
    """Extract and display joint limits from MuJoCo model."""
    model = mujoco.MjModel.from_xml_path(model_path)

    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]

    print("MuJoCo Joint Limits:")
    print("-" * 50)
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lower = model.jnt_range[joint_id, 0]
        upper = model.jnt_range[joint_id, 1]
        print(f"  {name:20s}: [{lower:+.3f}, {upper:+.3f}] rad")
        print(f"  {' '*20}  [{np.degrees(lower):+.1f}°, {np.degrees(upper):+.1f}°]")

# Example output:
# MuJoCo Joint Limits:
# --------------------------------------------------
#   FL_hip_joint        : [-1.047, +1.047] rad
#                         [-60.0°, +60.0°]
#   FL_thigh_joint      : [-1.464, +3.598] rad
#                         [-83.9°, +206.1°]
#   ...
```

---

# 10. Complete Code Reference

## 10.1 Go2 MuJoCo Environment (Complete)

```python
# File: source/go2.mujoco/go2.mujoco/go2.mujoco_env.py

"""Go2 MuJoCo environment for sim-to-sim transfer testing."""

from __future__ import annotations
import os
from typing import Tuple
import mujoco
import numpy as np


def get_gravity_orientation(quaternion):
    """Compute gravity vector in body frame from quaternion (wxyz format)."""
    qw, qx, qy, qz = quaternion
    gravity = np.zeros(3)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity


class Go2MuJoCoEnv:
    """MuJoCo environment for Go2 quadruped."""

    # Joint ordering mappings
    ISAAC_TO_MUJOCO = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    MUJOCO_TO_ISAAC = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])

    # Default poses
    DEFAULT_JOINT_POS_ISAAC = np.array([
        0.1, -0.1, 0.1, -0.1,    # hips
        0.8, 0.8, 1.0, 1.0,      # thighs
        -1.5, -1.5, -1.5, -1.5,  # calfs
    ], dtype=np.float32)

    DEFAULT_JOINT_POS_MUJOCO = np.array([
        0.1, 0.8, -1.5,    # FL
        -0.1, 0.8, -1.5,   # FR
        0.1, 1.0, -1.5,    # RL
        -0.1, 1.0, -1.5,   # RR
    ], dtype=np.float32)

    # PD gains
    KP = 35.0
    KD = 0.8

    def __init__(self, model_path=None, action_scale=0.25, dt=0.005,
                 decimation=4, render=False):
        # ... initialization code ...
        pass

    def reset(self, commands=None):
        """Reset to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0, 0, 0.40]
        self.data.qpos[3:7] = [1, 0, 0, 0]
        self.data.qpos[7:19] = self.DEFAULT_JOINT_POS_MUJOCO.copy()
        self.data.qvel[:] = 0
        # ... rest of reset ...
        return self._get_observation()

    def step(self, action):
        """Apply action and step."""
        # Clip actions
        action = np.clip(action, -2.0, 2.0)

        # Compute targets
        target_isaac = self.DEFAULT_JOINT_POS_ISAAC + self.action_scale * action
        target_mujoco = target_isaac[self.MUJOCO_TO_ISAAC]
        target_mujoco = np.clip(target_mujoco, self.joint_pos_lower,
                                self.joint_pos_upper)

        # PD control loop
        for _ in range(self.decimation):
            pos = self.data.qpos[7:19]
            vel = self.data.qvel[6:18]
            tau = self.KP * (target_mujoco - pos) - self.KD * vel
            self.data.ctrl[:] = tau
            mujoco.mj_step(self.model, self.data)

        # Update state
        self._last_actions = action.astype(np.float32).copy()

        return self._get_observation(), self._compute_reward(), \
               self._check_termination(), {"base_height": self.data.qpos[2]}

    def _get_observation(self):
        """Build 48-dim observation."""
        quat = self.data.qpos[3:7]
        gravity = get_gravity_orientation(quat)

        rot_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rot_mat, quat)
        rot_mat = rot_mat.reshape(3, 3)

        lin_vel_b = rot_mat.T @ self.data.qvel[0:3]
        ang_vel_b = rot_mat.T @ self.data.qvel[3:6]

        joint_pos = self.data.qpos[7:19] - self.DEFAULT_JOINT_POS_MUJOCO
        joint_vel = self.data.qvel[6:18]

        joint_pos_isaac = joint_pos[self.ISAAC_TO_MUJOCO]
        joint_vel_isaac = joint_vel[self.ISAAC_TO_MUJOCO]

        return np.concatenate([
            lin_vel_b, ang_vel_b, gravity, self._commands,
            joint_pos_isaac, joint_vel_isaac, self._last_actions
        ]).astype(np.float32)
```

## 10.2 Isaac Lab Environment Configuration (Complete)

```python
# File: source/go2.envs/go2.envs/direct/go2/go2_env_cfg.py

"""Configuration for Go2 Direct RL environment."""

from __future__ import annotations
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
import isaaclab.envs.mdp as mdp


@configclass
class EventCfg:
    """Domain randomization events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    randomize_joint_positions = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    """Go2 environment configuration."""

    # Environment
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 48
    state_space = 0

    # Domain randomization
    events: EventCfg = EventCfg()

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=20.0,
                damping=0.5,
                friction=0.0,
            ),
        },
    )

    # Contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )

    # Commands
    lin_vel_x_range = (-1.0, 1.0)
    lin_vel_y_range = (-1.0, 1.0)
    ang_vel_z_range = (-1.0, 1.0)
    command_resampling_time = 10.0

    # Reward weights (official Isaac Lab values)
    track_lin_vel_xy_exp_weight = 1.5
    track_ang_vel_z_exp_weight = 0.75
    feet_air_time_weight = 0.01
    lin_vel_z_l2_weight = -2.0
    ang_vel_xy_l2_weight = -0.05
    dof_torques_l2_weight = -0.0002
    dof_acc_l2_weight = -2.5e-7
    action_rate_l2_weight = -0.01
    flat_orientation_l2_weight = -2.5
    termination_penalty = -50.0

    # Observation noise
    add_noise = True
    noise_scales = {
        "lin_vel": 0.1,
        "ang_vel": 0.2,
        "projected_gravity": 0.05,
        "joint_pos": 0.01,
        "joint_vel": 1.5,
    }

    # Reset
    reset_position_noise = 0.5
    reset_yaw_noise = 3.14159
```

---

# 11. Successful Transfer Results

## 11.1 Final Working Configuration

After extended training (6000 iterations), the policy successfully transfers from Isaac Lab to MuJoCo.

### Test Results
```
Commands: vx=0.50, vy=0.00, wz=0.00

Step 0: height=0.399, vel=[0.00, -0.00, -0.09], time=0.0s
Step 100: height=0.357, vel=[0.43, -0.04, -0.01], time=2.0s
Step 200: height=0.355, vel=[0.47, 0.10, -0.03], time=4.0s
Step 300: height=0.353, vel=[0.41, -0.03, -0.01], time=6.0s
Step 400: height=0.352, vel=[0.38, 0.05, 0.10], time=8.0s

============================================================
TRANSFER TEST SUMMARY
============================================================
Episodes: 5
Max steps: 500
Mean episode length: 500.0 ± 0.0
Mean reward: 707.21 ± 0.00
Mean height: 0.355

Success rate (>= 400 steps): 100.0%

✓ TRANSFER SUCCESSFUL!
```

### Key Metrics
| Metric | Value |
|--------|-------|
| Success rate | **100%** |
| Mean episode length | 500 steps (max) |
| Mean reward | 707.21 |
| Mean height | 0.355m |
| Forward velocity | ~0.4-0.47 m/s (cmd: 0.5) |

## 11.2 Working Test Script

```python
# go2_experiment/test_go2_transfer.py
class Go2PolicyWrapper:
    """Wrapper to load and run Isaac Lab Go2 policy in MuJoCo."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Build actor network (same architecture as training)
        # [256, 256, 256] hidden dims with ELU activation
        self.actor = self._build_actor(obs_dim=48, action_dim=12, hidden_dims=[256, 256, 256])

        # Load weights
        actor_state = {}
        for key, value in state_dict.items():
            if key.startswith("actor."):
                actor_state[key.replace("actor.", "")] = value

        self.actor.load_state_dict(actor_state)
        self.actor.eval()

    def _build_actor(self, obs_dim: int, action_dim: int, hidden_dims: list) -> torch.nn.Module:
        """Build actor network matching RSL-RL architecture."""
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ELU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, action_dim))
        return torch.nn.Sequential(*layers)

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action = self.actor(obs_tensor)
        return action.squeeze(0).cpu().numpy()
```

### Usage
```bash
# Run benchmark test
python go2_experiment/test_go2_transfer.py \
    --model logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt

# Real-time visualization
python go2_experiment/test_go2_transfer.py \
    --model logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt \
    --visualize
```

## 11.3 Critical Success Factors

### 1. Extended Training
- **6000 iterations** vs early attempts with 500-1000
- Allowed policy to learn more robust locomotion gaits
- Domain randomization had time to improve generalization

### 2. Correct Joint Ordering
```python
# Isaac Lab order (by joint type)
ISAAC_JOINTS = [FL_hip, FR_hip, RL_hip, RR_hip,
                FL_thigh, FR_thigh, RL_thigh, RR_thigh,
                FL_calf, FR_calf, RL_calf, RR_calf]

# MuJoCo order (by leg)
MUJOCO_JOINTS = [FL_hip, FL_thigh, FL_calf,
                 FR_hip, FR_thigh, FR_calf,
                 RL_hip, RL_thigh, RL_calf,
                 RR_hip, RR_thigh, RR_calf]

# Mappings
ISAAC_TO_MUJOCO = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]
MUJOCO_TO_ISAAC = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
```

### 3. Gravity Orientation Formula
```python
def get_gravity_orientation(quaternion):
    """Quaternion in wxyz format - matches Unitree deployment."""
    qw, qx, qy, qz = quaternion
    gravity = np.zeros(3)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity
```

### 4. PD Gain Tuning
```python
# MuJoCo has built-in damping=2 on all joints
# Must compensate with higher Kp
KP = 35.0  # Stiffness
KD = 0.8   # Damping (adds to MuJoCo's built-in 2.0)

# Effective damping in MuJoCo = KD + built_in_damping = 0.8 + 2.0 = 2.8
```

### 5. Action Clipping
```python
# Prevent extreme actions that cause instability
action = np.clip(action, -2.0, 2.0)
```

### 6. Network Architecture Match
```python
# Must match training architecture exactly
hidden_dims = [256, 256, 256]  # 3 layers
activation = torch.nn.ELU()
```

---

# Summary

This document covers the complete technical details of the sim-to-sim transfer project:

1. **Simulator differences** - PhysX vs MuJoCo physics, built-in damping
2. **Cassie complexity** - 4-bar linkage, passive joints
3. **Go2 simplicity** - 12 direct actuators, ideal for verification
4. **Joint ordering** - Detailed mapping with verification
5. **Observation format** - 48-dim structure with transformations
6. **PD control** - Implementation and gain tuning
7. **Training config** - Reward weights, network architecture
8. **Debugging tools** - Scripts for analysis
9. **Successful transfer** - 100% success rate with extended training

## Key Findings

**✅ Sim-to-sim transfer IS achievable** when:
- Joint ordering is correctly mapped in both directions
- Gravity orientation uses consistent formula
- PD gains account for target simulator's physics (MuJoCo damping)
- Training duration is sufficient (6000+ iterations with DR)
- Network architecture matches exactly between training and deployment

**Successful Model**: `logs/rsl_rl/go2_direct_dr/2026-02-05_09-14-52/model_6000.pt`
