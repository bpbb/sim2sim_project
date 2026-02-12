import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass
import numpy as np
import torch
import torch.nn as nn
import collections
import os
import argparse
import time
from scipy.spatial.transform import Rotation as R

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    'left_hip_joint', 'right_hip_joint',
    'left_shoulder_joint', 'right_shoulder_joint',
    'left_leg_joint', 'right_leg_joint',
    'left_wheel_joint', 'right_wheel_joint'
]

ACT = {name: i for i, name in enumerate(JOINT_NAMES)}

STACK_FRAMES = 3 
OBS_DIM_SINGLE = 28 
NUM_OBS = 88 
NUM_ACTIONS = 8
OBS_JOINT_VEL_SCALE = 0.15
OBS_ANG_VEL_SCALE = 0.25
CMD_SCALE = np.array([2.0, 0.0, 0.25, 0.0]) # [vx, vy, wz, pos_z]

PD_GAINS = {
    'hip':      {'kp': 100.0, 'kd': 1.5},
    'shoulder': {'kp': 100.0, 'kd': 1.5},
    'leg':      {'kp': 120.0, 'kd': 1.5},
    'wheel':    {'kp': 0.0,   'kd': 0.7}
}

JOINT_TORQUE_LIMIT = 60.0
WHEEL_TORQUE_LIMIT = 36.0

# ═══════════════════════════════════════════════════════════════════════════════
# POLICY
# ═══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    def __init__(self, num_obs=88, num_actions=8,
                 actor_hidden_dims=[512, 256, 128], activation="elu"):
        super().__init__()
        act_fn = {"elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        layers = []
        prev = num_obs
        for h in actor_hidden_dims:
            layers.extend([nn.Linear(prev, h), act_fn])
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.actor = nn.Sequential(*layers)

    def forward(self, x):
        return self.actor(x)

def load_policy(checkpoint_path, num_obs=88, num_actions=8):
    print(f"Loading policy from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    policy = ActorCritic(num_obs=num_obs, num_actions=num_actions)
    state = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("actor.")}
    policy.load_state_dict(state, strict=False)
    policy.eval()
    return policy

# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def get_joint_ids(model):
    jnt_qpos = {}
    jnt_qvel = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        jnt_qpos[name] = model.jnt_qposadr[jid]
        jnt_qvel[name] = model.jnt_dofadr[jid]
    return jnt_qpos, jnt_qvel

def quat_to_euler(quat_wxyz):
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return r.as_euler('xyz')

def compute_projected_gravity(quat_wxyz):
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    r = R.from_quat(quat_xyzw)
    gravity_world = np.array([0.0, 0.0, -1.0])
    return r.apply(gravity_world, inverse=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def construct_observation(data, last_action, negate_wheels=True, model=None):
    j_qpos, j_qvel = get_joint_ids(model)
    obs = np.zeros(OBS_DIM_SINGLE, dtype=np.float32)
    i = 0
    obs[i:i+4] = [data.qpos[j_qpos[n]] for n in JOINT_NAMES[:4]]
    i += 4
    obs[i:i+2] = [data.qpos[j_qpos[n]] * (-1.5) for n in JOINT_NAMES[4:6]]
    i += 2
    obs[i:i+4] = [data.qvel[j_qvel[n]] * 0.15 for n in JOINT_NAMES[:4]]
    i += 4
    obs[i:i+2] = [data.qvel[j_qvel[n]] * (-1.5) * 0.15 for n in JOINT_NAMES[4:6]]
    i += 2
    wheel_vels = np.array([data.qvel[j_qvel[n]] for n in JOINT_NAMES[6:8]])
    if negate_wheels: obs[i:i+2] = -wheel_vels * 0.15
    else: obs[i:i+2] = wheel_vels * 0.15
    i += 2
    r = R.from_quat([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])
    ang_vel_body = r.apply(data.qvel[3:6], inverse=True)
    obs[i:i+3] = ang_vel_body * 0.25
    i += 3
    obs[i:i+3] = compute_projected_gravity(data.qpos[3:7])
    i += 3
    obs[i:i+8] = last_action
    return obs

def apply_actions(model, data, actions, pd_scale=1.0, negate_wheels=True):
    j_qpos, j_qvel = get_joint_ids(model)
    torques = np.zeros(model.nu)
    for n in JOINT_NAMES[:4]:
        target = actions[ACT[n]]
        pos, vel = data.qpos[j_qpos[n]], data.qvel[j_qvel[n]]
        kp = PD_GAINS['hip' if 'hip' in n else 'shoulder']['kp'] * pd_scale
        kd = PD_GAINS['hip' if 'hip' in n else 'shoulder']['kd'] * pd_scale
        t = kp * (target - pos) - kd * vel
        torques[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)] = np.clip(t, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    for n in JOINT_NAMES[4:6]:
        target = actions[ACT[n]]
        pos, vel = data.qpos[j_qpos[n]], data.qvel[j_qvel[n]]
        kp, kd = PD_GAINS['leg']['kp'] * pd_scale, PD_GAINS['leg']['kd'] * pd_scale
        # gear 1.5 (unsigned): Match the torque magnification without flipping direction
        t_joint = (kp * (target - pos) - kd * vel) * 1.5
        torques[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)] = np.clip(t_joint, -JOINT_TORQUE_LIMIT, JOINT_TORQUE_LIMIT)
    for n in JOINT_NAMES[6:8]:
        target = actions[ACT[n]] * 20.0
        if negate_wheels: target = -target
        vel = data.qvel[j_qvel[n]]
        t = PD_GAINS['wheel']['kd'] * pd_scale * (target - vel)
        torques[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)] = np.clip(t, -WHEEL_TORQUE_LIMIT, WHEEL_TORQUE_LIMIT)
    data.ctrl[:] = torques

def check_termination(model, data):
    """Check for termination conditions: height threshold or illegal contact."""
    # 1. Height termination (base_link origin)
    if data.qpos[2] < 0.25:
        return True, "Base too low (Fell)"
    
    # 2. Pitch termination (already checked in main but good to have here)
    pitch = np.degrees(quat_to_euler(data.qpos[3:7])[1])
    if abs(pitch) > 70:
        return True, "Pitch too high (Flipped)"
        
    # 3. Illegal Contact (any body other than wheels touching the ground)
    # Floor geom is usually name "groundplane" or ID 0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "groundplane")
    
    # Links that should NOT touch the ground:
    illegal_body_names = [
        "base_link", 
        "left_hip_link", "right_hip_link",
        "left_shoulder_link", "right_shoulder_link",
        "left_leg_link", "right_leg_link"
    ]
    illegal_geom_ids = []
    for body in illegal_body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        # Find geoms belonging to this body
        for g_id in range(model.ngeom):
            if model.geom_bodyid[g_id] == body_id:
                illegal_geom_ids.append(g_id)

    for i in range(data.ncon):
        contact = data.contact[i]
        # Check if contact is between floor and an illegal geom
        hit_id = -1
        if contact.geom1 == floor_id and contact.geom2 in illegal_geom_ids: hit_id = contact.geom2
        if contact.geom2 == floor_id and contact.geom1 in illegal_geom_ids: hit_id = contact.geom1
        
        if hit_id != -1:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, hit_id)
            if name is None:
                # Fallback to body name
                b_id = model.geom_bodyid[hit_id]
                name = f"body_{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b_id)}"
            return True, f"Illegal contact: {name}"

    return False, ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--xml", default="assets/flamingo_torque.xml")
    parser.add_argument("--init_height", type=float, default=0.2461)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--pd_scale", type=float, default=1.0)
    parser.add_argument("--no_negate", action="store_true")
    parser.add_argument("--vis", action="store_true", help="Launch MuJoCo viewer")
    args = parser.parse_args()
    
    negate_wheels = not args.no_negate
    
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    policy = load_policy(args.policy)
    
    viewer = None
    if args.vis:
        viewer = mujoco.viewer.launch_passive(model, data)
        print("Viewer launched.")

    mujoco.mj_resetData(model, data)
    data.qpos[2] = args.init_height
    mujoco.mj_forward(model, data)
    last_action = np.zeros(8, dtype=np.float32)
    obs_history = collections.deque(maxlen=STACK_FRAMES)
    init_obs = construct_observation(data, last_action, negate_wheels, model)
    for _ in range(STACK_FRAMES): obs_history.append(init_obs)
    cmd_vx = 0.0
    commands = np.array([cmd_vx, 0.0, 0.0, 0.0]) * CMD_SCALE
    print(f"Starting test: h={args.init_height}, pd={args.pd_scale}")
    steps = int(args.duration / 0.02)
    for s in range(steps):
        if viewer is not None and not viewer.is_running():
            break
            
        # Stack newest first [obs_t, obs_t-1, obs_t-2]
        input_obs = np.concatenate([obs_history[STACK_FRAMES - 1 - i] for i in range(STACK_FRAMES)] + [commands])
        obs_tensor = torch.from_numpy(input_obs).float().unsqueeze(0)
        with torch.no_grad():
            raw_action = policy(obs_tensor).numpy().squeeze()
        last_action = raw_action.copy()
        action = np.clip(raw_action, -1.0, 1.0)
        for _ in range(4):
            apply_actions(model, data, action, args.pd_scale, negate_wheels)
            mujoco.mj_step(model, data)
            
        if viewer is not None:
            viewer.sync()
            time.sleep(0.02)
            
        # 3. Check for termination
        done, reason = check_termination(model, data)
            
        # 4. Obs
        new_obs = construct_observation(data, last_action, negate_wheels, model)
        obs_history.append(new_obs)
        
        if s % 25 == 0 or done:
            pitch = np.degrees(quat_to_euler(data.qpos[3:7])[1])
            vx = data.qvel[0]
            print(f"t={s*0.02:.2f}s | h={data.qpos[2]:.3f} | vx={vx:.2f} | pitch={pitch:.1f}")
            if done:
                print(f"TERMINATED: {reason}")
                break

if __name__ == "__main__":
    main()
