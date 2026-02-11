# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint with command visualization."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# Custom arguments for command visualization
parser.add_argument("--fixed_command", action="store_true", default=False, help="Use fixed forward command instead of random.")
parser.add_argument("--cmd_vx", type=float, default=1.0, help="Fixed command linear velocity x.")
parser.add_argument("--cmd_vy", type=float, default=0.0, help="Fixed command linear velocity y.")
parser.add_argument("--cmd_wz", type=float, default=0.0, help="Fixed command angular velocity z.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Add project source paths for go2 and cassie environments
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source", "go2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "source", "cassie"))

try:
    import go2  # noqa: F401 — registers Go2 gym environments
except ImportError:
    pass
try:
    import cassie  # noqa: F401 — registers Cassie gym environments
except ImportError:
    pass


def main():
    """Play with RSL-RL agent."""
    # Validate task argument
    if args_cli.task is None:
        print("ERROR: --task argument is required.")
        print("\nUsage:")
        print("  python scripts/rsl_rl/play_arrows.py --task Isaac-Velocity-Go2-Direct-v0 --checkpoint <path>")
        print("\nAvailable Go2 tasks:")
        print("  Isaac-Velocity-Go2-Direct-v0      (baseline)")
        print("  Isaac-Velocity-Go2-Direct-v6      (fixed gait + strong DR)")
        print("  Isaac-Velocity-Go2-Direct-RMA-v0  (RMA with privileged obs)")
        simulation_app.close()
        return

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    robot_name = agent_cfg.experiment_name.split("_")[0]
    log_root_path = os.path.join("logs", robot_name, "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.physics_dt
    
    # Get the actual environment (unwrap from all wrappers)
    base_env = env.unwrapped

    # Print command info
    if args_cli.fixed_command:
        print(f"[INFO] Using FIXED command: vx={args_cli.cmd_vx}, vy={args_cli.cmd_vy}, wz={args_cli.cmd_wz}")
    else:
        print("[INFO] Using RANDOM commands from environment")
    
    print("\n" + "="*70)
    print("COMMAND VISUALIZATION")
    print("="*70)
    print("Terminal shows: Cmd (commanded) vs Actual (robot velocity)")
    print("  - vx: forward/backward velocity (m/s)")
    print("  - vy: left/right velocity (m/s)")
    print("  - wz: turn rate (rad/s)")
    print("Color: GREEN=good tracking, YELLOW=ok, RED=poor tracking")
    print("="*70 + "\n")

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    
    # Track performance metrics
    vel_errors = []
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # Override commands if using fixed command
        if args_cli.fixed_command and hasattr(base_env, 'command_manager'):
            cmd_term = base_env.command_manager.get_term("base_velocity")
            cmd_term.vel_command_b[:, 0] = args_cli.cmd_vx
            cmd_term.vel_command_b[:, 1] = args_cli.cmd_vy
            cmd_term.vel_command_b[:, 2] = args_cli.cmd_wz
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        
        # Print command and actual velocity periodically
        if timestep % 30 == 0:  # Every 30 steps
            robot = base_env.scene["robot"]
            if hasattr(base_env, 'command_manager'):
                cmd = base_env.command_manager.get_command("base_velocity")[0].cpu().numpy()
                actual_vel = robot.data.root_lin_vel_b[0].cpu().numpy()
                actual_ang_vel = robot.data.root_ang_vel_b[0, 2].cpu().item()
                
                # Calculate tracking error
                vx_err = abs(cmd[0] - actual_vel[0])
                vy_err = abs(cmd[1] - actual_vel[1])
                wz_err = abs(cmd[2] - actual_ang_vel)
                
                # Color coding for terminal (ANSI codes)
                def color_val(err, val, threshold=0.3):
                    if err < threshold:
                        return f"\033[92m{val:+.2f}\033[0m"  # Green - good
                    elif err < threshold * 2:
                        return f"\033[93m{val:+.2f}\033[0m"  # Yellow - ok
                    else:
                        return f"\033[91m{val:+.2f}\033[0m"  # Red - bad
                
                print(f"[Step {timestep:4d}] "
                      f"Cmd: vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} wz={cmd[2]:+.2f} | "
                      f"Actual: vx={color_val(vx_err, actual_vel[0])} "
                      f"vy={color_val(vy_err, actual_vel[1])} "
                      f"wz={color_val(wz_err, actual_ang_vel)} | "
                      f"Err: {vx_err + vy_err:.2f}")
                
                vel_errors.append(vx_err + vy_err)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    
    # Print summary
    if len(vel_errors) > 0:
        print("\n" + "="*70)
        print("PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Average velocity tracking error: {np.mean(vel_errors):.3f} m/s")
        print(f"Min error: {np.min(vel_errors):.3f} m/s")
        print(f"Max error: {np.max(vel_errors):.3f} m/s")
        if np.mean(vel_errors) < 0.3:
            print("Status: \033[92mGOOD - Robot is tracking commands well!\033[0m")
        elif np.mean(vel_errors) < 0.6:
            print("Status: \033[93mOK - Robot is partially tracking commands\033[0m")
        else:
            print("Status: \033[91mPOOR - Robot is NOT tracking commands well\033[0m")
        print("="*70)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()