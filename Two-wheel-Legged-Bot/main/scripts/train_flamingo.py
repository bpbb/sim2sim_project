from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a policy for Flamingo.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys
import os

# Add the parent directory (Two-wheel-Legged-Bot/main) to sys.path
# This is needed because some modules try to import 'scripts.co_rl...'
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import gymnasium as gym
import lab.flamingo.tasks  # This triggers registration
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_tasks.utils.hydra import register_task_to_hydra
from isaaclab_tasks.utils import get_checkpoint_path

import rsl_rl
# We can't easily import the main from train.py if it's not a module.
# But we can replicate the main logic or just import it if we add the path.


# Path to the actual train.py
TRAIN_PY_PATH = "/home/drl-68/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
sys.path.append(os.path.dirname(TRAIN_PY_PATH))

import train

if __name__ == "__main__":
    train.main()
