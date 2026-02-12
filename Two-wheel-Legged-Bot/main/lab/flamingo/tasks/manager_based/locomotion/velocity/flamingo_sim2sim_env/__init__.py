# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sim2Sim compatible Flamingo environment."""

import gymnasium as gym

from . import (
    agents,
    flat_env,
)

from .velocity_env_cfg import Sim2SimVelocityEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Sim2Sim-v1-ppo",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.FlamingoSim2SimFlatEnvCfg,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoSim2SimFlatPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Flamingo-Sim2Sim-v1-play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env.stand_drive.flat_env_stand_drive_cfg.FlamingoSim2SimFlatEnvCfg_PLAY,
        "co_rl_cfg_entry_point": agents.co_rl_cfg.FlamingoSim2SimFlatPPORunnerCfg,
    },
)
