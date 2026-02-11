"""Go2 Sim2Sim Transfer Extension for Isaac Lab."""

import gymnasium as gym

from . import agents

##
# Baseline sim2sim (minimal DR)
##

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_cfg:Go2Sim2SimEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_cfg:Go2Sim2SimEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimPPORunnerCfg",
    },
)

##
# Moderate DR (friction, mass, actuator gains, push)
##

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-Moderate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_moderate_cfg:Go2Sim2SimModerateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimModeratePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-Moderate-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_moderate_cfg:Go2Sim2SimModerateEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimModeratePPORunnerCfg",
    },
)

##
# Aggressive DR (widest ranges, strongest perturbations)
##

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-Aggressive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_aggressive_cfg:Go2Sim2SimAggressiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimAggressivePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-Aggressive-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_aggressive_cfg:Go2Sim2SimAggressiveEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimAggressivePPORunnerCfg",
    },
)

##
# Terrain baseline (rough terrain, no height scan, baseline DR)
##

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sim2Sim-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_terrain_cfg:Go2Sim2SimTerrainEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimTerrainPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sim2Sim-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_terrain_cfg:Go2Sim2SimTerrainEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimTerrainPPORunnerCfg",
    },
)

##
# Terrain + Moderate DR
##

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sim2Sim-Moderate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_terrain_moderate_cfg:Go2Sim2SimTerrainModerateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimTerrainModeratePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sim2Sim-Moderate-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_terrain_moderate_cfg:Go2Sim2SimTerrainModerateEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimTerrainModeratePPORunnerCfg",
    },
)

##
# Terrain + Aggressive DR
##

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sim2Sim-Aggressive-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_terrain_aggressive_cfg:Go2Sim2SimTerrainAggressiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimTerrainAggressivePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Go2-Sim2Sim-Aggressive-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_terrain_aggressive_cfg:Go2Sim2SimTerrainAggressiveEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimTerrainAggressivePPORunnerCfg",
    },
)

##
# RMA (Rapid Motor Adaptation) — Moderate DR + privileged critic observations
##

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_rma_cfg:Go2Sim2SimRMAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimRMAPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_rma_cfg:Go2Sim2SimRMAEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimRMAPPORunnerCfg",
    },
)

##
# ActuatorNet (learned actuator dynamics from MuJoCo)
##

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-ActuatorNet-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_actuator_net_cfg:Go2Sim2SimActuatorNetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimActuatorNetPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Go2-Sim2Sim-ActuatorNet-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_sim2sim_actuator_net_cfg:Go2Sim2SimActuatorNetEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2Sim2SimActuatorNetPPORunnerCfg",
    },
)
