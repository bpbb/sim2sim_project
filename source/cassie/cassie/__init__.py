"""Cassie Sim2Sim Transfer Extension for Isaac Lab."""

import gymnasium as gym

from . import agents

##
# Sim2Sim Baseline (primary config for train → export → deploy pipeline)
##

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-v0",
    entry_point=f"{__name__}.cassie_env:CassieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_cfg:CassieSim2SimEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Play-v0",
    entry_point=f"{__name__}.cassie_env:CassieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_cfg:CassieSim2SimEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

##
# Sim2Sim Moderate DR (wider ranges for stronger transfer robustness)
##

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Moderate-v0",
    entry_point=f"{__name__}.cassie_env:CassieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_moderate_cfg:CassieSim2SimModerateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimModeratePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Moderate-Play-v0",
    entry_point=f"{__name__}.cassie_env:CassieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_moderate_cfg:CassieSim2SimModerateEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimModeratePPORunnerCfg",
    },
)

##
# Sim2Sim Aggressive DR (widest ranges, strongest perturbations)
##

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Aggressive-v0",
    entry_point=f"{__name__}.cassie_env:CassieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_aggressive_cfg:CassieSim2SimAggressiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimAggressivePPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Aggressive-Play-v0",
    entry_point=f"{__name__}.cassie_env:CassieEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_aggressive_cfg:CassieSim2SimAggressiveEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimAggressivePPORunnerCfg",
    },
)

##
# Passive Ankle (MuJoCo Match)
##

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-v0",
    entry_point=f"{__name__}.cassie_passive_env:CassiePassiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_passive_cfg:CassieSim2SimPassiveEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Play-v0",
    entry_point=f"{__name__}.cassie_passive_env:CassiePassiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_passive_cfg:CassieSim2SimPassiveEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

##
# Passive Ankle Enhanced (Aggressive DR for better transfer)
##

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Enhanced-v0",
    entry_point=f"{__name__}.cassie_passive_enhanced_env:CassiePassiveEnhancedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_passive_enhanced_cfg:CassieSim2SimPassiveEnhancedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Enhanced-Play-v0",
    entry_point=f"{__name__}.cassie_passive_enhanced_env:CassiePassiveEnhancedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_passive_enhanced_cfg:CassieSim2SimPassiveEnhancedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

##
# Passive Ankle Improved (Conservative DR with Curriculum)
##

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Improved-v0",
    entry_point=f"{__name__}.cassie_passive_env:CassiePassiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_passive_improved_cfg:CassiePassiveImprovedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Improved-Play-v0",
    entry_point=f"{__name__}.cassie_passive_env:CassiePassiveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_sim2sim_passive_improved_cfg:CassiePassiveImprovedEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

##
# Balance Standing Task (Simplified for Easy Transfer)
##

gym.register(
    id="Isaac-Balance-Standing-Cassie-v0",
    entry_point=f"{__name__}.cassie_balance_standing_env:CassieBalanceStandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_balance_standing_cfg:CassieBalanceStandingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Balance-Standing-Cassie-Play-v0",
    entry_point=f"{__name__}.cassie_balance_standing_env:CassieBalanceStandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_balance_standing_cfg:CassieBalanceStandingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

##
# Balance Standing Simple (Easier to Learn)
##

gym.register(
    id="Isaac-Balance-Standing-Cassie-Simple-v0",
    entry_point=f"{__name__}.cassie_balance_standing_env:CassieBalanceStandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_balance_standing_simple_cfg:CassieBalanceStandingSimpleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Balance-Standing-Cassie-Simple-Play-v0",
    entry_point=f"{__name__}.cassie_balance_standing_env:CassieBalanceStandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cassie_balance_standing_simple_cfg:CassieBalanceStandingSimpleEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieSim2SimPPORunnerCfg",
    },
)
