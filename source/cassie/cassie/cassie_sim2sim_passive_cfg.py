# cassie_sim2sim_passive_cfg.py

from __future__ import annotations

from .cassie_sim2sim_cfg import CassieSim2SimEnvCfg, CassieSim2SimEnvCfg_PLAY
from isaaclab.utils import configclass

from isaaclab.scene import InteractiveSceneCfg

@configclass
class CassieSim2SimPassiveEnvCfg(CassieSim2SimEnvCfg):
    """Cassie sim2sim with Passive Ankles (matching MuJoCo).
    
    Removes ankle actuators to enforce passive dynamics.
    Updates action and observation spaces accordingly.
    """

    # Reduced action space: 12 - 2 (ankles) = 10
    action_space = 10
    
    # Reduced observation space: 48 - 2 (ankle actions) = 46
    # Note: Joint positions and velocities of passive joints are still observed.
    observation_space = 46

    def __post_init__(self):
        super().__post_init__()

        # Remove ankle actuators to make them passive
        # Try common naming conventions
        actuator_keys_to_remove = ["ankle_joint", "ankle", "ankle_left", "ankle_right"]
        
        # Collect keys to delete (avoid runtime error changing dict size)
        keys_found = []
        for key in self.robot.actuators.keys():
            # Check for exact match or substring if regex keys are used (e.g. ".*ankle.*")
            # But usually dictionary keys are the config names.
            # CASSIE_CFG uses specific names.
            if any(x in key for x in actuator_keys_to_remove):
                keys_found.append(key)
        
        for key in keys_found:
            print(f"[CassieSim2SimPassiveEnvCfg] Removing actuator: {key}")
            del self.robot.actuators[key]
            
@configclass
class CassieSim2SimPassiveEnvCfg_PLAY(CassieSim2SimPassiveEnvCfg):
    """Play configuration for Passive Cassie."""
    
    # Copy scene settings from the PLAY config (redefined to avoid AttributeError)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=50,
        env_spacing=4.0,
        replicate_physics=True,
    )
    add_noise = False
