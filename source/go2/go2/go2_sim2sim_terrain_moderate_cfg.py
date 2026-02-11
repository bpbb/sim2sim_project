# Copyright (c) 2024, Boonyaporn Preechasuth
# Go2 sim2sim config with ROUGH TERRAIN + MODERATE domain randomization

from isaaclab.utils import configclass

from .go2_sim2sim_moderate_cfg import Go2Sim2SimModerateEventCfg
from .go2_sim2sim_terrain_cfg import Go2Sim2SimTerrainEnvCfg


@configclass
class Go2Sim2SimTerrainModerateEnvCfg(Go2Sim2SimTerrainEnvCfg):
    """Rough terrain + moderate DR (friction, mass, gains, push)."""

    events: Go2Sim2SimModerateEventCfg = Go2Sim2SimModerateEventCfg()


@configclass
class Go2Sim2SimTerrainModerateEnvCfg_PLAY(Go2Sim2SimTerrainModerateEnvCfg):
    """Play configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
