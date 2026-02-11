# Copyright (c) 2024, Boonyaporn Preechasuth
# Go2 sim2sim config with ROUGH TERRAIN (baseline DR)
#
# Same observation space as flat (48-dim, no height scan) for MuJoCo
# deployment compatibility.  Rough terrain training improves robustness
# through proprioceptive adaptation alone ("blind" locomotion).

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .go2_sim2sim_cfg import (
    Go2Sim2SimEnvCfg,
    UNITREE_GO2_SIM2SIM_CFG,
)

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##############################################################################
# TERRAIN GENERATOR — scaled for Go2 (small robot)
##############################################################################

GO2_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Scaled down for Go2
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.025, 0.1),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


##############################################################################
# SCENE — rough terrain, no height scanner
##############################################################################

@configclass
class Go2TerrainSceneCfg(InteractiveSceneCfg):
    """Scene with rough terrain for Go2 sim2sim transfer."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=GO2_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = UNITREE_GO2_SIM2SIM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


##############################################################################
# CURRICULUM
##############################################################################

@configclass
class Go2TerrainCurriculumCfg:
    """Terrain difficulty curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##############################################################################
# ENVIRONMENT CONFIG
##############################################################################

@configclass
class Go2Sim2SimTerrainEnvCfg(Go2Sim2SimEnvCfg):
    """Go2 rough terrain env — same obs/rewards/DR as flat baseline.

    Observations: 48-dim (no height scan) for MuJoCo deployment compatibility.
    Terrain: rough with curriculum (stairs, slopes, random bumps).
    """

    scene: Go2TerrainSceneCfg = Go2TerrainSceneCfg(num_envs=4096, env_spacing=2.5)
    curriculum: Go2TerrainCurriculumCfg = Go2TerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Increase PhysX patch buffer for rough terrain collisions
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Enable curriculum for terrain generator
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True


@configclass
class Go2Sim2SimTerrainEnvCfg_PLAY(Go2Sim2SimTerrainEnvCfg):
    """Play configuration with fewer envs."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Smaller terrain grid for play
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
