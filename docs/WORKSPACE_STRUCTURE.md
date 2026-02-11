# Sim2Sim Project Workspace Structure

## Overview

This project implements sim-to-sim transfer research for quadruped (Go2) and bipedal (Cassie) robots, transferring policies from Isaac Lab to MuJoCo.

---

## Directory Structure

```
sim2sim_project/
├── source/                          # Isaac Lab environment definitions
│   ├── go2/                         # Go2 quadruped robot
│   │   └── go2/
│   │       ├── go2_sim2sim_cfg.py           # Baseline config
│   │       ├── go2_sim2sim_moderate_cfg.py  # Moderate DR
│   │       ├── go2_sim2sim_aggressive_cfg.py
│   │       ├── go2_sim2sim_terrain_cfg.py   # Rough terrain
│   │       ├── go2_sim2sim_rma_cfg.py       # RMA
│   │       ├── go2_sim2sim_actuator_net_cfg.py
│   │       ├── agents/rsl_rl_ppo_cfg.py     # PPO hyperparameters
│   │       └── mdp/observations.py          # Custom observation functions
│   │
│   └── cassie/                      # Cassie bipedal robot
│       └── cassie/
│           ├── cassie_env.py                # Base environment class
│           ├── cassie_passive_env.py        # Passive ankle variant
│           ├── cassie_passive_enhanced_env.py
│           ├── cassie_balance_standing_env.py
│           ├── cassie_sim2sim_cfg.py        # Baseline config
│           ├── cassie_sim2sim_passive_cfg.py # Passive ankle config
│           ├── cassie_sim2sim_passive_enhanced_cfg.py
│           ├── cassie_balance_standing_cfg.py
│           └── agents/rsl_rl_ppo_cfg.py
│
├── scripts/                         # Training and deployment scripts
│   ├── rsl_rl/                      # RSL-RL training framework
│   │   ├── train.py                 # Main training entry point
│   │   ├── play.py                  # Policy inference/visualization
│   │   ├── play_arrows.py           # Manual keyboard control
│   │   └── cli_args.py              # Command-line argument parsing
│   │
│   ├── go2/                         # Go2 deployment pipeline
│   │   ├── export_policy.py         # Export to TorchScript
│   │   ├── export_rma_policy.py     # Export RMA policy
│   │   ├── deploy_mujoco.py         # MuJoCo deployment
│   │   ├── deploy_rma_mujoco.py     # RMA deployment
│   │   ├── compare_policies.py      # Policy comparison
│   │   └── go2_*.yaml               # Deployment configs
│   │
│   └── cassie/                      # Cassie deployment pipeline
│       ├── export_policy.py         # Export active ankle policy
│       ├── export_policy_passive.py # Export passive ankle policy
│       ├── deploy_mujoco.py         # MuJoCo deployment
│       ├── verify_cassie_mujoco.py  # Deployment verification
│       └── cassie_*.yaml            # Deployment configs
│
├── actuator_net/                    # Actuator network training
│   ├── train_actuator_net.py        # Training script
│   ├── isaac_actuator.py            # Isaac Lab integration
│   ├── collect_mujoco_data.py       # Data collection from MuJoCo
│   └── configs/                     # Training configurations
│
├── rma/                             # Rapid Motor Adaptation
│   ├── train_rma.py                 # Phase 1: policy with privileged info
│   ├── train_adaptation.py          # Phase 2: adaptation module
│   ├── collect_rma_data.py          # Data collection in Isaac Lab
│   ├── collect_mujoco_rma_data.py   # Data collection in MuJoCo
│   ├── rma_policy.py                # RMA policy wrapper
│   ├── rma_env_wrapper.py           # Environment wrapper
│   ├── run_rma_pipeline.sh          # Complete pipeline script
│   └── configs/                     # Training configurations
│
├── experiments/go2/                 # Go2 evaluation framework
│   ├── eval_isaaclab.py             # Isaac Lab evaluation
│   ├── eval_mujoco.py               # MuJoCo evaluation
│   ├── eval_mujoco_rma_phase2.py    # RMA Phase 2 evaluation
│   ├── finetune_mujoco.py           # MuJoCo fine-tuning
│   ├── 1_domain_randomization/      # DR experiment results
│   ├── 2_actuator_net/              # ActuatorNet experiment results
│   ├── 3_rma/                       # RMA experiment results
│   └── 4_sim2sim_adaptation/        # Adaptation experiment results
│
├── external/                        # External robot models
│   └── agility_cassie/              # Cassie MuJoCo model (MJCF + meshes)
│       ├── cassie.xml               # Robot model
│       ├── scene.xml                # Scene wrapper
│       └── assets/                  # OBJ meshes
│
├── unitree_mujoco/                  # Go2 MuJoCo model (clone separately)
│   └── unitree_robots/go2/
│       ├── go2.xml                  # Robot model
│       ├── scene_flat.xml           # Flat terrain scene
│       └── scene_terrain.xml        # Rough terrain scene
│
├── cassie-mujoco-sim/               # Cassie MuJoCo simulator (clone separately)
│
├── docker/                          # Docker configuration
│   └── Dockerfile
│
├── docs/                            # Technical documentation
│
└── logs/                            # Training checkpoints (gitignored)
    ├── go2/rsl_rl/                  # Go2 training runs
    └── cassie/rsl_rl/               # Cassie training runs
```

---

## Robot Models

### Go2 (Quadruped)
- **Joints:** 12 (4 legs x 3 joints each)
- **Observation dim:** 48
- **Action dim:** 12
- **Action scale:** 0.25
- **Network:** [512, 256, 128], 5K iterations
- **MuJoCo model:** `unitree_mujoco/unitree_robots/go2/scene_flat.xml`

### Cassie (Biped)
- **Joints:** 10 actuated + 2 passive (ankle via 4-bar linkage)
- **Observation dim:** 46 (passive) or 48 (active)
- **Action dim:** 10 (passive) or 12 (active)
- **Action scale:** 0.5
- **Network:** [256, 256, 256], 30K iterations
- **MuJoCo model:** `external/agility_cassie/scene.xml`

---

## Training Commands

### Go2 Training

```bash
# Baseline with Domain Randomization
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-v0 --num_envs 4096

# Moderate DR (recommended for transfer)
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-Moderate-v0 --num_envs 4096

# Aggressive DR
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-Aggressive-v0 --num_envs 4096

# Rough terrain
python scripts/rsl_rl/train.py --task Isaac-Velocity-Rough-Go2-Sim2Sim-v0 --num_envs 4096

# RMA (with privileged critic)
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-RMA-v0 --num_envs 4096

# ActuatorNet
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-ActuatorNet-v0 --num_envs 4096
```

### Cassie Training

```bash
# Passive ankle baseline (matches MuJoCo, recommended)
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-v0 \
    --num_envs 4096 --max_iterations 30000

# Passive ankle with enhanced DR
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-Passive-Enhanced-v0 \
    --num_envs 4096 --max_iterations 30000

# Balance standing task
python scripts/rsl_rl/train.py --task Isaac-Balance-Standing-Cassie-v0 \
    --num_envs 4096 --max_iterations 20000

# Active ankle baseline (12 actions)
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Cassie-Sim2Sim-v0 \
    --num_envs 4096 --max_iterations 30000
```

---

## Export & Deploy Commands

### Go2

```bash
# Export policy to TorchScript
python scripts/go2/export_policy.py \
    --checkpoint logs/go2/rsl_rl/go2_sim2sim/<run>/model_4999.pt \
    --output logs/go2/policies/baseline.pt

# Deploy in MuJoCo
python scripts/go2/deploy_mujoco.py scripts/go2/go2_isaaclab.yaml

# Deploy with actuator network
python scripts/go2/deploy_mujoco.py scripts/go2/go2_isaaclab.yaml --actuator_net

# Deploy RMA policy
python scripts/go2/deploy_rma_mujoco.py scripts/go2/go2_rough.yaml
```

### Cassie

```bash
# Export passive ankle policy
python scripts/cassie/export_policy_passive.py \
    --checkpoint logs/cassie/rsl_rl/cassie_sim2sim/<run>/model_29999.pt \
    --output logs/cassie/policies/passive.pt

# Deploy in MuJoCo
python scripts/cassie/deploy_mujoco.py scripts/cassie/cassie_passive.yaml
```

---

## Logs Structure

```
logs/
├── go2/
│   ├── rsl_rl/
│   │   ├── go2_sim2sim/                    # Baseline training runs
│   │   ├── go2_sim2sim_moderate/           # Moderate DR runs
│   │   ├── go2_sim2sim_aggressive/         # Aggressive DR runs
│   │   ├── go2_sim2sim_terrain/            # Terrain runs
│   │   └── go2_rma/                        # RMA runs
│   └── policies/                           # Exported TorchScript policies
│
└── cassie/
    ├── rsl_rl/
    │   ├── cassie_sim2sim/                 # Active ankle runs
    │   ├── cassie_sim2sim_passive/         # Passive ankle runs
    │   └── cassie_balance_standing/        # Standing task runs
    └── policies/                           # Exported TorchScript policies
```

Each training run contains:
- `model_*.pt` - Checkpoints saved periodically
- `params/agent.yaml` - Training configuration
- `exported/policy.pt` - Auto-exported policy (from play.py)

---

## Key Files Reference

### Go2

| File | Purpose |
|------|---------|
| `source/go2/go2/go2_sim2sim_cfg.py` | Base environment configuration |
| `source/go2/go2/go2_sim2sim_moderate_cfg.py` | Moderate DR configuration |
| `source/go2/go2/agents/rsl_rl_ppo_cfg.py` | PPO training hyperparameters |
| `source/go2/go2/mdp/observations.py` | Custom observation functions |
| `scripts/go2/deploy_mujoco.py` | MuJoCo deployment script |
| `scripts/go2/go2_isaaclab.yaml` | Deployment configuration |

### Cassie

| File | Purpose |
|------|---------|
| `source/cassie/cassie/cassie_env.py` | Base environment class |
| `source/cassie/cassie/cassie_passive_env.py` | Passive ankle environment |
| `source/cassie/cassie/cassie_sim2sim_passive_cfg.py` | Passive ankle configuration |
| `source/cassie/cassie/cassie_balance_standing_cfg.py` | Standing task configuration |
| `scripts/cassie/deploy_mujoco.py` | MuJoCo deployment script |
| `scripts/cassie/cassie_passive.yaml` | Passive ankle deployment config |
