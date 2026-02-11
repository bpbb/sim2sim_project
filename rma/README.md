# RMA (Rapid Motor Adaptation) for Sim-to-Sim Transfer

This module implements RMA for robust policy transfer between Isaac Lab (PhysX) and MuJoCo.

## Overview

RMA trains a policy that can adapt online to unknown environment parameters by learning to estimate them from observation history.

```
Training:   obs + privileged_info → encoder → latent → policy → action
Deployment: obs + obs_history → adaptation → latent → policy → action
```

## Folder Structure

```
rma/
├── configs/                    # Robot-specific configurations
│   ├── go2.yaml               # Go2 RMA settings
│   └── cassie.yaml            # Cassie RMA settings
├── models/                     # Trained models
│   ├── go2/
│   │   ├── phase1_policy.pt   # Base policy with encoder
│   │   └── phase2_adaptation.pt # Adaptation module
│   └── cassie/
├── checkpoints/                # Training checkpoints
│   ├── go2/
│   └── cassie/
├── rma_policy.py               # RMA policy architecture
├── rma_env_wrapper.py          # Privileged observation wrapper
├── train_rma.py                # Training script
└── run_rma_pipeline.sh         # Complete workflow
```

## Quick Start

```bash
# Run complete pipeline for Go2
./rma/run_rma_pipeline.sh go2

# Or for Cassie
./rma/run_rma_pipeline.sh cassie
```

## Two-Phase Training

### Phase 1: Train Base Policy with Privileged Encoder

```
┌─────────────────────────────────────────────────────────────┐
│                      PHASE 1: TRAINING                      │
│                                                             │
│   Observations ────┐                                        │
│                    ├──► Base Policy ──► Actions             │
│   Privileged  ─┐   │        ▲                               │
│   Info         │   │        │                               │
│   (friction,   └───┴► Encoder ──► Latent                    │
│    mass, etc.)                                              │
│                                                             │
│   Loss: PPO policy loss                                     │
└─────────────────────────────────────────────────────────────┘
```

The encoder learns to compress privileged information into a compact latent vector that helps the policy.

```bash
python scripts/rsl_rl/train.py --task Isaac-Velocity-Go2-Direct-RMA-v0 --num_envs 4096
```

### Phase 2: Train Adaptation Module

```
┌─────────────────────────────────────────────────────────────┐
│                      PHASE 2: TRAINING                      │
│                                                             │
│   Obs History ──► Adaptation ──► ê (estimated latent)       │
│   (last 50 steps)    Module            │                    │
│                                        │ MSE Loss           │
│   Privileged ──► Encoder ──► e (true)──┘                    │
│   Info            (frozen)                                  │
│                                                             │
│   Base policy is frozen. Only adaptation module trains.     │
└─────────────────────────────────────────────────────────────┘
```

The adaptation module learns to predict the encoder's output from observation history.

```bash
python rma/train_rma.py --robot go2 --phase 2 --checkpoint <phase1_checkpoint>
```

## Privileged Information

Information available during training but NOT during deployment:

| Parameter | Dimension | Range | Description |
|-----------|-----------|-------|-------------|
| Friction coefficients | 4 | [0.5, 1.25] | Per-foot friction |
| Mass offset | 1 | [-1, 2] kg | Added base mass |
| Actuator strength | 4 | [0.8, 1.2] | Per-leg scale |
| Ground friction | 1 | [0.5, 1.5] | Global friction |
| Payload | 1 | [0, 3] kg | External load |
| COM offset | 1 | [-0.05, 0.05] m | Center of mass shift |

**Total: 12 dimensions** (Go2)

## Architecture

```
Environment Encoder:
  Input:  privileged_info [12]
  Hidden: 64 → 32
  Output: latent [8]

Base Policy:
  Input:  obs [48] + latent [8] = [56]
  Hidden: 256 → 256 → 256
  Output: action [12]

Adaptation Module:
  Input:  obs_history [50 × 48] = [2400]
  Hidden: 256 → 128
  Output: estimated_latent [8]
```

## Comparison with Other Methods

| Method | Training | Deployment | Adaptation |
|--------|----------|------------|------------|
| **Standard PPO** | Fixed params | Fixed params | None |
| **Domain Randomization** | Random params | Fixed policy | None |
| **Actuator Net** | PhysX | MuJoCo-like | Offline |
| **RMA** | Random + privileged | Estimated from history | Online ✓ |

## Why RMA for Sim-to-Sim?

1. **Online Adaptation**: Policy adapts to MuJoCo's specific dynamics in real-time
2. **No Prior Knowledge**: Doesn't need to know MuJoCo parameters beforehand
3. **Robust**: Works even if MuJoCo's dynamics are outside training distribution
4. **Complementary**: Can combine with DR and Actuator Net

## Files

| File | Description |
|------|-------------|
| `rma_policy.py` | RMA policy architecture (encoder, policy, adaptation) |
| `rma_env_wrapper.py` | Privileged observation and history buffers |
| `train_rma.py` | Two-phase training script |
| `run_rma_pipeline.sh` | Complete workflow |
| `configs/*.yaml` | Robot-specific configurations |

## Reference

Based on: [Rapid Motor Adaptation for Legged Robots](https://arxiv.org/abs/2107.04034)
- Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik
- RSS 2021
