# Go2 Experiments Reorganization Plan

## Current Structure (Messy)
- results/ - Mixed DR and RMA results
- results_dr/ - DR results with 3 comparison folders (confusing!)
- results_actuato_net/ - ActuatorNet results with 2 comparison folders

## New Structure (Clean)
experiments/go2/
├── 1_domain_randomization/
│   ├── eval_baseline/
│   ├── eval_moderate/
│   ├── eval_aggressive/
│   └── figures/                    # All DR comparison plots
├── 2_actuator_net/
│   ├── eval_baseline/
│   ├── eval_actuator_net/
│   └── figures/                    # ActuatorNet comparison plots
└── 3_rma/
    ├── eval_phase1_flat/
    ├── eval_phase1_rough/
    ├── eval_phase2_flat/
    └── figures/                    # RMA figures (if any)

## Migration Steps
1. Create new directory structure
2. Move evaluation data to clear locations
3. Consolidate comparison figures (use most complete versions)
4. Update exam2.tex paths
5. Remove old directories
