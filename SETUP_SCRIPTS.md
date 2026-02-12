# Setup and Verification Scripts

This document describes the setup verification and troubleshooting scripts available in this repository.

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| [`verify_setup.sh`](scripts/verify_setup.sh) | Comprehensive environment verification | Before training or deployment |
| [`fix_pytorch_conflict.sh`](scripts/fix_pytorch_conflict.sh) | Fix PyTorch/torchvision mixed installations | When you get "operator torchvision::nms does not exist" |
| [`quickstart.sh`](scripts/quickstart.sh) | Verify setup and run commands | Every time before running experiments |

## Usage

### 1. Verify Your Setup

Run this **before** any training or deployment:

```bash
bash scripts/verify_setup.sh
```

**Checks performed:**
- ✓ Python version (3.10+)
- ✓ Conda environment (`env_isaaclab`)
- ✓ PyTorch installation and version
- ✓ torchvision installation and version
- ✓ **Mixed installations** (PyTorch in different locations)
- ✓ CUDA availability
- ✓ Required packages (cassie, go2, lab.flamingo)
- ✓ Isaac Lab installation
- ✓ MuJoCo installation
- ✓ User site-packages conflicts

**Output:**
```
========================================
Sim2Sim Setup Verification
========================================

[1/10] Checking Python version...
✓ Python 3.10.19 (>= 3.10 required)

[2/10] Checking conda environment...
✓ Active environment: env_isaaclab

...

========================================
Verification Summary
========================================

✓ All checks passed! Your environment is ready.
```

### 2. Fix PyTorch Conflicts

If you encounter this error:
```
RuntimeError: operator torchvision::nms does not exist
```

Run the automatic fix script:

```bash
bash scripts/fix_pytorch_conflict.sh
```

**What it does:**
1. Detects mixed PyTorch installations (conda env vs user site-packages)
2. Uninstalls all PyTorch packages from all locations
3. Reinstalls PyTorch in the correct location (conda environment)
4. Verifies the fix

**Interactive prompts:**
- Confirms before uninstalling
- Lets you choose CUDA version (12.1, 11.8, or CPU)
- Verifies installation after fixing

### 3. Quick Start with Verification

Use this to verify setup **before** running any command:

```bash
# Just verify (no command):
bash scripts/quickstart.sh

# Verify and train:
bash scripts/quickstart.sh python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-v0 --num_envs 4096

# Verify and play:
bash scripts/quickstart.sh python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0 --num_envs 1

# Verify and deploy:
bash scripts/quickstart.sh python scripts/go2/deploy_mujoco.py scripts/go2/go2_isaaclab.yaml
```

**Benefits:**
- Catches configuration issues before they cause cryptic errors
- Saves time by detecting problems early
- Provides clear error messages and solutions

## Common Issues and Solutions

### Issue 1: "operator torchvision::nms does not exist"

**Cause:** PyTorch and torchvision installed in different locations

**Symptoms:**
```
RuntimeError: operator torchvision::nms does not exist
File: /home/user/.local/lib/python3.10/site-packages/torch/...  (user site)
File: /home/user/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/torchvision/...  (conda)
```

**Solution:**
```bash
bash scripts/fix_pytorch_conflict.sh
```

**Manual fix:**
```bash
conda activate env_isaaclab
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
bash scripts/verify_setup.sh
```

### Issue 2: Permission Errors During Installation

**Cause:** Trying to install to system directories or using `--user` flag in conda environment

**Symptoms:**
```
error: can't create or remove files in install directory
[Errno 13] Permission denied: '/usr/local/lib/python3.10/dist-packages/...'
```

**Solution:**
```bash
# 1. Activate conda environment
conda activate env_isaaclab

# 2. Install WITHOUT --user flag
pip install -e source/cassie
pip install -e source/go2 --no-deps
pip install -e Two-wheel-Legged-Bot/isaac_lab_envs --no-deps

# 3. Verify
bash scripts/verify_setup.sh
```

### Issue 3: Packages Already Installed (Different User)

**Cause:** Packages installed for different user account

**Check:**
```bash
pip list | grep -E "(cassie|go2|lab.flamingo)"
python -c "import cassie; print(cassie.__file__)"
```

**Solution:** Reinstall in your environment if paths don't match

### Issue 4: CUDA Not Available

**Symptoms:**
```
⚠ CUDA not available (CPU-only mode)
```

**Checks:**
1. NVIDIA driver installed: `nvidia-smi`
2. PyTorch built with CUDA: `python -c "import torch; print(torch.version.cuda)"`
3. GPU accessible: `python -c "import torch; print(torch.cuda.is_available())"`

**Solution:**
```bash
# Reinstall PyTorch with CUDA support
bash scripts/fix_pytorch_conflict.sh
# Select option 1 or 2 (CUDA 12.1 or 11.8)
```

## Best Practices

1. **Always activate conda environment first:**
   ```bash
   conda activate env_isaaclab
   ```

2. **Never use `--user` flag in conda environments:**
   ```bash
   # Wrong:
   pip install --user -e source/cassie

   # Correct:
   pip install -e source/cassie
   ```

3. **Run verification before long experiments:**
   ```bash
   bash scripts/verify_setup.sh
   # If all checks pass, then:
   python scripts/rsl_rl/train.py --task ... --max_iterations 30000
   ```

4. **Use quickstart for one-off commands:**
   ```bash
   bash scripts/quickstart.sh python scripts/rsl_rl/play.py --task ...
   ```

5. **Check after environment changes:**
   - After installing new packages
   - After updating PyTorch
   - After switching conda environments
   - After system updates

## Integration with Workflow

### Recommended Workflow

```bash
# 1. Activate environment
conda activate env_isaaclab

# 2. Verify setup
bash scripts/verify_setup.sh

# 3. If issues detected, fix them
bash scripts/fix_pytorch_conflict.sh  # If PyTorch conflict
# Or follow other suggestions from verify_setup.sh

# 4. Run your experiment
python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-v0 --num_envs 4096
```

### One-Line Verification + Run

```bash
bash scripts/quickstart.sh python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-v0 --num_envs 4096
```

## Script Locations

All scripts are in the [`scripts/`](scripts/) directory:

```
scripts/
├── verify_setup.sh          # Main verification script
├── fix_pytorch_conflict.sh  # PyTorch conflict resolver
├── quickstart.sh           # Verify + run wrapper
├── rsl_rl/
│   ├── train.py            # Training script
│   └── play.py             # Inference script
├── go2/
│   ├── deploy_mujoco.py    # Go2 deployment
│   └── ...
└── cassie/
    ├── deploy_mujoco.py    # Cassie deployment
    └── ...
```

## Exit Codes

All scripts return proper exit codes for scripting:

- `0`: Success, all checks passed
- `1`: Failure, one or more checks failed

**Example:**
```bash
if bash scripts/verify_setup.sh; then
    echo "Setup verified, running experiment..."
    python scripts/rsl_rl/train.py --task ...
else
    echo "Setup verification failed, aborting"
    exit 1
fi
```

## Automated Testing

You can use these scripts in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Verify setup
  run: bash scripts/verify_setup.sh

- name: Run tests
  run: python -m pytest tests/
```

## Contributing

When adding new dependencies:
1. Update `verify_setup.sh` to check for them
2. Document in this file
3. Test with `bash scripts/verify_setup.sh`
