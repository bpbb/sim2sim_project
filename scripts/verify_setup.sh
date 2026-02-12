#!/bin/bash
# Sim2Sim Setup Verification Script
# Checks all dependencies and configurations before running Isaac Lab tasks

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track overall status
ALL_CHECKS_PASSED=true

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sim2Sim Setup Verification${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 1. Check Python version
echo -e "${BLUE}[1/10] Checking Python version...${NC}"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION (>= 3.10 required)${NC}\n"
else
    echo -e "${RED}✗ Python $PYTHON_VERSION found, but 3.10+ required${NC}\n"
    ALL_CHECKS_PASSED=false
fi

# 2. Check if in conda environment
echo -e "${BLUE}[2/10] Checking conda environment...${NC}"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    if [[ "$CONDA_DEFAULT_ENV" == *"isaaclab"* ]] || [[ "$CONDA_DEFAULT_ENV" == *"isaac"* ]]; then
        echo -e "${GREEN}✓ Active environment: $CONDA_DEFAULT_ENV${NC}\n"
    else
        echo -e "${YELLOW}⚠ Active environment: $CONDA_DEFAULT_ENV (expected 'env_isaaclab' or similar)${NC}\n"
    fi
else
    echo -e "${YELLOW}⚠ No conda environment detected (venv or system Python)${NC}\n"
fi

# 3. Check PyTorch installation
echo -e "${BLUE}[3/10] Checking PyTorch installation...${NC}"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
if [ "$TORCH_VERSION" != "NOT_INSTALLED" ]; then
    echo -e "${GREEN}✓ PyTorch $TORCH_VERSION installed${NC}"
else
    echo -e "${RED}✗ PyTorch not found${NC}"
    echo -e "${YELLOW}  Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121${NC}\n"
    ALL_CHECKS_PASSED=false
fi

# 4. Check torchvision installation
echo -e "${BLUE}[4/10] Checking torchvision installation...${NC}"
TORCHVISION_VERSION=$(python -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
if [ "$TORCHVISION_VERSION" != "NOT_INSTALLED" ]; then
    echo -e "${GREEN}✓ torchvision $TORCHVISION_VERSION installed${NC}\n"
else
    echo -e "${RED}✗ torchvision not found${NC}\n"
    ALL_CHECKS_PASSED=false
fi

# 5. Check for mixed PyTorch installations (CRITICAL)
echo -e "${BLUE}[5/10] Checking for mixed PyTorch installations...${NC}"
if [ "$TORCH_VERSION" != "NOT_INSTALLED" ] && [ "$TORCHVISION_VERSION" != "NOT_INSTALLED" ]; then
    TORCH_PATH=$(python -c "import torch; print(torch.__file__)" 2>/dev/null | sed 's|/__init__.py||')
    TORCHVISION_PATH=$(python -c "import torchvision; print(torchvision.__file__)" 2>/dev/null | sed 's|/__init__.py||')

    TORCH_DIR=$(dirname $(dirname "$TORCH_PATH"))
    TORCHVISION_DIR=$(dirname $(dirname "$TORCHVISION_PATH"))

    if [ "$TORCH_DIR" == "$TORCHVISION_DIR" ]; then
        echo -e "${GREEN}✓ PyTorch and torchvision in same location:${NC}"
        echo -e "  $TORCH_DIR${NC}\n"
    else
        echo -e "${RED}✗ MIXED INSTALLATION DETECTED!${NC}"
        echo -e "${RED}  torch:       $TORCH_PATH${NC}"
        echo -e "${RED}  torchvision: $TORCHVISION_PATH${NC}"
        echo -e "${YELLOW}  This causes 'operator torchvision::nms does not exist' error!${NC}"
        echo -e "${YELLOW}  Fix: pip uninstall torch torchvision torchaudio${NC}"
        echo -e "${YELLOW}       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121${NC}\n"
        ALL_CHECKS_PASSED=false
    fi
fi

# 6. Check CUDA availability
echo -e "${BLUE}[6/10] Checking CUDA availability...${NC}"
if [ "$TORCH_VERSION" != "NOT_INSTALLED" ]; then
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")
    if [ "$CUDA_AVAILABLE" == "True" ]; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null)
        echo -e "${GREEN}✓ CUDA $CUDA_VERSION available ($GPU_COUNT GPU(s) detected)${NC}\n"
    else
        echo -e "${YELLOW}⚠ CUDA not available (CPU-only mode)${NC}\n"
    fi
fi

# 7. Check required packages
echo -e "${BLUE}[7/10] Checking required packages...${NC}"
REQUIRED_PACKAGES=("cassie" "go2" "lab.flamingo")
PACKAGES_OK=true

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    PKG_VERSION=$(pip list 2>/dev/null | grep -E "^$pkg " | awk '{print $2}')
    if [ -n "$PKG_VERSION" ]; then
        echo -e "${GREEN}✓ $pkg $PKG_VERSION installed${NC}"
    else
        echo -e "${RED}✗ $pkg not installed${NC}"
        PACKAGES_OK=false
        ALL_CHECKS_PASSED=false
    fi
done

if [ "$PACKAGES_OK" = false ]; then
    echo -e "${YELLOW}  Run installation commands from README.md${NC}"
fi
echo ""

# 8. Check Isaac Lab installation
echo -e "${BLUE}[8/10] Checking Isaac Lab...${NC}"
ISAACLAB_PATH=$(python -c "import isaaclab; print(isaaclab.__path__[0])" 2>/dev/null || echo "NOT_FOUND")
if [ "$ISAACLAB_PATH" != "NOT_FOUND" ]; then
    echo -e "${GREEN}✓ Isaac Lab found at: $ISAACLAB_PATH${NC}\n"
else
    echo -e "${RED}✗ Isaac Lab not found${NC}"
    echo -e "${YELLOW}  Make sure Isaac Lab is installed and activated${NC}\n"
    ALL_CHECKS_PASSED=false
fi

# 9. Check MuJoCo installation
echo -e "${BLUE}[9/10] Checking MuJoCo...${NC}"
MUJOCO_VERSION=$(python -c "import mujoco; print(mujoco.__version__)" 2>/dev/null || echo "NOT_INSTALLED")
if [ "$MUJOCO_VERSION" != "NOT_INSTALLED" ]; then
    echo -e "${GREEN}✓ MuJoCo $MUJOCO_VERSION installed${NC}\n"
else
    echo -e "${YELLOW}⚠ MuJoCo not found (required for deployment)${NC}"
    echo -e "${YELLOW}  Install with: pip install mujoco${NC}\n"
fi

# 10. Check for common issues
echo -e "${BLUE}[10/10] Checking for common issues...${NC}"

# Check for user site-packages pollution
USER_SITE_ENABLED=$(python -c "import site; print(site.ENABLE_USER_SITE)" 2>/dev/null)
USER_TORCH=$(python -c "import sys; import os; user_site = os.path.expanduser('~/.local/lib/python3.10/site-packages'); print(os.path.exists(os.path.join(user_site, 'torch')))" 2>/dev/null || echo "false")

if [ "$USER_SITE_ENABLED" == "True" ] && [ "$USER_TORCH" == "True" ]; then
    echo -e "${YELLOW}⚠ PyTorch found in user site-packages (~/.local/)${NC}"
    echo -e "${YELLOW}  This may conflict with conda environment${NC}"
    echo -e "${YELLOW}  Consider: pip uninstall torch torchvision (from conda env)${NC}\n"
else
    echo -e "${GREEN}✓ No user site-packages conflicts detected${NC}\n"
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Verification Summary${NC}"
echo -e "${BLUE}========================================${NC}\n"

if [ "$ALL_CHECKS_PASSED" = true ]; then
    echo -e "${GREEN}✓ All checks passed! Your environment is ready.${NC}\n"
    echo -e "${BLUE}You can now run:${NC}"
    echo -e "  ${YELLOW}# Training${NC}"
    echo -e "  python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-v0 --num_envs 4096"
    echo -e ""
    echo -e "  ${YELLOW}# Playing/Testing${NC}"
    echo -e "  python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0 --num_envs 1"
    echo -e ""
    echo -e "  ${YELLOW}# MuJoCo Deployment${NC}"
    echo -e "  python scripts/go2/deploy_mujoco.py scripts/go2/go2_isaaclab.yaml"
    echo -e ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}\n"
    exit 1
fi
