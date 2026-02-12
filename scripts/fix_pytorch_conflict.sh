#!/bin/bash
# Fix PyTorch/torchvision mixed installation conflicts
# This resolves the "operator torchvision::nms does not exist" error

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PyTorch Conflict Fixer${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check current environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${GREEN}Active environment: $CONDA_DEFAULT_ENV${NC}\n"
else
    echo -e "${YELLOW}⚠ No conda environment detected${NC}"
    echo -e "${YELLOW}  Activate your Isaac Lab environment first:${NC}"
    echo -e "${YELLOW}  conda activate env_isaaclab${NC}\n"
    exit 1
fi

# Check current PyTorch installation
echo -e "${BLUE}Step 1: Checking current PyTorch installation...${NC}"
TORCH_PATH=$(python -c "import torch; print(torch.__file__)" 2>/dev/null | sed 's|/__init__.py||' || echo "NOT_FOUND")
TORCHVISION_PATH=$(python -c "import torchvision; print(torchvision.__file__)" 2>/dev/null | sed 's|/__init__.py||' || echo "NOT_FOUND")

if [ "$TORCH_PATH" != "NOT_FOUND" ]; then
    echo -e "  torch:       ${TORCH_PATH}"
fi
if [ "$TORCHVISION_PATH" != "NOT_FOUND" ]; then
    echo -e "  torchvision: ${TORCHVISION_PATH}"
fi
echo ""

# Uninstall all PyTorch packages
echo -e "${BLUE}Step 2: Removing all PyTorch installations...${NC}"
echo -e "${YELLOW}This will uninstall torch, torchvision, and torchaudio from all locations${NC}"
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted.${NC}"
    exit 1
fi

echo -e "${YELLOW}Uninstalling PyTorch packages...${NC}"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Also remove from user site-packages if they exist
if [ -d "$HOME/.local/lib/python3.10/site-packages" ]; then
    echo -e "${YELLOW}Cleaning user site-packages...${NC}"
    rm -rf "$HOME/.local/lib/python3.10/site-packages/torch"* 2>/dev/null || true
fi

# Reinstall PyTorch in conda environment
echo -e "\n${BLUE}Step 3: Reinstalling PyTorch in conda environment...${NC}"

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION_FULL=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    echo -e "NVIDIA driver detected: $CUDA_VERSION_FULL"

    # Recommend PyTorch version based on CUDA capability
    echo -e "\n${YELLOW}Select PyTorch version:${NC}"
    echo -e "  1) PyTorch 2.5.1 + CUDA 12.1 (recommended for CUDA 12.x)"
    echo -e "  2) PyTorch 2.5.1 + CUDA 11.8 (for older GPUs)"
    echo -e "  3) PyTorch 2.4.0 + CUDA 12.1"
    echo -e "  4) PyTorch 2.4.0 + CUDA 11.8"
    read -p "Choice (1-4): " choice

    case $choice in
        1)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
            TORCH_VERSION="2.5.1+cu121"
            ;;
        2)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
            TORCH_VERSION="2.5.1+cu118"
            ;;
        3)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
            TORCH_VERSION="2.4.0+cu121"
            ;;
        4)
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
            TORCH_VERSION="2.4.0+cu118"
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
else
    echo -e "${YELLOW}No NVIDIA GPU detected, installing CPU version${NC}"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    TORCH_VERSION="2.5.1+cpu"
fi

echo -e "\n${YELLOW}Installing PyTorch $TORCH_VERSION...${NC}"
pip install torch torchvision torchaudio --index-url $TORCH_INDEX_URL

# Verify installation
echo -e "\n${BLUE}Step 4: Verifying installation...${NC}"
TORCH_PATH_NEW=$(python -c "import torch; print(torch.__file__)" 2>/dev/null | sed 's|/__init__.py||')
TORCHVISION_PATH_NEW=$(python -c "import torchvision; print(torchvision.__file__)" 2>/dev/null | sed 's|/__init__.py||')
TORCH_VERSION_NEW=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
TORCHVISION_VERSION_NEW=$(python -c "import torchvision; print(torchvision.__version__)" 2>/dev/null)

TORCH_DIR=$(dirname $(dirname "$TORCH_PATH_NEW"))
TORCHVISION_DIR=$(dirname $(dirname "$TORCHVISION_PATH_NEW"))

if [ "$TORCH_DIR" == "$TORCHVISION_DIR" ]; then
    echo -e "${GREEN}✓ Success! Both packages in same location:${NC}"
    echo -e "  Location: $TORCH_DIR"
    echo -e "  torch:       $TORCH_VERSION_NEW"
    echo -e "  torchvision: $TORCHVISION_VERSION_NEW"

    # Test CUDA
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" == "True" ]; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        echo -e "${GREEN}✓ CUDA available ($GPU_COUNT GPU(s))${NC}\n"
    else
        echo -e "${YELLOW}⚠ CUDA not available (CPU mode)${NC}\n"
    fi

    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}PyTorch conflict resolved!${NC}"
    echo -e "${GREEN}========================================${NC}\n"
    echo -e "${BLUE}Run verify_setup.sh to confirm:${NC}"
    echo -e "  bash scripts/verify_setup.sh\n"
else
    echo -e "${RED}✗ Warning: Packages still in different locations${NC}"
    echo -e "  torch:       $TORCH_PATH_NEW"
    echo -e "  torchvision: $TORCHVISION_PATH_NEW\n"
fi
