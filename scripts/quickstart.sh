#!/bin/bash
# Sim2Sim Quick Start Script
# Verifies setup and optionally runs a command

set -e

BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Run verification first
echo -e "${BLUE}Running setup verification...${NC}\n"
bash "$SCRIPT_DIR/verify_setup.sh"

# Check exit code
if [ $? -ne 0 ]; then
    echo -e "\n${YELLOW}Setup verification failed. Fix the issues above before proceeding.${NC}"
    echo -e "${YELLOW}If you have PyTorch conflicts, run:${NC}"
    echo -e "  bash scripts/fix_pytorch_conflict.sh\n"
    exit 1
fi

# If verification passed and arguments provided, run the command
if [ $# -gt 0 ]; then
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}Running command:${NC} $@"
    echo -e "${BLUE}========================================${NC}\n"

    # Execute the provided command
    "$@"
else
    echo -e "\n${YELLOW}Usage:${NC}"
    echo -e "  ${BLUE}# Just verify setup:${NC}"
    echo -e "  bash scripts/quickstart.sh"
    echo -e ""
    echo -e "  ${BLUE}# Verify and run training:${NC}"
    echo -e "  bash scripts/quickstart.sh python scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-v0 --num_envs 4096"
    echo -e ""
    echo -e "  ${BLUE}# Verify and play:${NC}"
    echo -e "  bash scripts/quickstart.sh python scripts/rsl_rl/play.py --task Isaac-Velocity-Flat-Go2-Sim2Sim-Play-v0 --num_envs 1"
    echo -e ""
    echo -e "  ${BLUE}# Verify and deploy:${NC}"
    echo -e "  bash scripts/quickstart.sh python scripts/go2/deploy_mujoco.py scripts/go2/go2_isaaclab.yaml"
    echo -e ""
fi
