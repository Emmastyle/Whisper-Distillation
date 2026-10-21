#!/bin/bash

# Script to train all Large models with Temperature=2.0
# This includes both KD (alpha=0.5) and DKD (gamma=0.6) models
# Covers all student sizes: S2, S3, S4

set -e  # Exit on error

# Set CUDA device
echo "使用 Device: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

# Resolve the repository root from this script, not the caller's directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training All Large T2 Models${NC}"
echo -e "${BLUE}========================================${NC}"

# Array of config files (6 configs total)
configs=(
    # KD models with T=2.0, alpha=0.5
    "configs/large_kd_s2_t2_a05.yaml"
    "configs/large_kd_s3_t2_a05.yaml"
    "configs/large_kd_s4_t2_a05.yaml"
    
    # DKD models with T=2.0, gamma=0.6
    "configs/large_dkd_s2_t2_g06.yaml"
    "configs/large_dkd_s3_t2_g06.yaml"
    "configs/large_dkd_s4_t2_g06.yaml"
)

# Determine which training script to use for each config
declare -A train_scripts
train_scripts["configs/large_kd_s2_t2_a05.yaml"]="src/training/train_kd.py"
train_scripts["configs/large_kd_s3_t2_a05.yaml"]="src/training/train_kd.py"
train_scripts["configs/large_kd_s4_t2_a05.yaml"]="src/training/train_kd.py"
train_scripts["configs/large_dkd_s2_t2_g06.yaml"]="src/training/train_dkd.py"
train_scripts["configs/large_dkd_s3_t2_g06.yaml"]="src/training/train_dkd.py"
train_scripts["configs/large_dkd_s4_t2_g06.yaml"]="src/training/train_dkd.py"

echo -e "${YELLOW}Total configs to train: ${#configs[@]}${NC}"
echo -e "${YELLOW}KD models: 3 (S2, S3, S4)${NC}"
echo -e "${YELLOW}DKD models: 3 (S2, S3, S4)${NC}\n"

# Train each model
count=0
for config in "${configs[@]}"; do
    count=$((count + 1))
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}[$count/${#configs[@]}] Training: $config${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Extract experiment name from config path
    exp_name=$(basename "$config" .yaml)
    
    # Get the appropriate training script
    train_script="${train_scripts[$config]}"
    
    echo -e "${BLUE}Using training script: $train_script${NC}"
    
    # Run training
    python "$train_script" --config "$config"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully completed: $exp_name${NC}"
    else
        echo -e "${RED}✗ Failed: $exp_name${NC}"
        exit 1
    fi
done

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}All Large T2 trainings completed!${NC}"
echo -e "${BLUE}========================================${NC}"

# Summary
echo -e "\n${GREEN}Trained models (Temperature=2.0):${NC}"
echo -e "\n${YELLOW}KD Models (alpha=0.5):${NC}"
echo -e "  ✓ large_kd_s2_t2_a05"
echo -e "  ✓ large_kd_s3_t2_a05"
echo -e "  ✓ large_kd_s4_t2_a05"

echo -e "\n${YELLOW}DKD Models (gamma=0.6):${NC}"
echo -e "  ✓ large_dkd_s2_t2_g06"
echo -e "  ✓ large_dkd_s3_t2_g06"
echo -e "  ✓ large_dkd_s4_t2_g06"
