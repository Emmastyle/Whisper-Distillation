#!/bin/bash

# Script to train all Large KD models with alpha=0.1
# This covers all combinations of student sizes (S2, S3, S4) and temperatures (T1, T2)

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
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training Large KD Models (Alpha=0.1)${NC}"
echo -e "${BLUE}========================================${NC}"

# Array of config files
configs=(
    "configs/large_kd_s2_t1_a01.yaml"
    "configs/large_kd_s2_t2_a01.yaml"
    "configs/large_kd_s3_t1_a01.yaml"
    "configs/large_kd_s3_t2_a01.yaml"
    "configs/large_kd_s4_t1_a01.yaml"
    "configs/large_kd_s4_t2_a01.yaml"
)

# Train each model
for config in "${configs[@]}"; do
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Training: $config${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Extract experiment name from config path
    exp_name=$(basename "$config" .yaml)
    
    # Run training
    python src/training/train_kd.py --config "$config"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully completed: $exp_name${NC}"
    else
        echo -e "${RED}✗ Failed: $exp_name${NC}"
        exit 1
    fi
done

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}All Large KD (Alpha=0.1) trainings completed!${NC}"
echo -e "${BLUE}========================================${NC}"

# Summary
echo -e "\n${GREEN}Trained models:${NC}"
for config in "${configs[@]}"; do
    exp_name=$(basename "$config" .yaml)
    echo -e "  ✓ $exp_name"
done
