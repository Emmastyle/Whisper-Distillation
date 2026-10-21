#!/bin/bash
# Train T2 G09 models - Batch 1 (selected device)
# 3 large models with temperature=2.0, gamma=0.9 (串行训练)

set -e  # Exit on error

# Set CUDA device

# Set PYTHONPATH to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "============================================="
echo "Training T2 G09 Models - Batch 1 (selected device)"
echo "3 Large Models (Temperature=2.0, Gamma=0.9)"
echo "============================================="
echo ""

# Task 1: large_dkd_s2_t2_g09
echo "[1/3] Training large_dkd_s2_t2_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s2_t2_g09.yaml
echo "Completed at: $(date)"
echo ""

# Task 2: large_dkd_s3_t2_g09
echo "[2/3] Training large_dkd_s3_t2_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s3_t2_g09.yaml
echo "Completed at: $(date)"
echo ""

# Task 3: large_dkd_s4_t2_g09
echo "[3/3] Training large_dkd_s4_t2_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s4_t2_g09.yaml
echo "Completed at: $(date)"
echo ""

echo "============================================="
echo "All training tasks in Batch 1 completed!"
echo "Finished at: $(date)"
echo "============================================="

