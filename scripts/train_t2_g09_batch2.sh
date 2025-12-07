#!/bin/bash
# Train T2 G09 models - Batch 2 (GPU 4)
# 3 small models with temperature=2.0, gamma=0.9 (串行训练)

set -e  # Exit on error

# Set CUDA device
export CUDA_VISIBLE_DEVICES=4

# Set PYTHONPATH to project root
PROJECT_ROOT="/home/zz359/workspace-whisper"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "============================================="
echo "Training T2 G09 Models - Batch 2 (GPU 4)"
echo "3 Small Models (Temperature=2.0, Gamma=0.9)"
echo "============================================="
echo ""

# Task 1: small_dkd_s2_t2_g09
echo "[1/3] Training small_dkd_s2_t2_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s2_t2_g09.yaml
echo "Completed at: $(date)"
echo ""

# Task 2: small_dkd_s3_t2_g09
echo "[2/3] Training small_dkd_s3_t2_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s3_t2_g09.yaml
echo "Completed at: $(date)"
echo ""

# Task 3: small_dkd_s4_t2_g09
echo "[3/3] Training small_dkd_s4_t2_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s4_t2_g09.yaml
echo "Completed at: $(date)"
echo ""

echo "============================================="
echo "All training tasks in Batch 2 completed!"
echo "Finished at: $(date)"
echo "============================================="

