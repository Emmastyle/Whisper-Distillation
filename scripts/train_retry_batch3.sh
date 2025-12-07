#!/bin/bash
# Retry failed training - Batch 3
# 2 large g09 + 2 small g09 (串行训练)

set -e  # Exit on error

# Set CUDA device
export CUDA_VISIBLE_DEVICES=3

# Set PYTHONPATH to project root
PROJECT_ROOT="/home/zz359/workspace-whisper"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "============================================="
echo "Retry Training - Batch 3 (GPU 3)"
echo "2 Large Gamma=0.9 + 2 Small Gamma=0.9"
echo "============================================="
echo ""

# Task 1: large_dkd_s4_t1_g09
echo "[1/4] Training large_dkd_s4_t1_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s4_t1_g09.yaml
echo "Completed at: $(date)"
echo ""

# Task 2: small_dkd_s2_t1_g09
echo "[2/4] Training small_dkd_s2_t1_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s2_t1_g09.yaml
echo "Completed at: $(date)"
echo ""

# Task 3: small_dkd_s3_t1_g09
echo "[3/4] Training small_dkd_s3_t1_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s3_t1_g09.yaml
echo "Completed at: $(date)"
echo ""

# Task 4: small_dkd_s4_t1_g09
echo "[4/4] Training small_dkd_s4_t1_g09..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s4_t1_g09.yaml
echo "Completed at: $(date)"
echo ""

echo "============================================="
echo "All training tasks in Batch 3 completed!"
echo "Finished at: $(date)"
echo "============================================="

