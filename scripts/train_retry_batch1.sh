#!/bin/bash
# Retry failed training - Batch 1
# 4 small models with alpha=2.0 (串行训练)

set -e  # Exit on error

# Set CUDA device

# Set PYTHONPATH to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "============================================="
echo "Retry Training - Batch 1 (selected device)"
echo "4 Small Models with Alpha=2.0"
echo "============================================="
echo ""

# Task 1: small_dkd_s2_t1_a20
echo "[1/4] Training small_dkd_s2_t1_a20..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s2_t1_a20.yaml
echo "Completed at: $(date)"
echo ""

# Task 2: small_dkd_s2_t2_a20
echo "[2/4] Training small_dkd_s2_t2_a20..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s2_t2_a20.yaml
echo "Completed at: $(date)"
echo ""

# Task 3: small_dkd_s3_t1_a20
echo "[3/4] Training small_dkd_s3_t1_a20..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s3_t1_a20.yaml
echo "Completed at: $(date)"
echo ""

# Task 4: small_dkd_s3_t2_a20
echo "[4/4] Training small_dkd_s3_t2_a20..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s3_t2_a20.yaml
echo "Completed at: $(date)"
echo ""

echo "============================================="
echo "All training tasks in Batch 1 completed!"
echo "Finished at: $(date)"
echo "============================================="

