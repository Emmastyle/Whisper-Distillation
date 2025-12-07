#!/bin/bash
# Train TCKD-only and NCKD-only (S3) on CUDA:2

set -e

export CUDA_VISIBLE_DEVICES=2

PROJECT_ROOT="/home/zz359/workspace-whisper"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "====================================="
echo "TCKD/NCKD S3 Training (CUDA:2)"
echo "====================================="
echo ""

echo "[1/2] Training large_tckd_s3_t1..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_tckd_s3_t1.yaml
echo "Completed at: $(date)"
echo ""

echo "[2/2] Training large_nckd_s3_t1..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_nckd_s3_t1.yaml
echo "Completed at: $(date)"
echo ""

echo "====================================="
echo "S3 training completed"
echo "Finished at: $(date)"
echo "====================================="


