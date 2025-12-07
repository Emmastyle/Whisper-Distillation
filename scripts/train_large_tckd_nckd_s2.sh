#!/bin/bash
# Train TCKD-only and NCKD-only (S2) on CUDA:1

set -e

export CUDA_VISIBLE_DEVICES=1

PROJECT_ROOT="/home/zz359/workspace-whisper"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "====================================="
echo "TCKD/NCKD S2 Training (CUDA:1)"
echo "====================================="
echo ""

echo "[1/2] Training large_tckd_s2_t1..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_tckd_s2_t1.yaml
echo "Completed at: $(date)"
echo ""

echo "[2/2] Training large_nckd_s2_t1..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_nckd_s2_t1.yaml
echo "Completed at: $(date)"
echo ""

echo "====================================="
echo "S2 training completed"
echo "Finished at: $(date)"
echo "====================================="


