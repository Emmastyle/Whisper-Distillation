#!/bin/bash
# Run DKD a0.5/b2/g0.6 configs on CUDA:4 (S2/S3/S4 T1)

set -e

export CUDA_VISIBLE_DEVICES=4

PROJECT_ROOT="/home/zz359/workspace-whisper"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "====================================="
echo "DKD a0.5 b2.0 g0.6 (CUDA:4)"
echo "====================================="
echo ""

echo "[1/3] Training large_dkd_s2_t1_a05_b20_g06..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s2_t1_a05_b20_g06.yaml
echo "Completed at: $(date)"
echo ""

echo "[2/3] Training large_dkd_s3_t1_a05_b20_g06..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s3_t1_a05_b20_g06.yaml
echo "Completed at: $(date)"
echo ""

echo "[3/3] Training large_dkd_s4_t1_a05_b20_g06..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s4_t1_a05_b20_g06.yaml
echo "Completed at: $(date)"
echo ""

echo "====================================="
echo "All runs completed"
echo "Finished at: $(date)"
echo "====================================="


