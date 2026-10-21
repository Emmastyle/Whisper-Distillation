#!/bin/bash
# Train DKD S4 T2 paired experiments (large + small)
# Excluding g09 variants

set -e  # Exit on error

# Set CUDA device

# Set PYTHONPATH to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "====================================="
echo "DKD S4 T2 Paired Training"
echo "====================================="
echo ""

# Script 1: large_dkd_s4_t2_g06.yaml
echo "[1/2] Training large_dkd_s4_t2_g06..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/large_dkd_s4_t2_g06.yaml
echo "Completed at: $(date)"
echo ""

# Script 2: small_dkd_s4_t2_a20.yaml
echo "[2/2] Training small_dkd_s4_t2_a20..."
echo "Started at: $(date)"
python src/training/train_dkd.py --config configs/small_dkd_s4_t2_a20.yaml
echo "Completed at: $(date)"
echo ""

echo "====================================="
echo "All S4 T2 paired training completed!"
echo "Finished at: $(date)"
echo "====================================="

