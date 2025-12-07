#!/bin/bash
# Train Large CE models on GPU 0/1/2 (parallel training)
# 3 large models with CE-only baseline

set -e  # Exit on error

# Set PYTHONPATH to project root
PROJECT_ROOT="/home/zz359/workspace-whisper"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "${PROJECT_ROOT}"

echo "============================================="
echo "Training Large CE Models (Parallel on GPU 0/1/2)"
echo "3 Large Models (CE-only baseline)"
echo "============================================="
echo ""

# Train all 3 models in parallel on different GPUs
echo "Starting all 3 training jobs in parallel..."
echo "Started at: $(date)"
echo ""

# GPU 0: large_ce_s2
CUDA_VISIBLE_DEVICES=0 python -m src.training.train_ce --config configs/large_ce_s2.yaml > logs/large_ce_s2.log 2>&1 &
PID1=$!
echo "[GPU 0] Training large_ce_s2 (PID: $PID1)"

# GPU 1: large_ce_s3
CUDA_VISIBLE_DEVICES=1 python -m src.training.train_ce --config configs/large_ce_s3.yaml > logs/large_ce_s3.log 2>&1 &
PID2=$!
echo "[GPU 1] Training large_ce_s3 (PID: $PID2)"

# GPU 2: large_ce_s4
CUDA_VISIBLE_DEVICES=2 python -m src.training.train_ce --config configs/large_ce_s4.yaml > logs/large_ce_s4.log 2>&1 &
PID3=$!
echo "[GPU 2] Training large_ce_s4 (PID: $PID3)"

echo ""
echo "All training jobs started! Waiting for completion..."
echo ""

# Wait for all background jobs to finish
wait $PID1
echo "[GPU 0] large_ce_s2 completed at $(date)"

wait $PID2
echo "[GPU 1] large_ce_s3 completed at $(date)"

wait $PID3
echo "[GPU 2] large_ce_s4 completed at $(date)"

echo ""
echo "============================================="
echo "All Large CE training jobs completed!"
echo "Finished at: $(date)"
echo "============================================="
echo ""
echo "Check logs in logs/ directory:"
echo "  - logs/large_ce_s2.log"
echo "  - logs/large_ce_s3.log"
echo "  - logs/large_ce_s4.log"





