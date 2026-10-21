#!/usr/bin/env bash
# Run the three large-teacher CE configs concurrently on caller-selected GPUs.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

GPU_LIST="${GPU_IDS:-${1:-}}"
if [[ -z "$GPU_LIST" ]]; then
    echo "Usage: GPU_IDS=0,1,2 bash scripts/train_large_ce_parallel.sh" >&2
    echo "Provide exactly three CUDA device indexes; no device is selected by default." >&2
    exit 2
fi

IFS=',' read -r -a DEVICES <<< "$GPU_LIST"
if [[ ${#DEVICES[@]} -ne 3 ]]; then
    echo "Expected exactly three comma-separated GPU indexes, got: $GPU_LIST" >&2
    exit 2
fi

CONFIGS=(
    "configs/large_ce_s2.yaml"
    "configs/large_ce_s3.yaml"
    "configs/large_ce_s4.yaml"
)
NAMES=("large_ce_s2" "large_ce_s3" "large_ce_s4")
PIDS=()

for index in "${!CONFIGS[@]}"; do
    device="cuda:${DEVICES[$index]}"
    echo "Starting ${NAMES[$index]} on $device"
    python -m src.training.train_ce \
        --config "${CONFIGS[$index]}" \
        --device "$device" \
        > "logs/${NAMES[$index]}.log" 2>&1 &
    PIDS+=("$!")
done

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "Completed ${NAMES[$index]}"
    else
        echo "Failed ${NAMES[$index]}; see logs/${NAMES[$index]}.log" >&2
        failed=1
    fi
done

exit "$failed"
