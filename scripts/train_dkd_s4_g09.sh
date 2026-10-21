#!/bin/bash
# Serial training script for DKD S4 experiments with gamma=0.9
# Runs large and small teacher experiments sequentially on selected device

set -e  # Exit on error

# Set CUDA device
echo "使用 Device: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

# Set PYTHONPATH to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

cd "${PROJECT_ROOT}"

# Create log directory
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# Generate timestamp for this batch
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="${LOG_DIR}/batch_dkd_s4_g09_${BATCH_TIMESTAMP}.log"

echo "========================================" | tee -a "${BATCH_LOG}"
echo "Starting DKD S4 Gamma=0.9 Training" | tee -a "${BATCH_LOG}"
echo "Timestamp: ${BATCH_TIMESTAMP}" | tee -a "${BATCH_LOG}"
echo "Device: ${CUDA_VISIBLE_DEVICES:-auto}" | tee -a "${BATCH_LOG}"
echo "========================================" | tee -a "${BATCH_LOG}"
echo "" | tee -a "${BATCH_LOG}"

# Array of configurations
CONFIGS=(
    "configs/large_dkd_s4_t1_g09.yaml"
    "configs/small_dkd_s4_t1_g09.yaml"
)

# Training script (both use train_dkd.py)
TRAIN_SCRIPT="src/training/train_dkd.py"

# Function to extract config name from path
get_config_name() {
    basename "$1" .yaml
}

# Track overall progress
TOTAL=${#CONFIGS[@]}
CURRENT=0
SUCCESS=0
FAILED=0

# Train each configuration
for CONFIG in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    CONFIG_NAME=$(get_config_name "${CONFIG}")
    
    echo "========================================" | tee -a "${BATCH_LOG}"
    echo "[$CURRENT/$TOTAL] Training: ${CONFIG_NAME}" | tee -a "${BATCH_LOG}"
    echo "Config: ${CONFIG}" | tee -a "${BATCH_LOG}"
    echo "Script: ${TRAIN_SCRIPT}" | tee -a "${BATCH_LOG}"
    echo "Started at: $(date)" | tee -a "${BATCH_LOG}"
    echo "========================================" | tee -a "${BATCH_LOG}"
    echo "" | tee -a "${BATCH_LOG}"
    
    # Create individual log file
    RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_LOG="${LOG_DIR}/${CONFIG_NAME}_${RUN_TIMESTAMP}.log"
    
    # Run training
    if python "${TRAIN_SCRIPT}" --config "${CONFIG}" 2>&1 | tee "${RUN_LOG}"; then
        SUCCESS=$((SUCCESS + 1))
        echo "" | tee -a "${BATCH_LOG}"
        echo "✓ SUCCESS: ${CONFIG_NAME}" | tee -a "${BATCH_LOG}"
        echo "Completed at: $(date)" | tee -a "${BATCH_LOG}"
        echo "Log saved to: ${RUN_LOG}" | tee -a "${BATCH_LOG}"
    else
        FAILED=$((FAILED + 1))
        echo "" | tee -a "${BATCH_LOG}"
        echo "✗ FAILED: ${CONFIG_NAME}" | tee -a "${BATCH_LOG}"
        echo "Failed at: $(date)" | tee -a "${BATCH_LOG}"
        echo "Check log: ${RUN_LOG}" | tee -a "${BATCH_LOG}"
        
        # Don't exit on failure, continue with next config
        echo "⚠ Continuing with remaining experiments..." | tee -a "${BATCH_LOG}"
    fi
    
    echo "" | tee -a "${BATCH_LOG}"
    echo "Progress: ${SUCCESS} succeeded, ${FAILED} failed, $((TOTAL - CURRENT)) remaining" | tee -a "${BATCH_LOG}"
    echo "" | tee -a "${BATCH_LOG}"
done

# Final summary
echo "========================================" | tee -a "${BATCH_LOG}"
echo "Batch Training Complete!" | tee -a "${BATCH_LOG}"
echo "========================================" | tee -a "${BATCH_LOG}"
echo "Total experiments: ${TOTAL}" | tee -a "${BATCH_LOG}"
echo "Successful: ${SUCCESS}" | tee -a "${BATCH_LOG}"
echo "Failed: ${FAILED}" | tee -a "${BATCH_LOG}"
echo "Finished at: $(date)" | tee -a "${BATCH_LOG}"
echo "Batch log: ${BATCH_LOG}" | tee -a "${BATCH_LOG}"
echo "========================================" | tee -a "${BATCH_LOG}"

# Show summary of trained configs
if [ ${SUCCESS} -gt 0 ]; then
    echo ""
    echo "成功训练的配置："
    for CONFIG in "${CONFIGS[@]}"; do
        CONFIG_NAME=$(get_config_name "${CONFIG}")
        if [ -d "./outputs/${CONFIG_NAME}" ]; then
            echo "  ✓ ${CONFIG_NAME}"
        fi
    done
fi

# Exit with error if any runs failed
if [ ${FAILED} -gt 0 ]; then
    exit 1
fi


