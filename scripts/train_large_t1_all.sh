#!/bin/bash
# Batch training script for whisper-large teacher with T=1
# Classical KD and DKD for S2/S3/S4 students
# Total: 6 experiments

set -e  # Exit on error

# Set PYTHONPATH to project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
PROJECT_ROOT="/home/zz359/workspace-whisper"
LOG_DIR="${PROJECT_ROOT}/logs"
PYTHON="python"

# Create log directory
mkdir -p "${LOG_DIR}"

# Generate timestamp for this batch
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BATCH_LOG="${LOG_DIR}/batch_large_t1_${BATCH_TIMESTAMP}.log"

echo "========================================" | tee -a "${BATCH_LOG}"
echo "Starting Large-T1 Batch Training" | tee -a "${BATCH_LOG}"
echo "Timestamp: ${BATCH_TIMESTAMP}" | tee -a "${BATCH_LOG}"
echo "========================================" | tee -a "${BATCH_LOG}"
echo "" | tee -a "${BATCH_LOG}"

cd "${PROJECT_ROOT}"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Array of configurations
CONFIGS=(
    "configs/large_kd_s2_t1.yaml"
    "configs/large_kd_s3_t1.yaml"
    "configs/large_kd_s4_t1.yaml"
    "configs/large_dkd_s2_t1.yaml"
    "configs/large_dkd_s3_t1.yaml"
    "configs/large_dkd_s4_t1.yaml"
)

# Training scripts mapping
declare -A TRAIN_SCRIPTS
TRAIN_SCRIPTS["large_kd_s2_t1"]="src/training/train_kd.py"
TRAIN_SCRIPTS["large_kd_s3_t1"]="src/training/train_kd.py"
TRAIN_SCRIPTS["large_kd_s4_t1"]="src/training/train_kd.py"
TRAIN_SCRIPTS["large_dkd_s2_t1"]="src/training/train_dkd.py"
TRAIN_SCRIPTS["large_dkd_s3_t1"]="src/training/train_dkd.py"
TRAIN_SCRIPTS["large_dkd_s4_t1"]="src/training/train_dkd.py"

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
    SCRIPT="${TRAIN_SCRIPTS[${CONFIG_NAME}]}"
    
    echo "========================================" | tee -a "${BATCH_LOG}"
    echo "[$CURRENT/$TOTAL] Training: ${CONFIG_NAME}" | tee -a "${BATCH_LOG}"
    echo "Config: ${CONFIG}" | tee -a "${BATCH_LOG}"
    echo "Script: ${SCRIPT}" | tee -a "${BATCH_LOG}"
    echo "Started at: $(date)" | tee -a "${BATCH_LOG}"
    echo "========================================" | tee -a "${BATCH_LOG}"
    echo "" | tee -a "${BATCH_LOG}"
    
    # Create individual log file
    RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_LOG="${LOG_DIR}/${CONFIG_NAME}_${RUN_TIMESTAMP}.log"
    
    # Run training
    if ${PYTHON} "${SCRIPT}" --config "${CONFIG}" 2>&1 | tee "${RUN_LOG}"; then
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
        echo "Error details in batch log above" | tee -a "${BATCH_LOG}"
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

# Exit with error if any runs failed
if [ ${FAILED} -gt 0 ]; then
    exit 1
fi

