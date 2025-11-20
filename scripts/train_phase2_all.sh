#!/bin/bash
# Phase 2 批量训练脚本
# 训练所有 CE-only 和 KD 配置的 S2/S3/S4 模型

set -e

echo "========================================"
echo "Phase 2 批量训练脚本"
echo "========================================"
echo ""

# 激活 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "当前 conda 环境: $CONDA_DEFAULT_ENV"
else
    echo "激活 whisper 环境..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate whisper
fi

# 获取脚本所在目录的父目录（项目根目录）
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"
echo ""

# 创建输出目录
mkdir -p outputs
mkdir -p logs

# 训练函数
train_model() {
    local config=$1
    local model_name=$2
    local log_file="logs/${model_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "----------------------------------------"
    echo "开始训练: $model_name"
    echo "配置文件: $config"
    echo "日志文件: $log_file"
    echo "----------------------------------------"
    
    if [[ $config == *"_ce_"* ]]; then
        # CE-only 训练
        python -m src.training.train_ce --config "$config" 2>&1 | tee "$log_file"
    else
        # KD 训练
        python -m src.training.train_kd --config "$config" 2>&1 | tee "$log_file"
    fi
    
    if [ $? -eq 0 ]; then
        echo "✓ $model_name 训练完成"
    else
        echo "✗ $model_name 训练失败"
        exit 1
    fi
    echo ""
}

# Phase 2 训练计划
# 按照顺序训练，确保每个模型都成功完成

echo "========================================"
echo "Step 1: CE-only 训练（S2/S3/S4）"
echo "========================================"
echo ""

train_model "configs/small_ce_s2.yaml" "small_ce_s2"
train_model "configs/small_ce_s3.yaml" "small_ce_s3"
train_model "configs/small_ce_s4.yaml" "small_ce_s4"

echo "========================================"
echo "Step 2: KD 训练（S2/S3/S4，温度 T=4）"
echo "========================================"
echo ""

train_model "configs/small_kd_s2.yaml" "small_kd_s2"
train_model "configs/small_kd_s3.yaml" "small_kd_s3"
train_model "configs/small_kd_s4.yaml" "small_kd_s4"

echo "========================================"
echo "✓ Phase 2 所有模型训练完成！"
echo "========================================"
echo ""
echo "训练结果保存在: outputs/"
echo "训练日志保存在: logs/"
echo ""
echo "下一步："
echo "1. 运行温度实验（S3）: scripts/train_temperature_sweep.sh"
echo "2. 评估所有模型: scripts/evaluate_all_models.py"
echo ""

