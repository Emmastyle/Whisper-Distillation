#!/bin/bash
# KD Temperature 实验脚本
# 测试 T=1, T=2 以及不同学习率对 KD 训练的影响
# 训练 S2/S3/S4 模型

set -e

echo "========================================"
echo "KD Temperature 实验脚本"
echo "========================================"
echo ""

# 设置 GPU 设备
export CUDA_VISIBLE_DEVICES=4
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
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
    
    python -m src.training.train_kd --config "$config" 2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ $model_name 训练完成"
    else
        echo "✗ $model_name 训练失败"
        exit 1
    fi
    echo ""
}

# 实验计划
echo "========================================"
echo "实验设置："
echo "  - Temperature: T=1, T=2"
echo "  - Learning Rate: 1e-4, 1e-5 (0.1x)"
echo "  - 模型架构: S2 (2层), S3 (3层), S4 (4层)"
echo "  - Alpha: 0.5 (CE和KD权重相等)"
echo "========================================"
echo ""

# 实验 1: Temperature = 1.0 (Sharp Distribution)
echo "========================================"
echo "实验 1: Temperature = 1.0"
echo "========================================"
echo ""

train_model "configs/small_kd_s2_t1.yaml" "small_kd_s2_t1"
train_model "configs/small_kd_s3_t1.yaml" "small_kd_s3_t1"
train_model "configs/small_kd_s4_t1.yaml" "small_kd_s4_t1"

# 实验 2: Temperature = 2.0 (Moderate Distribution)
echo "========================================"
echo "实验 2: Temperature = 2.0"
echo "========================================"
echo ""

train_model "configs/small_kd_s2_t2.yaml" "small_kd_s2_t2"
train_model "configs/small_kd_s3_t2.yaml" "small_kd_s3_t2"
train_model "configs/small_kd_s4_t2.yaml" "small_kd_s4_t2"

# 实验 3: Temperature = 2.0 + Lower LR (0.1x)
echo "========================================"
echo "实验 3: Temperature = 2.0 + LR = 1e-5"
echo "========================================"
echo ""

train_model "configs/small_kd_s2_t2_lr01.yaml" "small_kd_s2_t2_lr01"
train_model "configs/small_kd_s3_t2_lr01.yaml" "small_kd_s3_t2_lr01"
train_model "configs/small_kd_s4_t2_lr01.yaml" "small_kd_s4_t2_lr01"

echo "========================================"
echo "✓ 所有 KD Temperature 实验完成！"
echo "========================================"
echo ""
echo "实验结果保存在: outputs/"
echo "训练日志保存在: logs/"
echo ""
echo "实验配置总结："
echo ""
echo "| 模型 | Temperature | Learning Rate | 输出目录 |"
echo "|------|-------------|---------------|----------|"
echo "| S2   | 1.0         | 1e-4          | outputs/small_kd_s2_t1 |"
echo "| S3   | 1.0         | 1e-4          | outputs/small_kd_s3_t1 |"
echo "| S4   | 1.0         | 1e-4          | outputs/small_kd_s4_t1 |"
echo "| S2   | 2.0         | 1e-4          | outputs/small_kd_s2_t2 |"
echo "| S3   | 2.0         | 1e-4          | outputs/small_kd_s3_t2 |"
echo "| S4   | 2.0         | 1e-4          | outputs/small_kd_s4_t2 |"
echo "| S2   | 2.0         | 1e-5          | outputs/small_kd_s2_t2_lr01 |"
echo "| S3   | 2.0         | 1e-5          | outputs/small_kd_s3_t2_lr01 |"
echo "| S4   | 2.0         | 1e-5          | outputs/small_kd_s4_t2_lr01 |"
echo ""
echo "下一步："
echo "1. 评估所有模型: python scripts/evaluate_all_models.py"
echo "2. 分析结果: 查看各个 outputs/*/train.log"
echo ""

