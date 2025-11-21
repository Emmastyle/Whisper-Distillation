#!/bin/bash
# 快速训练脚本: KD S2 模型 (T=1)

set -e

export CUDA_VISIBLE_DEVICES=3

# 激活环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate whisper
fi

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo "训练 KD S2 模型 (Temperature=1.0)"
python -m src.training.train_kd --config configs/small_kd_s2_t1.yaml

