#!/bin/bash
# 温度扫描实验脚本
# 对 S3 student 使用不同温度进行 KD 训练
# 测试温度: T = 2, 3, 4, 5, 6, 8

set -e

echo "========================================"
echo "温度扫描实验 - S3 Student"
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

# 获取项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo "项目根目录: $PROJECT_ROOT"
echo ""

# 创建临时配置目录
mkdir -p configs/temp
mkdir -p logs

# 温度列表
temperatures=(2 3 4 5 6 8)

echo "测试温度: ${temperatures[@]}"
echo ""

# 为每个温度创建配置并训练
for temp in "${temperatures[@]}"; do
    config_file="configs/temp/small_kd_s3_t${temp}.yaml"
    output_dir="outputs/small_kd_s3_t${temp}"
    log_file="logs/small_kd_s3_t${temp}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "----------------------------------------"
    echo "创建配置: 温度 T=$temp"
    echo "----------------------------------------"
    
    # 基于 small_kd_s3.yaml 创建新配置
    cat > "$config_file" << EOF
# Temperature Sweep: T=$temp for S3 student

model:
  teacher_model: "openai/whisper-small"
  student_decoder_layers: 3
  student_decoder_heads: 12
  student_hidden_size: 768
  dropout: 0.1

data:
  dataset_name: "librispeech_asr"
  dataset_config: "clean"
  split: "train.100"
  num_samples: null
  max_duration: 10.0
  batch_size: 16
  num_workers: 8
  sample_rate: 16000
  train_ratio: 0.9
  val_ratio: 0.05
  test_ratio: 0.05

training:
  epochs: 15
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 500
  gradient_clip: 1.0
  log_interval: 100
  eval_interval: 1000
  save_interval: 2000
  output_dir: "$output_dir"
  seed: 42
  fp16: true

kd:
  temperature: $temp.0
  alpha: 0.5
EOF
    
    echo "配置文件: $config_file"
    echo "输出目录: $output_dir"
    echo "日志文件: $log_file"
    echo ""
    
    echo "开始训练 (T=$temp)..."
    python -m src.training.train_kd --config "$config_file" 2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ T=$temp 训练完成"
    else
        echo "✗ T=$temp 训练失败"
        exit 1
    fi
    echo ""
done

echo "========================================"
echo "✓ 温度扫描实验完成！"
echo "========================================"
echo ""
echo "训练结果保存在: outputs/small_kd_s3_t*/"
echo "训练日志保存在: logs/small_kd_s3_t*.log"
echo ""
echo "下一步："
echo "运行评估脚本分析不同温度的效果: python scripts/evaluate_all_models.py"
echo ""

