#!/bin/bash
# 列出所有 KD Temperature 实验配置

echo "========================================"
echo "KD Temperature 实验配置列表"
echo "========================================"
echo ""

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo "📋 配置文件列表："
echo ""
ls -1 configs/small_kd_s*_t*.yaml | while read config; do
    name=$(basename "$config" .yaml)
    
    # 提取配置信息
    layers=$(grep "student_decoder_layers:" "$config" | awk '{print $2}')
    temp=$(grep "temperature:" "$config" | grep -v "#" | awk '{print $2}')
    lr=$(grep "learning_rate:" "$config" | awk '{print $2}')
    output=$(grep "output_dir:" "$config" | awk '{print $2}' | tr -d '"')
    
    echo "📄 $name"
    echo "   Layers: $layers | Temperature: $temp | LR: $lr"
    echo "   Output: $output"
    echo ""
done

echo "========================================"
echo "📊 实验矩阵："
echo "========================================"
echo ""
echo "| 模型 | Temperature | Learning Rate | 配置文件 |"
echo "|------|-------------|---------------|----------|"

for layers in 2 3 4; do
    for temp in 1 2; do
        config="configs/small_kd_s${layers}_t${temp}.yaml"
        if [ -f "$config" ]; then
            lr=$(grep "learning_rate:" "$config" | awk '{print $2}')
            echo "| S$layers   | $temp.0         | $lr       | small_kd_s${layers}_t${temp}.yaml |"
        fi
    done
    
    # LR=0.1x 版本
    config="configs/small_kd_s${layers}_t2_lr01.yaml"
    if [ -f "$config" ]; then
        lr=$(grep "learning_rate:" "$config" | awk '{print $2}')
        echo "| S$layers   | 2.0         | $lr       | small_kd_s${layers}_t2_lr01.yaml |"
    fi
done

echo ""
echo "========================================"
echo "🚀 快速启动命令："
echo "========================================"
echo ""
echo "# 运行所有实验（批量）"
echo "bash scripts/train_kd_temperature_sweep.sh"
echo ""
echo "# 单独运行（示例）"
echo "bash scripts/train_kd_t1_s2.sh  # T=1.0, S2"
echo "bash scripts/train_kd_t2_s2.sh  # T=2.0, S2"
echo "bash scripts/train_kd_t1_s3.sh  # T=1.0, S3"
echo "bash scripts/train_kd_t2_s3.sh  # T=2.0, S3"
echo ""
echo "# 直接调用 Python"
echo "python -m src.training.train_kd --config configs/small_kd_s2_t1.yaml"
echo ""

echo "========================================"
echo "📁 输出目录："
echo "========================================"
echo ""
ls -1d outputs/small_kd_s*_t* 2>/dev/null | while read dir; do
    if [ -d "$dir" ]; then
        name=$(basename "$dir")
        if [ -f "$dir/train.log" ]; then
            lines=$(wc -l < "$dir/train.log")
            last_line=$(tail -1 "$dir/train.log")
            echo "✓ $name (train.log: $lines 行)"
            if echo "$last_line" | grep -q "Training completed"; then
                echo "  状态: 已完成 ✅"
            elif echo "$last_line" | grep -q "Step"; then
                echo "  状态: 训练中 🔄"
            else
                echo "  状态: 未知"
            fi
        else
            echo "○ $name (未开始)"
        fi
    fi
done

if ! ls -1d outputs/small_kd_s*_t* 2>/dev/null | grep -q .; then
    echo "暂无输出目录（实验尚未开始）"
fi

echo ""
echo "========================================"
echo "📖 详细说明："
echo "========================================"
echo "查看 KD_TEMPERATURE_EXPERIMENTS.md 了解实验设计和预期结果"
echo ""

