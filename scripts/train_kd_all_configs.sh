#!/bin/bash
# 统一训练脚本：运行所有 KD 配置
# S2/S3/S4 × T1/T2 (alpha=0.1)

set -e

echo "========================================"
echo "KD 实验批量训练"
echo "========================================"
echo ""
echo "配置: S2/S3/S4 × T1/T2"
echo "Alpha: 0.1 (KD 权重)"
echo "Temperature: 1.0, 2.0"
echo ""

# 设置 GPU 设备
export CUDA_VISIBLE_DEVICES=4
echo "使用 GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 激活 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ 当前 conda 环境: $CONDA_DEFAULT_ENV"
else
    echo "激活 whisper 环境..."
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate whisper
fi

# 获取项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo "✓ 项目根目录: $PROJECT_ROOT"
echo ""

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 状态:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    echo ""
else
    echo "⚠️  警告: 未检测到 GPU"
    echo ""
fi

# 创建日志目录
mkdir -p logs

# 配置列表
configs=(
    "configs/small_kd_s2_t1_a01.yaml"
    "configs/small_kd_s2_t2_a01.yaml"
    "configs/small_kd_s3_t1_a01.yaml"
    "configs/small_kd_s3_t2_a01.yaml"
    "configs/small_kd_s4_t1_a01.yaml"
    "configs/small_kd_s4_t2_a01.yaml"
)

# 配置名称（用于显示）
config_names=(
    "S2-T1"
    "S2-T2"
    "S3-T1"
    "S3-T2"
    "S4-T1"
    "S4-T2"
)

# 预计训练时间（小时）
estimated_times=(
    2.0
    2.0
    2.5
    2.5
    3.0
    3.0
)

# 显示实验计划
echo "========================================"
echo "实验计划"
echo "========================================"
echo ""
printf "%-10s %-40s %-15s\n" "模型" "配置文件" "预计时间"
echo "------------------------------------------------------------------------"
for i in "${!configs[@]}"; do
    printf "%-10s %-40s %-15s\n" "${config_names[$i]}" "${configs[$i]}" "${estimated_times[$i]}h"
done
echo ""

total_time=0
for time in "${estimated_times[@]}"; do
    total_time=$(echo "$total_time + $time" | bc)
done
echo "总预计时间: ${total_time} 小时"
echo ""

# 询问是否继续
read -p "是否开始训练？(y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消训练"
    exit 0
fi

echo ""
echo "========================================"
echo "开始训练"
echo "========================================"
echo ""

# 记录开始时间
start_time=$(date +%s)

# 训练每个配置
success_count=0
failed_count=0
failed_configs=()

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    name="${config_names[$i]}"
    
    echo "========================================"
    echo "[$((i+1))/${#configs[@]}] 训练 $name"
    echo "========================================"
    echo ""
    echo "配置文件: $config"
    
    # 检查配置文件是否存在
    if [ ! -f "$config" ]; then
        echo "✗ 错误: 配置文件不存在: $config"
        failed_count=$((failed_count + 1))
        failed_configs+=("$name (配置文件不存在)")
        continue
    fi
    
    # 提取输出目录
    output_dir=$(grep "output_dir:" "$config" | awk '{print $2}' | tr -d '"' | sed 's|^./||')
    echo "输出目录: $output_dir"
    
    # 创建日志文件名
    log_file="logs/train_${name}_$(date +%Y%m%d_%H%M%S).log"
    echo "日志文件: $log_file"
    echo ""
    
    # 记录本次训练开始时间
    train_start=$(date +%s)
    
    # 运行训练
    echo "开始训练..."
    if python -m src.training.train_kd --config "$config" 2>&1 | tee "$log_file"; then
        train_end=$(date +%s)
        train_duration=$((train_end - train_start))
        train_hours=$(echo "scale=2; $train_duration / 3600" | bc)
        
        echo ""
        echo "✓ $name 训练完成 (用时: ${train_hours}h)"
        
        # 提取最佳 WER
        if [ -f "$output_dir/train.log" ]; then
            best_wer=$(grep "Best WER" "$output_dir/train.log" | tail -1 | grep -oP '\d+\.\d+')
            if [ -n "$best_wer" ]; then
                echo "  最佳 WER: ${best_wer}%"
            fi
        fi
        
        success_count=$((success_count + 1))
    else
        train_end=$(date +%s)
        train_duration=$((train_end - train_start))
        train_hours=$(echo "scale=2; $train_duration / 3600" | bc)
        
        echo ""
        echo "✗ $name 训练失败 (用时: ${train_hours}h)"
        failed_count=$((failed_count + 1))
        failed_configs+=("$name")
    fi
    
    echo ""
    echo "进度: $((i+1))/${#configs[@]} 完成 (成功: $success_count, 失败: $failed_count)"
    echo ""
done

# 记录结束时间
end_time=$(date +%s)
total_duration=$((end_time - start_time))
total_hours=$(echo "scale=2; $total_duration / 3600" | bc)

echo "========================================"
echo "训练完成"
echo "========================================"
echo ""
echo "总用时: ${total_hours} 小时"
echo "成功: $success_count / ${#configs[@]}"
echo "失败: $failed_count / ${#configs[@]}"
echo ""

if [ $failed_count -gt 0 ]; then
    echo "失败的配置:"
    for config in "${failed_configs[@]}"; do
        echo "  - $config"
    done
    echo ""
fi

echo "========================================"
echo "结果汇总"
echo "========================================"
echo ""

printf "%-10s %-15s %-15s\n" "模型" "最佳 WER" "状态"
echo "------------------------------------------------------------------------"

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    name="${config_names[$i]}"
    output_dir=$(grep "output_dir:" "$config" | awk '{print $2}' | tr -d '"' | sed 's|^./||')
    
    if [ -f "$output_dir/train.log" ]; then
        best_wer=$(grep "Best WER" "$output_dir/train.log" | tail -1 | grep -oP '\d+\.\d+' || echo "N/A")
        status="✓ 完成"
    else
        best_wer="N/A"
        status="✗ 失败"
    fi
    
    printf "%-10s %-15s %-15s\n" "$name" "${best_wer}%" "$status"
done

echo ""
echo "========================================"
echo "下一步"
echo "========================================"
echo ""
echo "1. 查看训练日志:"
echo "   tail -100 outputs/small_kd_s2_t1/train.log"
echo ""
echo "2. 对比所有模型结果:"
echo "   grep 'Best WER' outputs/small_kd_s*/train.log"
echo ""
echo "3. 运行评估脚本:"
echo "   python scripts/evaluate_all_models.py"
echo ""
echo "4. 查看详细结果:"
echo "   bash scripts/list_kd_experiments.sh"
echo ""

# 退出码
if [ $failed_count -gt 0 ]; then
    exit 1
else
    exit 0
fi
