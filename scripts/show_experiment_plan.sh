#!/bin/bash
# 显示实验计划摘要

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         KD Temperature 实验计划                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo "📋 实验配置"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Teacher: openai/whisper-small"
echo "  Student: S2 (2层) / S3 (3层) / S4 (4层)"
echo "  Temperature: T=1.0, T=2.0"
echo "  Alpha: 0.1 (CE 权重 10%, KD 权重 90%)"
echo "  Learning Rate: 1e-4"
echo "  Batch Size: 16"
echo "  Epochs: 15"
echo ""

echo "📊 配置矩阵"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "  %-8s %-12s %-12s %-40s\n" "模型" "T=1.0" "T=2.0" "配置文件"
echo "  ────────────────────────────────────────────────────────────────"
printf "  %-8s %-12s %-12s %-40s\n" "S2" "α=0.1" "α=0.1" "small_kd_s2_t{1,2}_a01.yaml"
printf "  %-8s %-12s %-12s %-40s\n" "S3" "α=0.1" "α=0.1" "small_kd_s3_t{1,2}_a01.yaml"
printf "  %-8s %-12s %-12s %-40s\n" "S4" "α=0.1" "α=0.1" "small_kd_s4_t{1,2}_a01.yaml"
echo ""

echo "⏱️  预计时间"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "  %-10s %-15s %-15s\n" "模型" "每个配置" "两个温度"
echo "  ────────────────────────────────────────────────────"
printf "  %-10s %-15s %-15s\n" "S2" "~2.0h" "~4.0h"
printf "  %-10s %-15s %-15s\n" "S3" "~2.5h" "~5.0h"
printf "  %-10s %-15s %-15s\n" "S4" "~3.0h" "~6.0h"
echo "  ────────────────────────────────────────────────────"
printf "  %-10s %-15s %-15s\n" "总计" "-" "~15.0h"
echo ""

echo "🎯 实验目标"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. 验证 KD 是否优于 CE baseline"
echo "  2. 找到最佳 Temperature (T=1 vs T=2)"
echo "  3. 分析模型大小的影响 (S2 vs S3 vs S4)"
echo ""
echo "  CE Baseline (已完成):"
echo "    • small_ce_s2: 23.29% WER"
echo "    • small_ce_s3: 18.88% WER"
echo ""
echo "  KD 目标:"
echo "    • small_kd_s2: < 23% WER ✨"
echo "    • small_kd_s3: < 18% WER ✨"
echo "    • small_kd_s4: < 15% WER ✨"
echo ""

echo "🚀 快速开始"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1️⃣  验证配置:"
echo "     python scripts/verify_configs.py"
echo ""
echo "  2️⃣  批量训练 (所有 6 个配置):"
echo "     bash scripts/train_kd_all_configs.sh"
echo ""
echo "  3️⃣  单独训练 (快速测试):"
echo "     python -m src.training.train_kd --config configs/small_kd_s2_t2_a01.yaml"
echo ""
echo "  4️⃣  监控训练:"
echo "     tail -f outputs/small_kd_s2_t2/train.log"
echo ""
echo "  5️⃣  查看结果:"
echo "     grep 'Best WER' outputs/small_kd_s*/train.log"
echo ""

echo "📁 文件清单"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 检查配置文件
config_count=$(ls -1 configs/small_kd_s*_t*_a01.yaml 2>/dev/null | wc -l)
echo "  配置文件: $config_count / 6"
if [ $config_count -eq 6 ]; then
    echo "    ✅ 所有配置文件已就绪"
else
    echo "    ⚠️  配置文件不完整"
fi

# 检查训练脚本
if [ -f "scripts/train_kd_all_configs.sh" ]; then
    echo "  训练脚本: ✅ train_kd_all_configs.sh"
else
    echo "  训练脚本: ❌ 未找到"
fi

# 检查验证脚本
if [ -f "scripts/verify_configs.py" ]; then
    echo "  验证脚本: ✅ verify_configs.py"
else
    echo "  验证脚本: ❌ 未找到"
fi

# 检查训练代码
if [ -f "src/training/train_kd.py" ]; then
    echo "  训练代码: ✅ train_kd.py"
else
    echo "  训练代码: ❌ 未找到"
fi

echo ""

# 检查 GPU
echo "💻 系统状态"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "  GPU: ✅ 可用"
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "  GPU 数量: $gpu_count"
    
    # 显示 GPU 信息
    nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader | while read line; do
        echo "    • $line"
    done
else
    echo "  GPU: ⚠️  未检测到"
fi

echo ""

# 检查 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "  Conda 环境: ✅ $CONDA_DEFAULT_ENV"
else
    echo "  Conda 环境: ⚠️  未激活"
    echo "    运行: conda activate whisper"
fi

echo ""

echo "📖 文档"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  • KD_EXPERIMENTS_GUIDE.md - 完整实验指南"
echo "  • README.md - 项目说明"
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  准备就绪！运行 'bash scripts/train_kd_all_configs.sh' 开始   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

