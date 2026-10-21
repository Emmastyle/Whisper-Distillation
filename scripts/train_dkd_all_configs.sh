#!/bin/bash
# 批量训练所有 DKD 配置
# DKD (Decoupled Knowledge Distillation) 实验
# 模型: S2/S3/S4 (2/3/4层 decoder)
# Temperature: T=1.0, T=2.0
# Alpha=0.8 (TCKD), Beta=1.2 (NCKD), Gamma=0.1 (CE)

set -e  # Exit on error

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 创建日志目录
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的标题
print_header() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}  $1"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# 打印信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 打印成功
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 打印警告
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 打印错误
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印分隔线
print_separator() {
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
}

print_header "DKD (Decoupled Knowledge Distillation) 批量训练"

print_info "项目目录: $PROJECT_ROOT"
print_info "日志目录: $LOG_DIR"
echo ""

print_info "实验配置:"
echo "  Teacher: openai/whisper-small"
echo "  Student: S2 (2层) / S3 (3层) / S4 (4层)"
echo "  Temperature: T=1.0, T=2.0"
echo "  Alpha: 0.8 (TCKD 权重)"
echo "  Beta: 1.2 (NCKD 权重)"
echo "  Gamma: 0.1 (CE 权重, DKD 权重 = 0.9)"
echo ""

# 设置 GPU 设备
print_info "使用 Device: ${CUDA_VISIBLE_DEVICES:-auto}"
echo ""

# 定义所有配置
declare -a CONFIGS=(
    "small_dkd_s2_t1_a08:S2-T1:2.0h"
    "small_dkd_s2_t2_a08:S2-T2:2.0h"
    "small_dkd_s3_t1_a08:S3-T1:2.5h"
    "small_dkd_s3_t2_a08:S3-T2:2.5h"
    "small_dkd_s4_t1_a08:S4-T1:3.0h"
    "small_dkd_s4_t2_a08:S4-T2:3.0h"
)

TOTAL_CONFIGS=${#CONFIGS[@]}
print_info "总配置数: $TOTAL_CONFIGS"
print_info "预计总时间: ~15 小时"
echo ""

# 显示配置列表
print_separator
printf "  %-5s %-25s %-15s %-15s\n" "序号" "配置" "名称" "预计时间"
print_separator

for i in "${!CONFIGS[@]}"; do
    IFS=':' read -r config_name display_name est_time <<< "${CONFIGS[$i]}"
    printf "  %-5s %-25s %-15s %-15s\n" "$((i+1))" "$config_name" "$display_name" "$est_time"
done

print_separator
echo ""

# 询问是否继续
read -p "是否开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "训练已取消"
    exit 0
fi

echo ""
print_header "开始训练"

# 记录开始时间
START_TIME=$(date +%s)
START_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')

print_info "开始时间: $START_TIME_STR"
echo ""

# 训练结果记录
declare -a RESULTS=()
SUCCESS_COUNT=0
FAIL_COUNT=0

# 训练每个配置
for i in "${!CONFIGS[@]}"; do
    IFS=':' read -r config_name display_name est_time <<< "${CONFIGS[$i]}"
    
    CONFIG_FILE="configs/${config_name}.yaml"
    LOG_FILE="$LOG_DIR/train_${display_name}_$(date +%Y%m%d_%H%M%S).log"
    
    CURRENT=$((i+1))
    
    print_separator
    echo ""
    print_info "[$CURRENT/$TOTAL_CONFIGS] 训练配置: $display_name"
    print_info "配置文件: $CONFIG_FILE"
    print_info "日志文件: $LOG_FILE"
    print_info "预计时间: $est_time"
    echo ""
    
    # 检查配置文件是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "配置文件不存在: $CONFIG_FILE"
        RESULTS+=("$display_name:FAILED:配置文件不存在")
        FAIL_COUNT=$((FAIL_COUNT+1))
        continue
    fi
    
    # 开始训练
    TRAIN_START=$(date +%s)
    print_info "开始训练... ($(date '+%H:%M:%S'))"
    
    # 运行训练 (输出到日志文件和终端)
    if python -m src.training.train_dkd --config "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"; then
        TRAIN_END=$(date +%s)
        TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
        TRAIN_DURATION_STR=$(printf '%02dh:%02dm:%02ds' $((TRAIN_DURATION/3600)) $((TRAIN_DURATION%3600/60)) $((TRAIN_DURATION%60)))
        
        print_success "训练完成! 用时: $TRAIN_DURATION_STR"
        
        # 提取最佳 WER
        BEST_WER=$(grep "Best WER" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+' || echo "N/A")
        print_success "最佳 WER: ${BEST_WER}%"
        
        RESULTS+=("$display_name:SUCCESS:WER=${BEST_WER}%:Time=${TRAIN_DURATION_STR}")
        SUCCESS_COUNT=$((SUCCESS_COUNT+1))
    else
        TRAIN_END=$(date +%s)
        TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
        TRAIN_DURATION_STR=$(printf '%02dh:%02dm:%02ds' $((TRAIN_DURATION/3600)) $((TRAIN_DURATION%3600/60)) $((TRAIN_DURATION%60)))
        
        print_error "训练失败! 用时: $TRAIN_DURATION_STR"
        print_error "查看日志: $LOG_FILE"
        
        RESULTS+=("$display_name:FAILED:Time=${TRAIN_DURATION_STR}")
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
    
    echo ""
    
    # 显示进度
    PROGRESS=$((CURRENT * 100 / TOTAL_CONFIGS))
    print_info "总进度: $CURRENT/$TOTAL_CONFIGS ($PROGRESS%)"
    
    # 估算剩余时间
    if [ $CURRENT -lt $TOTAL_CONFIGS ]; then
        ELAPSED=$(($(date +%s) - START_TIME))
        AVG_TIME=$((ELAPSED / CURRENT))
        REMAINING=$((TOTAL_CONFIGS - CURRENT))
        EST_REMAINING=$((AVG_TIME * REMAINING))
        EST_REMAINING_STR=$(printf '%02dh:%02dm:%02ds' $((EST_REMAINING/3600)) $((EST_REMAINING%3600/60)) $((EST_REMAINING%60)))
        print_info "预计剩余时间: $EST_REMAINING_STR"
    fi
    
    echo ""
done

# 记录结束时间
END_TIME=$(date +%s)
END_TIME_STR=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_DURATION_STR=$(printf '%02dh:%02dm:%02ds' $((TOTAL_DURATION/3600)) $((TOTAL_DURATION%3600/60)) $((TOTAL_DURATION%60)))

# 打印最终结果
print_header "训练完成"

print_info "开始时间: $START_TIME_STR"
print_info "结束时间: $END_TIME_STR"
print_info "总用时: $TOTAL_DURATION_STR"
echo ""

print_info "训练结果统计:"
print_success "成功: $SUCCESS_COUNT"
print_error "失败: $FAIL_COUNT"
echo ""

# 显示详细结果表格
print_separator
printf "  %-15s %-10s %-20s %-15s\n" "配置" "状态" "结果" "用时"
print_separator

for result in "${RESULTS[@]}"; do
    IFS=':' read -r config status info time <<< "$result"
    
    if [ "$status" = "SUCCESS" ]; then
        STATUS_COLOR=$GREEN
        STATUS_SYMBOL="✓"
    else
        STATUS_COLOR=$RED
        STATUS_SYMBOL="✗"
    fi
    
    printf "  %-15s ${STATUS_COLOR}%-10s${NC} %-20s %-15s\n" "$config" "$STATUS_SYMBOL $status" "$info" "$time"
done

print_separator
echo ""

# 保存结果到文件
RESULT_FILE="$LOG_DIR/dkd_training_results_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "DKD Training Results"
    echo "===================="
    echo ""
    echo "Start Time: $START_TIME_STR"
    echo "End Time: $END_TIME_STR"
    echo "Total Duration: $TOTAL_DURATION_STR"
    echo ""
    echo "Success: $SUCCESS_COUNT"
    echo "Failed: $FAIL_COUNT"
    echo ""
    echo "Detailed Results:"
    echo "-----------------"
    for result in "${RESULTS[@]}"; do
        echo "$result"
    done
} > "$RESULT_FILE"

print_success "结果已保存到: $RESULT_FILE"
echo ""

# 显示最佳结果
if [ $SUCCESS_COUNT -gt 0 ]; then
    print_header "最佳结果"
    
    # 提取所有成功的 WER
    BEST_WER=999.99
    BEST_CONFIG=""
    
    for result in "${RESULTS[@]}"; do
        IFS=':' read -r config status info time <<< "$result"
        if [ "$status" = "SUCCESS" ]; then
            WER=$(echo "$info" | grep -oP 'WER=\K[\d.]+' || echo "999.99")
            if (( $(echo "$WER < $BEST_WER" | bc -l) )); then
                BEST_WER=$WER
                BEST_CONFIG=$config
            fi
        fi
    done
    
    if [ -n "$BEST_CONFIG" ]; then
        print_success "最佳配置: $BEST_CONFIG"
        print_success "最佳 WER: ${BEST_WER}%"
        echo ""
        
        # 与 baseline 对比
        print_info "与 Baseline 对比:"
        echo "  CE Baseline:"
        echo "    • small_ce_s2: 23.29% WER"
        echo "    • small_ce_s3: 18.88% WER"
        echo ""
        echo "  KD Baseline (如果已完成):"
        echo "    • 查看 outputs/small_kd_s*/train.log"
        echo ""
        echo "  DKD (本次实验):"
        echo "    • $BEST_CONFIG: ${BEST_WER}% WER"
        echo ""
    fi
fi

# 提供后续步骤
print_header "后续步骤"

echo "1. 查看训练日志:"
echo "   ls -lh $LOG_DIR/"
echo ""

echo "2. 查看最佳模型:"
echo "   ls -lh outputs/small_dkd_*/best_model.pt"
echo ""

echo "3. 对比所有结果:"
echo "   grep 'Best WER' outputs/small_dkd_*/train.log"
echo ""

echo "4. 分析 TCKD vs NCKD:"
echo "   grep 'tckd=.*nckd=' logs/train_*.log | tail -20"
echo ""

echo "5. 可视化结果 (如果有工具):"
echo "   python scripts/plot_results.py"
echo ""

print_separator
print_success "所有训练任务完成! 🎉"
print_separator
echo ""

