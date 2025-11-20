# Phase 2 待执行任务清单

## 当前状态
✅ **代码和配置已全部完成**  
🔄 **以下实验需要您执行（需要GPU和时间）**

## 查看已有结果

### 1. 数据分析结果（已完成 ✅）
```bash
cat outputs/data_100h_statistics.txt
```

**位置**: `outputs/data_100h_statistics.txt`

**内容摘要**:
- 总样本数: 28,539
- 有效样本数: 5,366（≤10秒）
- 训练集: 8.23 小时
- 验证集: 0.49 小时
- 测试集: 0.45 小时

---

## 需要运行的实验

### Step 1: Teacher Model 基线测试（约10-30分钟）

**运行命令**:
```bash
cd /home/zz359/workspace-whisper
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper
python scripts/test_teacher_small.py --num_samples 100
```

**结果位置**:
- `outputs/teacher_small_report.txt` - 性能报告
- `outputs/teacher_small_samples.txt` - 预测样本

**包含内容**:
- whisper-small 的 WER/CER
- 推理速度（RTF）
- 显存占用
- 参数量

---

### Step 2: 训练 CE-only 模型（约6-12小时）

**运行命令**:
```bash
# S2 (2层 decoder)
bash scripts/train_ce.sh configs/small_ce_s2.yaml

# S3 (3层 decoder) - 推荐优先运行
bash scripts/train_ce.sh configs/small_ce_s3.yaml

# S4 (4层 decoder)
bash scripts/train_ce.sh configs/small_ce_s4.yaml
```

**结果位置**:
- `outputs/small_ce_s2/best_model.pt` + `train.log`
- `outputs/small_ce_s3/best_model.pt` + `train.log`
- `outputs/small_ce_s4/best_model.pt` + `train.log`

**查看训练日志**:
```bash
# 实时查看
tail -f logs/small_ce_s3_*.log

# 或查看已完成的训练输出
cat outputs/small_ce_s3/train.log
```

---

### Step 3: 训练 KD 模型（约6-12小时）

**运行命令**:
```bash
# S2 KD
bash scripts/train_kd.sh configs/small_kd_s2.yaml

# S3 KD - 推荐优先运行
bash scripts/train_kd.sh configs/small_kd_s3.yaml

# S4 KD
bash scripts/train_kd.sh configs/small_kd_s4.yaml
```

**结果位置**:
- `outputs/small_kd_s2/best_model.pt` + `train.log`
- `outputs/small_kd_s3/best_model.pt` + `train.log`
- `outputs/small_kd_s4/best_model.pt` + `train.log`

---

### Step 4: 温度扫描实验（约12-24小时，可选）

**运行命令**:
```bash
bash scripts/train_temperature_sweep.sh
```

**结果位置**:
- `outputs/small_kd_s3_t2/`
- `outputs/small_kd_s3_t3/`
- `outputs/small_kd_s3_t4/`（与 Step 3 的 S3 KD 相同）
- `outputs/small_kd_s3_t5/`
- `outputs/small_kd_s3_t6/`
- `outputs/small_kd_s3_t8/`

---

### Step 5: 统一评估（约30-60分钟）

**前提**: 至少完成 Step 1-3

**运行命令**:
```bash
python scripts/evaluate_all_models.py --num_samples 100
```

**结果位置**:
- `results/model_comparison.csv` - 对比表格（CSV格式）
- `results/model_comparison.tex` - 对比表格（LaTeX格式）
- `results/speed_accuracy_curve.png` - 速度-准确率曲线
- `results/speed_accuracy_curve.pdf` - 速度-准确率曲线（PDF）
- `results/temperature_comparison.png` - 温度对比图（如果运行了Step 4）
- `results/temperature_comparison.pdf` - 温度对比图（PDF）

**查看结果**:
```bash
# 查看CSV表格
cat results/model_comparison.csv

# 或在Python中查看
python -c "import pandas as pd; print(pd.read_csv('results/model_comparison.csv'))"

# 下载图片到本地查看
# 使用 scp 或其他方式下载 results/*.png
```

---

## 最小化快速开始（推荐）

如果时间有限，只运行核心实验：

```bash
# 1. Teacher 基线
python scripts/test_teacher_small.py --num_samples 50

# 2. 只训练 S3（最平衡的配置）
bash scripts/train_ce.sh configs/small_ce_s3.yaml
bash scripts/train_kd.sh configs/small_kd_s3.yaml

# 3. 评估这两个模型
python scripts/evaluate_all_models.py --num_samples 50
```

**预计时间**: 4-8 小时

---

## 监控训练进度

### 查看GPU使用情况
```bash
watch -n 1 nvidia-smi
```

### 查看训练日志
```bash
# 查看最新的训练日志
ls -lt logs/*.log | head -5
tail -f logs/$(ls -t logs/*.log | head -1)
```

### 检查已完成的模型
```bash
ls -lh outputs/small_*/best_model.pt
```

### 查看磁盘空间
```bash
df -h
du -sh outputs/
```

---

## 目录结构说明

```
workspace-whisper/
├── outputs/                    # 训练输出
│   ├── data_100h_statistics.txt      ✅ 已生成
│   ├── teacher_small_report.txt      🔄 待生成
│   ├── small_ce_s2/                  🔄 待训练
│   ├── small_ce_s3/                  🔄 待训练
│   ├── small_ce_s4/                  🔄 待训练
│   ├── small_kd_s2/                  🔄 待训练
│   ├── small_kd_s3/                  🔄 待训练
│   └── small_kd_s4/                  🔄 待训练
│
├── results/                    # 评估结果
│   ├── model_comparison.csv          🔄 待生成
│   ├── model_comparison.tex          🔄 待生成
│   ├── speed_accuracy_curve.png      🔄 待生成
│   └── speed_accuracy_curve.pdf      🔄 待生成
│
└── logs/                       # 训练日志
    └── *.log                         🔄 训练时生成
```

---

## 预期结果说明

### Teacher Model 报告示例
```
模型名称: openai/whisper-small
总参数量: 244M
WER: ~5-8%
RTF: <1.0
推理速度: >1.0x 实时
```

### 模型对比表格示例
```csv
Model,Type,Layers,Params(M),WER(%),RTF,Speed(x)
Teacher,Teacher,12,244,5.2,0.45,2.2
S2 CE,CE-only,2,85,8.5,0.15,6.7
S2 KD,KD,2,85,7.2,0.15,6.7
S3 CE,CE-only,3,95,7.1,0.20,5.0
S3 KD,KD,3,95,6.3,0.20,5.0
...
```

---

## 详细文档位置

- **执行指南**: `doc/phase2_execution_guide.md`
- **实施进度**: `PHASE2_PROGRESS.md`
- **完成报告**: `PHASE2_IMPLEMENTATION_COMPLETE.md`
- **结果模板**: `doc/phase2_results.md`

---

## 推荐执行顺序

1. ✅ **已完成**: 数据分析
2. 🔄 **立即可做**: Teacher 基线测试（10分钟）
3. 🔄 **核心实验**: S3 的 CE 和 KD 训练（4-8小时）
4. 🔄 **扩展实验**: S2 和 S4 训练（可选）
5. 🔄 **温度扫描**: 6个温度的实验（可选，12-24小时）
6. 🔄 **最终评估**: 统一评估所有模型（30分钟）

---

## 帮助

如有问题，请查看：
- 故障排除: `doc/phase2_execution_guide.md` 的 "故障排除" 章节
- 训练日志: `logs/*.log`
- GPU状态: `nvidia-smi`

---

**下一步建议**: 
先运行 `python scripts/test_teacher_small.py --num_samples 100` 验证环境，
然后开始训练 S3 模型（CE + KD），这是最有代表性的配置。

