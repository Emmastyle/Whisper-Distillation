# Whisper Knowledge Distillation

基于 Whisper 的知识蒸馏项目，实现渐进式模型压缩与知识传递。

- **Phase 1 (已完成)**: whisper-tiny 基线验证（小数据集）
- **Phase 2 (进行中)**: whisper-small 完整实验（~9h 数据）+ 温度优化

## 项目结构

```
.
├── configs/              # 配置文件
├── src/
│   ├── data/            # 数据集与 DataLoader
│   ├── models/          # Teacher 和 Student 模型
│   ├── training/        # 训练循环（CE / KD）
│   └── utils/           # 工具函数（日志、metrics）
├── scripts/             # 一键运行脚本
└── requirements.txt     # 依赖清单
```

## 环境设置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 硬件要求

- Python 3.10+
- 单卡 GPU（A100 / V100 / RTX 3090 或同等）
- 推荐：16GB+ GPU 显存

## 使用方法

### 训练 CE Baseline

```bash
python -m src.training.train_ce --config configs/ce_tiny_s2.yaml
```

或使用脚本：

```bash
bash scripts/train_ce.sh
```

### 训练 KD Baseline

```bash
python -m src.training.train_kd --config configs/kd_tiny_s2.yaml
```

或使用脚本：

```bash
bash scripts/train_kd.sh
```

## 实验结果

### CE vs KD on Tiny Dataset

| 方法 | WER (%) | CER (%) | 训练时长 |
|------|---------|---------|----------|
| CE-only | TBD | TBD | TBD |
| KD (T=2, α=0.5) | TBD | TBD | TBD |

## Phase 1 目标

1. ✅ Whisper-tiny 作为 teacher encoder + decoder（推理）
2. ✅ Student：复用 teacher encoder（冻结），自定义小 decoder
3. ✅ CE-only baseline 训练与评估
4. ✅ Standard KD（Hinton）验证
5. ✅ 清晰的 repo 结构，便于 Phase 2 扩展

## 配置说明

主要配置参数（在 `configs/*.yaml` 中）：

- `data.max_duration`: 最大音频时长（秒）
- `data.num_samples`: 训练样本数
- `data.batch_size`: 批次大小
- `model.student_decoder_layers`: Student decoder 层数
- `training.epochs`: 训练轮数
- `training.learning_rate`: 学习率
- `kd.temperature`: KD 温度（仅 KD 模式）
- `kd.alpha`: CE/KD 损失权重（仅 KD 模式）

## 文档导航

### Phase 1 文档
- **[doc/QUICKSTART.md](doc/QUICKSTART.md)** - 快速开始指南
- **[doc/STRUCTURE.md](doc/STRUCTURE.md)** - 项目结构说明
- **[doc/PHASE1_COMPLETION.md](doc/PHASE1_COMPLETION.md)** - Phase 1 完成报告
- **[doc/Project Plan Phase 1.md](doc/Project%20Plan%20Phase%201.md)** - Phase 1 计划

### Phase 2 文档
- **[doc/Project Plan Phase 2.md](doc/Project%20Plan%20Phase%202.md)** - Phase 2 详细计划
- **[doc/phase2_execution_guide.md](doc/phase2_execution_guide.md)** - Phase 2 执行指南
- **[doc/phase2_results.md](doc/phase2_results.md)** - Phase 2 结果报告（模板）
- **[PHASE2_PROGRESS.md](PHASE2_PROGRESS.md)** - Phase 2 实施进度

### 整体规划
- **[doc/Project Plan.md](doc/Project%20Plan.md)** - 完整项目计划
- **[doc/PROJECT_ROADMAP.md](doc/PROJECT_ROADMAP.md)** - 项目路线图

## Phase 2 快速开始

### 验证 Phase 2 设置

```bash
cd /home/zz359/workspace-whisper
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper
python test_phase2_setup.py
```

### 运行数据分析

```bash
python scripts/analyze_data_100h.py
```

### 开始训练

```bash
# 批量训练所有模型（CE + KD）
bash scripts/train_phase2_all.sh

# 或单独训练
bash scripts/train_ce.sh configs/small_ce_s3.yaml
bash scripts/train_kd.sh configs/small_kd_s3.yaml
```

### 温度扫描实验

```bash
bash scripts/train_temperature_sweep.sh
```

### 评估所有模型

```bash
python scripts/evaluate_all_models.py --num_samples 100
```

详细步骤请参考 **[doc/phase2_execution_guide.md](doc/phase2_execution_guide.md)**

## Phase 3 展望

- DKD (Decoupled Knowledge Distillation)
  - TCKD (Target-Class KD)
  - NCKD (Non-Target-Class KD)
- 模型量化与剪枝
- 更大规模数据集

## 快速验证

在开始实验前，运行测试脚本验证环境：

```bash
python test_setup.py
```

这会测试：
- 设备（CPU/GPU）
- 配置加载
- 数据集加载（10 样本）
- 模型初始化
- 前向传播和生成

