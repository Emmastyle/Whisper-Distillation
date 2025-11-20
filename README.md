# Whisper Knowledge Distillation - Phase 1

基于 Whisper 的知识蒸馏项目，Phase 1 实现了小规模数据集上的 CE 和 KD 基线。

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

- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始指南，5 分钟上手
- **[STRUCTURE.md](STRUCTURE.md)** - 详细的项目结构说明
- **[PHASE1_COMPLETION.md](PHASE1_COMPLETION.md)** - Phase 1 完成报告
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - 项目总结

## 后续计划

Phase 2 将在此基础上：
- 扩展到更大数据集
- 实现 DKD（Decoupled Knowledge Distillation）
- 更多模型压缩技术

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

