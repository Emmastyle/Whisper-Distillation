# 项目结构说明

## 目录树

```
workspace-whisper/
├── configs/                      # 配置文件
│   ├── ce_tiny_s2.yaml          # CE baseline 配置
│   └── kd_tiny_s2.yaml          # KD baseline 配置
│
├── scripts/                      # 运行脚本
│   ├── train_ce.sh              # CE 训练脚本
│   └── train_kd.sh              # KD 训练脚本
│
├── src/                         # 源代码
│   ├── __init__.py
│   │
│   ├── data/                    # 数据模块
│   │   ├── __init__.py
│   │   └── dataset.py           # WhisperASRDataset 与 collate_fn
│   │
│   ├── models/                  # 模型模块
│   │   ├── __init__.py
│   │   ├── teacher.py           # Teacher Whisper 封装
│   │   ├── student_decoder.py   # Student Decoder 实现
│   │   └── student_asr_model.py # 完整 Student ASR 模型
│   │
│   ├── training/                # 训练模块
│   │   ├── __init__.py
│   │   ├── train_ce.py          # CE baseline 训练脚本
│   │   └── train_kd.py          # KD baseline 训练脚本
│   │
│   └── utils/                   # 工具模块
│       ├── __init__.py
│       ├── config.py            # 配置加载与定义
│       ├── logger.py            # 日志工具
│       └── metrics.py           # WER/CER 计算
│
├── .gitignore                   # Git 忽略文件
├── README.md                    # 项目说明
├── requirements.txt             # 依赖清单
├── test_setup.py               # 设置测试脚本
└── STRUCTURE.md                # 本文件
```

## 模块说明

### 1. 数据模块 (`src/data/`)

- **dataset.py**: 实现了 `WhisperASRDataset` 类
  - 加载 LibriSpeech 数据集
  - 使用 Whisper feature extractor 提取 log-mel 特征
  - 使用 Whisper tokenizer 处理文本
  - 支持按时长过滤和样本数限制
  - 提供 `whisper_collate_fn` 用于批处理

### 2. 模型模块 (`src/models/`)

- **teacher.py**: `TeacherWhisper` 类
  - 封装 `WhisperForConditionalGeneration`
  - 提供 `encode()` 方法获取 encoder 输出（无梯度）
  - 提供 `decode()` 方法获取 decoder logits（无梯度）
  - 所有参数冻结，仅用于推理

- **student_decoder.py**: `StudentDecoder` 类
  - 轻量级 Transformer decoder 实现
  - 可配置层数、注意力头数
  - 使用 cross-attention 连接 encoder 输出
  - 支持因果掩码（causal mask）

- **student_asr_model.py**: `StudentASRModel` 类
  - 组合冻结的 teacher encoder 和可训练的 student decoder
  - 支持 CE loss 计算
  - 支持返回 teacher logits（用于 KD）
  - 提供 `generate()` 方法用于推理

### 3. 训练模块 (`src/training/`)

- **train_ce.py**: CE baseline 训练
  - 仅使用 Cross Entropy loss
  - 实现完整的训练循环
  - 支持验证、保存 checkpoint、计算 WER/CER

- **train_kd.py**: KD baseline 训练
  - 使用 CE + KD loss（KL divergence）
  - 支持温度（temperature）和权重（alpha）配置
  - 同时记录 CE 和 KD loss

### 4. 工具模块 (`src/utils/`)

- **config.py**: 配置管理
  - 定义 `DataConfig`, `ModelConfig`, `TrainingConfig`, `KDConfig`
  - 提供 `load_config()` 从 YAML 加载配置

- **logger.py**: 日志工具
  - 设置控制台和文件日志
  - 统一的日志格式

- **metrics.py**: 评估指标
  - `calculate_wer()`: 计算 Word Error Rate
  - `calculate_cer()`: 计算 Character Error Rate
  - 使用 jiwer 库实现

### 5. 配置文件 (`configs/`)

两个主要配置文件，包含：
- 数据配置（数据集、批次大小、采样等）
- 模型配置（teacher 模型、student 结构等）
- 训练配置（学习率、优化器、日志等）
- KD 配置（仅 kd_tiny_s2.yaml，包含温度和 alpha）

### 6. 脚本 (`scripts/`)

- **train_ce.sh**: 运行 CE 训练的便捷脚本
- **train_kd.sh**: 运行 KD 训练的便捷脚本

## 数据流

1. **训练阶段**:
   ```
   音频 → Feature Extractor → Log-mel 特征
                                    ↓
   文本 → Tokenizer → Token IDs → Labels
                                    ↓
                          Teacher Encoder (frozen)
                                    ↓
                          Encoder Hidden States
                                    ↓
                          Student Decoder (trainable)
                                    ↓
                          Logits → CE Loss (+ KD Loss)
   ```

2. **推理阶段**:
   ```
   音频 → Feature Extractor → Teacher Encoder (frozen)
                                    ↓
                          Encoder Hidden States
                                    ↓
                          Student Decoder (greedy decode)
                                    ↓
                          Generated Token IDs → Text
   ```

## 关键特性

### Phase 1 完成的功能

1. ✅ **数据管线**: LibriSpeech 加载、特征提取、批处理
2. ✅ **Teacher 模型**: Whisper-tiny 封装，encoder/decoder 推理
3. ✅ **Student 模型**: 2 层 Transformer decoder，复用 teacher encoder
4. ✅ **CE Baseline**: 完整的训练、验证、评估流程
5. ✅ **KD Baseline**: Hinton KD（温度软化 + KL divergence）
6. ✅ **评估指标**: WER 和 CER 计算
7. ✅ **配置系统**: YAML 配置文件，灵活的超参数管理
8. ✅ **日志系统**: 训练过程完整记录
9. ✅ **模型保存**: Checkpoint 和 best model 保存

### 为 Phase 2 预留的扩展点

- 数据集可轻松切换和扩展
- 模型结构参数化，易于调整
- 训练循环模块化，便于添加新的 loss
- 配置系统支持添加新的参数
- 代码结构清晰，便于集成 DKD 等新方法

## 使用流程

### 1. 环境设置
```bash
pip install -r requirements.txt
```

### 2. 测试设置
```bash
python test_setup.py
```

### 3. 训练 CE baseline
```bash
python -m src.training.train_ce --config configs/ce_tiny_s2.yaml
# 或
bash scripts/train_ce.sh
```

### 4. 训练 KD baseline
```bash
python -m src.training.train_kd --config configs/kd_tiny_s2.yaml
# 或
bash scripts/train_kd.sh
```

### 5. 查看结果
训练输出保存在 `outputs/` 目录下：
- `outputs/ce_tiny_s2/`: CE baseline 结果
- `outputs/kd_tiny_s2/`: KD baseline 结果

每个目录包含：
- `train.log`: 训练日志
- `best_model.pt`: 最佳模型（按 WER）
- `checkpoint_*.pt`: 定期保存的 checkpoint

## 参数量对比

- **Teacher (Whisper-tiny)**: ~39M 参数
  - Encoder: ~27M（冻结）
  - Decoder: ~12M（仅用于 KD）

- **Student Decoder (2-layer)**: ~1.5M 参数
  - 所有参数可训练
  - 压缩比约 8:1

## 预期性能

在 1000 样本的小数据集上：
- 训练时间: 约 30-60 分钟（单 GPU）
- CE baseline: WER ~40-60%（取决于数据）
- KD baseline: WER 应低于 CE（期望提升 2-5%）

注意：这些是小数据集的快速实验结果，不代表最终性能。

