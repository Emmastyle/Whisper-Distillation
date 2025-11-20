# Whisper Knowledge Distillation - Phase 2 详细实施计划

**Phase 2: 标准 KD 完整版（Standard KD Complete Version）**

---

## 概述

Phase 2 在 Phase 1 的基础上进行规模扩展，使用更大的 teacher 模型和更多数据，完善 Classical Knowledge Distillation 实验。这个阶段**不涉及 DKD**，专注于建立完整的 Classical KD baseline。

### Phase 2 与 Phase 1 的对比

| 维度 | Phase 1 (Mini) | Phase 2 (Full) |
|------|---------------|---------------|
| **Teacher** | whisper-tiny (39M) | whisper-small (244M) |
| **数据量** | 1000 samples (~1小时) | 50-100 小时 |
| **Student** | S2 (2层) | S2/S3/S4 (2/3/4层) |
| **训练模式** | CE + KD 验证 | CE + KD 完整实验 |
| **温度实验** | 简单尝试 (T=1,2,4) | 系统分析 (找最优 T) |
| **目标** | 验证可行性 | 产出课程成果 |

### Phase 2 的核心目标

1. **放大规模**: Teacher 升级到 small/base，数据扩展到 50-100 小时
2. **多种 Student**: S2/S3/S4 三种深度的 decoder
3. **完整对比**: CE-only vs Classical KD 全面对比
4. **温度优化**: 找到最优温度 T
5. **性能分析**: 参数量、FLOPs、RTF、WER/CER
6. **速度-准确率曲线**: 不同 decoder 深度的权衡分析

---

## Step 1: Teacher 模型升级

### 1.1 目标

从 whisper-tiny 升级到 whisper-small (或 whisper-base)，获得更强的 teacher logits 质量。

### 1.2 Teacher 模型对比

| 模型 | 参数量 | Encoder | Decoder | Hidden Size | Layers |
|------|--------|---------|---------|-------------|--------|
| whisper-tiny | 39M | 27M | 12M | 384 | 4+4 |
| whisper-small | 244M | 167M | 77M | 768 | 12+12 |
| whisper-base | 74M | 51M | 23M | 512 | 6+6 |

**推荐**: 优先使用 **whisper-small**，它在性能和效率上平衡最好。

### 1.3 实施任务

#### Task 1.1: 创建 Small Teacher 配置

创建 `configs/teacher_small.yaml`:

```yaml
model:
  teacher_model: "openai/whisper-small"
  # Model info
  # - Total params: 244M
  # - Encoder params: ~167M (12 layers, hidden 768)
  # - Decoder params: ~77M (12 layers, hidden 768)
  # - Vocab size: 51865
  # - Hidden size: 768
```

#### Task 1.2: 测试 Teacher 性能

创建 `scripts/test_teacher_small.py`:

```python
"""
测试 whisper-small teacher 的性能
用途：
1. 验证模型可以正常加载
2. 测量推理速度 (RTF)
3. 在验证集上测试 WER/CER（作为性能上限）
4. 测量显存占用
"""

import torch
from src.models import TeacherWhisper
from src.data import WhisperASRDataset
from src.utils.metrics import calculate_wer, calculate_cer
import time

def test_teacher():
    # Load teacher
    teacher = TeacherWhisper("openai/whisper-small")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device)
    
    # Load small validation set
    val_dataset = WhisperASRDataset(
        split="validation",
        num_samples=100
    )
    
    # Test inference
    # ... (WER, RTF 测试代码)
    
if __name__ == "__main__":
    test_teacher()
```

**测量指标**:
- WER / CER (Teacher 上限)
- RTF (Real-Time Factor)
- 显存占用 (GPU memory)
- 推理时间 (per sample)

#### Task 1.3: 更新模型加载代码

在 `src/models/teacher.py` 中确保支持动态选择 teacher:

```python
class TeacherWhisper(nn.Module):
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        super().__init__()
        print(f"Loading teacher model: {model_name}")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        # ... 其余代码不变
        
    def get_model_info(self):
        """返回模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        encoder_params = sum(p.numel() for p in self.model.get_encoder().parameters())
        decoder_params = sum(p.numel() for p in self.model.get_decoder().parameters())
        
        return {
            'total_params': total_params,
            'encoder_params': encoder_params,
            'decoder_params': decoder_params,
            'hidden_size': self.d_model,
            'vocab_size': self.vocab_size,
        }
```

### 1.4 交付物

- [ ] `configs/teacher_small.yaml` - Small teacher 配置
- [ ] `scripts/test_teacher_small.py` - Teacher 性能测试脚本
- [ ] `doc/teacher_comparison.md` - Teacher 性能对比报告
  - Tiny vs Small 的 WER/CER 对比
  - 推理速度对比
  - 显存占用对比

---

## Step 2: 数据集扩展

### 2.1 目标

从 Phase 1 的 1000 样本 (~1小时) 扩展到 50-100 小时的训练数据。

### 2.2 数据方案

**推荐方案**: 使用 LibriSpeech `train-clean-100`

- **总量**: ~100 小时音频
- **样本数**: ~28,000 个音频片段
- **说话人**: 251 人
- **质量**: 高质量朗读语音
- **格式**: 16kHz, 16-bit WAV

**数据划分**:
- 训练集: 90% (~25,200 samples, 90h)
- 验证集: 5% (~1,400 samples, 5h)
- 测试集: 5% (~1,400 samples, 5h)

### 2.3 实施任务

#### Task 2.1: 创建数据配置

创建 `configs/data_100h.yaml`:

```yaml
data:
  dataset_name: "librispeech_asr"
  dataset_config: "clean"
  split: "train.100"
  
  # 使用全部数据（不限制 num_samples）
  num_samples: null
  
  # 时长限制（保持与 Phase 1 一致）
  max_duration: 10.0  # 秒
  
  # 采样率
  sample_rate: 16000
  
  # 数据划分
  train_ratio: 0.9
  val_ratio: 0.05
  test_ratio: 0.05
  
  # DataLoader 配置
  batch_size: 16  # 增大 batch size（Phase 1 是 8）
  num_workers: 8   # 增加并行加载
```

#### Task 2.2: 数据统计脚本

创建 `scripts/analyze_data.py`:

```python
"""
分析 LibriSpeech train-clean-100 数据集
生成统计报告：
1. 总样本数、总时长
2. 时长分布直方图
3. 文本长度分布
4. 说话人统计
"""

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

def analyze_librispeech():
    # Load dataset
    dataset = load_dataset("librispeech_asr", "clean", split="train.100")
    
    # Filter by duration (≤10s)
    dataset = dataset.filter(
        lambda x: len(x['audio']['array']) / x['audio']['sampling_rate'] <= 10.0
    )
    
    # Statistics
    print(f"Total samples: {len(dataset)}")
    
    durations = []
    text_lengths = []
    
    for item in dataset:
        duration = len(item['audio']['array']) / item['audio']['sampling_rate']
        durations.append(duration)
        text_lengths.append(len(item['text']))
    
    print(f"Total duration: {sum(durations)/3600:.2f} hours")
    print(f"Mean duration: {np.mean(durations):.2f}s")
    print(f"Mean text length: {np.mean(text_lengths):.1f} chars")
    
    # Plot histograms
    # ...

if __name__ == "__main__":
    analyze_librispeech()
```

#### Task 2.3: 更新 Dataset 类

在 `src/data/dataset.py` 中添加数据划分功能:

```python
def create_data_splits(dataset, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, seed=42):
    """
    划分数据集为 train/val/test
    
    Args:
        dataset: HuggingFace Dataset
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from datasets import DatasetDict
    
    # Split dataset
    splits = dataset.train_test_split(test_size=1-train_ratio, seed=seed)
    train_dataset = splits['train']
    
    # Further split test into val and test
    val_test_splits = splits['test'].train_test_split(
        test_size=test_ratio/(val_ratio+test_ratio),
        seed=seed
    )
    val_dataset = val_test_splits['train']
    test_dataset = val_test_splits['test']
    
    return train_dataset, val_dataset, test_dataset
```

### 2.4 交付物

- [ ] `configs/data_100h.yaml` - 100h 数据配置
- [ ] `scripts/analyze_data.py` - 数据分析脚本
- [ ] `doc/data_statistics.md` - 数据统计报告
  - 样本数、总时长
  - 时长分布图
  - 文本长度分布图

---

## Step 3: 多种 Student Decoder (S2/S3/S4)

### 3.1 目标

实现三种不同深度的 student decoder，用于 speed-accuracy tradeoff 分析。

### 3.2 Student 配置设计

| Student | Layers | Hidden | Heads | FFN Dim | 参数量 | 用途 |
|---------|--------|--------|-------|---------|--------|------|
| **S2** | 2 | 768 | 12 | 3072 | ~3M | 最快，最小 |
| **S3** | 3 | 768 | 12 | 3072 | ~4.5M | 平衡 |
| **S4** | 4 | 768 | 12 | 3072 | ~6M | 更强性能 |

**注意**: 
- Hidden size 与 whisper-small 保持一致 (768)
- Heads 数量与 whisper-small 保持一致 (12)
- 只改变 decoder 层数

### 3.3 实施任务

#### Task 3.1: 验证 StudentDecoder 实现

确认 `src/models/student_decoder.py` 支持可变层数:

```python
class StudentDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 2,  # 可配置！
        n_heads: int = 12,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_length: int = 448,
    ):
        super().__init__()
        # ... 实现已经支持可变 n_layers
```

#### Task 3.2: 创建三个配置文件

**configs/small_ce_s2.yaml** (CE-only baseline):
```yaml
model:
  teacher_model: "openai/whisper-small"
  student_decoder_layers: 2
  student_decoder_heads: 12
  student_hidden_size: 768
  dropout: 0.1

data:
  dataset_name: "librispeech_asr"
  dataset_config: "clean"
  split: "train.100"
  num_samples: null  # 使用全部数据
  max_duration: 10.0
  batch_size: 16
  num_workers: 8
  sample_rate: 16000

training:
  epochs: 15
  learning_rate: 1.0e-4
  weight_decay: 0.01
  warmup_steps: 500
  gradient_clip: 1.0
  log_interval: 100
  eval_interval: 1000
  save_interval: 2000
  output_dir: "./outputs/small_ce_s2"
  seed: 42
  fp16: true
```

**configs/small_kd_s2.yaml** (KD with S2):
```yaml
# 继承 small_ce_s2.yaml 的所有配置
# 额外添加 KD 参数

model:
  teacher_model: "openai/whisper-small"
  student_decoder_layers: 2
  student_decoder_heads: 12
  student_hidden_size: 768
  dropout: 0.1

data:
  # ... 同上

training:
  # ... 同上
  output_dir: "./outputs/small_kd_s2"

kd:
  temperature: 4.0  # 后续会优化这个值
  alpha: 0.5        # CE 和 KD 的权重
```

类似地创建:
- `configs/small_kd_s3.yaml` (student_decoder_layers: 3)
- `configs/small_kd_s4.yaml` (student_decoder_layers: 4)

#### Task 3.3: 参数量和 FLOPs 计算

创建 `src/utils/model_analysis.py`:

```python
"""
模型分析工具：计算参数量、FLOPs、RTF
"""

import torch
import torch.nn as nn
from typing import Dict
import time

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数量
    
    Returns:
        {
            'total': 总参数量,
            'trainable': 可训练参数量,
            'frozen': 冻结参数量
        }
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen
    }

def estimate_decoder_flops(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
    n_heads: int,
    d_ff: int = None
) -> int:
    """
    估算 decoder 的 FLOPs（一次前向传播）
    
    主要计算：
    1. Self-attention: O(seq_len^2 * hidden_size)
    2. Cross-attention: O(seq_len * enc_len * hidden_size)
    3. FFN: O(seq_len * hidden_size * d_ff)
    4. Output projection: O(seq_len * hidden_size * vocab_size)
    """
    if d_ff is None:
        d_ff = 4 * hidden_size
    
    enc_len = 1500  # Whisper encoder 输出长度（约）
    
    # Per layer
    self_attn_flops = 2 * batch_size * seq_len * seq_len * hidden_size
    cross_attn_flops = 2 * batch_size * seq_len * enc_len * hidden_size
    ffn_flops = 2 * batch_size * seq_len * hidden_size * d_ff
    
    layer_flops = self_attn_flops + cross_attn_flops + ffn_flops
    
    # All layers
    total_layer_flops = n_layers * layer_flops
    
    # Output projection
    output_flops = 2 * batch_size * seq_len * hidden_size * vocab_size
    
    total_flops = total_layer_flops + output_flops
    
    return total_flops

def measure_rtf(model, dataloader, device, num_samples=100):
    """
    测量 Real-Time Factor
    RTF = 处理时间 / 音频时长
    RTF < 1 表示可以实时处理
    """
    model.eval()
    total_audio_duration = 0
    total_process_time = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            input_features = batch['input_features'].to(device)
            
            # 计算音频时长（假设 16kHz, 30s max）
            audio_duration = input_features.shape[0] * 10.0  # batch_size * 10s
            
            # 测量处理时间
            start = time.time()
            _ = model.generate(input_features)
            end = time.time()
            
            total_audio_duration += audio_duration
            total_process_time += (end - start)
    
    rtf = total_process_time / total_audio_duration
    return rtf
```

### 3.4 交付物

- [ ] `configs/small_ce_s2.yaml` - S2 CE-only 配置
- [ ] `configs/small_kd_s2.yaml` - S2 KD 配置
- [ ] `configs/small_kd_s3.yaml` - S3 KD 配置
- [ ] `configs/small_kd_s4.yaml` - S4 KD 配置
- [ ] `src/utils/model_analysis.py` - 模型分析工具

---

## Step 4: Classical KD 完整训练

### 4.1 目标

在 S2/S3/S4 上分别训练 CE-only 和 Classical KD，收集完整的性能数据。

### 4.2 实验矩阵

| 实验 | Teacher | Student | 数据 | Loss | 配置文件 |
|------|---------|---------|------|------|---------|
| 1 | small | S2 | 100h | CE | small_ce_s2.yaml |
| 2 | small | S2 | 100h | KD | small_kd_s2.yaml |
| 3 | small | S3 | 100h | KD | small_kd_s3.yaml |
| 4 | small | S4 | 100h | KD | small_kd_s4.yaml |

**说明**:
- 实验 1: CE-only baseline (S2)
- 实验 2-4: Classical KD 在不同深度上

### 4.3 实施任务

#### Task 4.1: 批量训练脚本

创建 `scripts/train_phase2_all.sh`:

```bash
#!/bin/bash
# Phase 2: 批量训练所有实验

set -e  # 出错即停止

echo "========================================="
echo "Phase 2: Standard KD Complete Training"
echo "========================================="

# 1. CE-only baseline (S2)
echo "[1/4] Training CE-only baseline (S2)..."
python -m src.training.train_ce \
    --config configs/small_ce_s2.yaml

# 2. KD with S2
echo "[2/4] Training KD with S2..."
python -m src.training.train_kd \
    --config configs/small_kd_s2.yaml

# 3. KD with S3
echo "[3/4] Training KD with S3..."
python -m src.training.train_kd \
    --config configs/small_kd_s3.yaml

# 4. KD with S4
echo "[4/4] Training KD with S4..."
python -m src.training.train_kd \
    --config configs/small_kd_s4.yaml

echo "========================================="
echo "Phase 2 training completed!"
echo "========================================="
```

#### Task 4.2: 单独训练脚本（方便调试）

创建 `scripts/train_ce_s2.sh`:
```bash
#!/bin/bash
python -m src.training.train_ce --config configs/small_ce_s2.yaml
```

创建 `scripts/train_kd_s2.sh`, `train_kd_s3.sh`, `train_kd_s4.sh` 类似。

#### Task 4.3: 训练监控

在训练过程中记录：
- 训练 loss (CE / KD)
- 验证 WER / CER
- 学习率变化
- 每个 epoch 的时间

这些都已在 `src/training/train_ce.py` 和 `train_kd.py` 中实现。

### 4.4 交付物

- [ ] `scripts/train_phase2_all.sh` - 批量训练脚本
- [ ] `scripts/train_ce_s2.sh` - S2 CE-only 训练
- [ ] `scripts/train_kd_s2.sh` - S2 KD 训练
- [ ] `scripts/train_kd_s3.sh` - S3 KD 训练
- [ ] `scripts/train_kd_s4.sh` - S4 KD 训练
- [ ] 训练日志 (自动生成在 `outputs/*/train.log`)

---

## Step 5: 温度 T 优化实验

### 5.1 目标

系统地寻找 Classical KD 的最优温度，对应课程要求的 "Try different temperature T and analyze influence"。

### 5.2 实验设计

**固定设置**:
- Teacher: whisper-small
- Student: S3 (中等规模，代表性强)
- 数据: 100h
- Loss: Classical KD

**变量**:
- Temperature T ∈ {1, 2, 4, 6, 8}

**观察指标**:
- 验证集 WER / CER
- KD loss 收敛情况
- 训练稳定性

### 5.3 实施任务

#### Task 5.1: 创建温度扫描配置

创建配置文件:
- `configs/temperature/kd_s3_t1.yaml` (temperature: 1.0)
- `configs/temperature/kd_s3_t2.yaml` (temperature: 2.0)
- `configs/temperature/kd_s3_t4.yaml` (temperature: 4.0)
- `configs/temperature/kd_s3_t6.yaml` (temperature: 6.0)
- `configs/temperature/kd_s3_t8.yaml` (temperature: 8.0)

配置模板:
```yaml
model:
  teacher_model: "openai/whisper-small"
  student_decoder_layers: 3
  student_decoder_heads: 12
  student_hidden_size: 768
  dropout: 0.1

data:
  # ... 同 Step 3

training:
  # ... 同 Step 3
  output_dir: "./outputs/temperature/kd_s3_t{T}"

kd:
  temperature: {T}  # 1, 2, 4, 6, 8
  alpha: 0.5
```

#### Task 5.2: 温度扫描脚本

创建 `scripts/run_temperature_sweep.sh`:

```bash
#!/bin/bash
# Temperature sweep for Classical KD

temperatures=(1 2 4 6 8)

for T in "${temperatures[@]}"; do
    echo "========================================="
    echo "Training KD with Temperature T=$T"
    echo "========================================="
    
    python -m src.training.train_kd \
        --config configs/temperature/kd_s3_t${T}.yaml
done

echo "Temperature sweep completed!"
```

#### Task 5.3: 温度分析脚本

创建 `scripts/analyze_temperature.py`:

```python
"""
分析温度实验结果
生成：
1. T vs WER 曲线
2. T vs loss 曲线
3. 最优温度推荐
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_results(output_dir, temperatures):
    """从各个实验目录加载结果"""
    results = {}
    
    for T in temperatures:
        exp_dir = f"{output_dir}/temperature/kd_s3_t{T}"
        
        # 读取最佳模型的 WER
        best_model_path = f"{exp_dir}/best_model.pt"
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            results[T] = {
                'wer': checkpoint.get('wer', None),
                'cer': checkpoint.get('cer', None),
                'epoch': checkpoint.get('epoch', None),
            }
    
    return results

def plot_temperature_analysis(results):
    """绘制温度分析图"""
    temperatures = sorted(results.keys())
    wers = [results[T]['wer'] for T in temperatures]
    cers = [results[T]['cer'] for T in temperatures]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # WER vs T
    ax1.plot(temperatures, wers, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Temperature T', fontsize=12)
    ax1.set_ylabel('WER (%)', fontsize=12)
    ax1.set_title('WER vs Temperature', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # CER vs T
    ax2.plot(temperatures, cers, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Temperature T', fontsize=12)
    ax2.set_ylabel('CER (%)', fontsize=12)
    ax2.set_title('CER vs Temperature', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/temperature_analysis.pdf')
    plt.savefig('results/temperature_analysis.png', dpi=300)
    
    # 找最优温度
    best_T = temperatures[np.argmin(wers)]
    print(f"\nBest temperature: T = {best_T}")
    print(f"Best WER: {min(wers):.2f}%")
    print(f"Best CER: {cers[np.argmin(wers)]:.2f}%")

if __name__ == "__main__":
    temperatures = [1, 2, 4, 6, 8]
    results = load_results("./outputs", temperatures)
    plot_temperature_analysis(results)
```

### 5.4 理论分析

在报告中分析温度的作用：

**小温度 (T=1, 2)**:
- Softmax 分布尖锐
- 主要学习 top-1, top-2 token
- 接近 hard labels
- 适合 student 容量小的情况

**中等温度 (T=4, 6)**:
- 分布适度平滑
- 学习 token 之间的相对关系
- 通常是最优区间

**大温度 (T=8+)**:
- 分布过度平滑
- 所有 token 概率接近
- 可能导致信息损失

### 5.5 交付物

- [ ] `configs/temperature/kd_s3_t*.yaml` - 温度配置 (5个)
- [ ] `scripts/run_temperature_sweep.sh` - 温度扫描脚本
- [ ] `scripts/analyze_temperature.py` - 温度分析脚本
- [ ] `results/temperature_analysis.pdf` - 温度分析图表
- [ ] `doc/temperature_analysis.md` - 温度实验报告

---

## Step 6: 性能评估与可视化

### 6.1 目标

全面评估所有模型，生成 speed-accuracy 曲线和对比表格。

### 6.2 评估指标

**准确性指标**:
- WER (Word Error Rate)
- CER (Character Error Rate)

**效率指标**:
- 参数量 (Parameters)
- FLOPs (浮点运算次数)
- RTF (Real-Time Factor)
- 推理时间 (Inference time per sample)

### 6.3 实施任务

#### Task 6.1: 统一评估脚本

创建 `scripts/evaluate_all_models.py`:

```python
"""
评估所有 Phase 2 模型
生成完整的对比表格和图表
"""

import torch
from pathlib import Path
from src.models import TeacherWhisper, StudentASRModel
from src.data import WhisperASRDataset
from src.utils.model_analysis import count_parameters, estimate_decoder_flops, measure_rtf
from src.utils.metrics import calculate_wer, calculate_cer
from transformers import WhisperTokenizer
import pandas as pd

def evaluate_model(model_path, config, test_loader, tokenizer, device):
    """评估单个模型"""
    # Load model
    checkpoint = torch.load(model_path)
    
    # ... 评估代码
    
    return {
        'wer': wer,
        'cer': cer,
        'rtf': rtf,
        # ...
    }

def evaluate_all():
    """评估所有模型"""
    models = [
        {'name': 'CE-S2', 'path': 'outputs/small_ce_s2/best_model.pt', 'layers': 2},
        {'name': 'KD-S2', 'path': 'outputs/small_kd_s2/best_model.pt', 'layers': 2},
        {'name': 'KD-S3', 'path': 'outputs/small_kd_s3/best_model.pt', 'layers': 3},
        {'name': 'KD-S4', 'path': 'outputs/small_kd_s4/best_model.pt', 'layers': 4},
    ]
    
    results = []
    for model_info in models:
        print(f"Evaluating {model_info['name']}...")
        result = evaluate_model(model_info['path'], ...)
        result['name'] = model_info['name']
        result['layers'] = model_info['layers']
        results.append(result)
    
    # 生成对比表格
    df = pd.DataFrame(results)
    df.to_csv('results/model_comparison.csv')
    df.to_latex('results/model_comparison.tex')
    
    return df

if __name__ == "__main__":
    df = evaluate_all()
    print(df)
```

#### Task 6.2: 可视化脚本

创建 `scripts/plot_results.py`:

```python
"""
生成所有可视化图表：
1. Speed-Accuracy Curve (WER vs Params, WER vs RTF)
2. CE vs KD Comparison
3. Training Curves
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_speed_accuracy_curve(df):
    """绘制速度-准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # WER vs Parameters
    ax1.scatter(df['params']/1e6, df['wer'], s=150, alpha=0.7)
    for _, row in df.iterrows():
        ax1.annotate(row['name'], 
                     (row['params']/1e6, row['wer']),
                     xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('Parameters (M)', fontsize=12)
    ax1.set_ylabel('WER (%)', fontsize=12)
    ax1.set_title('WER vs Model Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # WER vs RTF
    ax2.scatter(df['rtf'], df['wer'], s=150, alpha=0.7, color='orange')
    for _, row in df.iterrows():
        ax2.annotate(row['name'], 
                     (row['rtf'], row['wer']),
                     xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Real-Time Factor', fontsize=12)
    ax2.set_ylabel('WER (%)', fontsize=12)
    ax2.set_title('WER vs Inference Speed', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/speed_accuracy_curve.pdf')
    plt.savefig('results/speed_accuracy_curve.png', dpi=300)

def plot_ce_vs_kd():
    """对比 CE-only vs KD (S2)"""
    # 从日志文件读取训练曲线
    # 绘制 loss 和 WER 随 epoch 变化
    pass

if __name__ == "__main__":
    df = pd.read_csv('results/model_comparison.csv')
    plot_speed_accuracy_curve(df)
    plot_ce_vs_kd()
```

#### Task 6.3: 生成最终报告

创建 `scripts/generate_phase2_report.py`:

```python
"""
自动生成 Phase 2 报告
包含所有表格和图表
"""

def generate_report():
    """生成 Markdown 报告"""
    
    report = """
# Phase 2: Standard KD Complete Version - Results

## 模型对比

{model_table}

## Speed-Accuracy Tradeoff

![Speed-Accuracy Curve](../results/speed_accuracy_curve.png)

## CE vs KD 对比

在 S2 上，Classical KD 相比 CE-only 的提升：
- WER: {wer_improvement}%
- CER: {cer_improvement}%

## 温度实验结果

最优温度: T = {best_temperature}

![Temperature Analysis](../results/temperature_analysis.png)

## 结论

1. Classical KD 显著优于 CE-only
2. Decoder 深度增加带来性能提升，但速度下降
3. 最优温度为 T={best_temperature}
4. S3 是性能和速度的最佳平衡点
"""
    
    # 填充数据
    # ...
    
    with open('doc/phase2_results.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    generate_report()
```

### 6.4 预期结果

**模型对比表**:

| 模型 | Layers | Params | FLOPs | RTF | WER (%) | CER (%) |
|------|--------|--------|-------|-----|---------|---------|
| Teacher (small) | 12 | 244M | - | 0.3 | ~5.0 | ~2.0 |
| CE-S2 | 2 | 3M | 8G | 0.08 | 12.0 | 6.0 |
| KD-S2 | 2 | 3M | 8G | 0.08 | 10.5 | 5.2 |
| KD-S3 | 3 | 4.5M | 12G | 0.12 | 9.0 | 4.5 |
| KD-S4 | 4 | 6M | 16G | 0.15 | 8.0 | 4.0 |

**关键发现**:
1. KD 相比 CE 提升约 1.5% WER
2. Decoder 层数增加带来性能提升
3. S3 是最佳平衡点 (性能 vs 速度)
4. 最优温度约为 T=4

### 6.5 交付物

- [ ] `scripts/evaluate_all_models.py` - 统一评估脚本
- [ ] `scripts/plot_results.py` - 可视化脚本
- [ ] `scripts/generate_phase2_report.py` - 报告生成脚本
- [ ] `results/model_comparison.csv` - 对比表格 (CSV)
- [ ] `results/model_comparison.tex` - 对比表格 (LaTeX)
- [ ] `results/speed_accuracy_curve.pdf` - 速度-准确率曲线
- [ ] `results/temperature_analysis.pdf` - 温度分析图
- [ ] `doc/phase2_results.md` - Phase 2 结果报告

---

## Phase 2 时间线

### Week 4: 基础设施扩展

**Day 1-2: Teacher 升级**
- [ ] 创建 small teacher 配置
- [ ] 测试 teacher 性能
- [ ] 记录 teacher baseline

**Day 3-4: 数据扩展**
- [ ] 加载 LibriSpeech train-clean-100
- [ ] 数据统计分析
- [ ] 验证数据质量

**Day 5-7: Student 配置与测试**
- [ ] 创建 S2/S3/S4 配置
- [ ] 测试参数量和 FLOPs
- [ ] 准备训练脚本

### Week 5: Classical KD 训练

**Day 1-3: CE 和 KD baseline**
- [ ] 训练 CE-S2 (baseline)
- [ ] 训练 KD-S2
- [ ] 初步验证 KD 提升

**Day 4-6: S3 和 S4 训练**
- [ ] 训练 KD-S3
- [ ] 训练 KD-S4
- [ ] 收集所有结果

**Day 7: 温度实验启动**
- [ ] 配置温度扫描实验
- [ ] 开始运行 (可能跨到 Week 6)

### Week 6: 评估与报告

**Day 1-2: 温度实验完成**
- [ ] 完成所有温度实验
- [ ] 分析最优温度

**Day 3-4: 性能评估**
- [ ] 评估所有模型
- [ ] 测量 RTF 和 FLOPs
- [ ] 生成对比表格

**Day 5-7: 可视化与报告**
- [ ] 生成所有图表
- [ ] 撰写 Phase 2 报告
- [ ] 整理代码和文档

---

## 文件组织

### 新增配置文件

```
configs/
├── teacher_small.yaml              # Small teacher 配置
├── data_100h.yaml                  # 100h 数据配置
├── small_ce_s2.yaml               # S2 CE-only
├── small_kd_s2.yaml               # S2 KD
├── small_kd_s3.yaml               # S3 KD
├── small_kd_s4.yaml               # S4 KD
└── temperature/                    # 温度实验
    ├── kd_s3_t1.yaml
    ├── kd_s3_t2.yaml
    ├── kd_s3_t4.yaml
    ├── kd_s3_t6.yaml
    └── kd_s3_t8.yaml
```

### 新增脚本

```
scripts/
├── test_teacher_small.py          # Teacher 测试
├── analyze_data.py                # 数据分析
├── train_phase2_all.sh           # 批量训练
├── train_ce_s2.sh                # 单独训练脚本
├── train_kd_s2.sh
├── train_kd_s3.sh
├── train_kd_s4.sh
├── run_temperature_sweep.sh       # 温度扫描
├── analyze_temperature.py         # 温度分析
├── evaluate_all_models.py         # 统一评估
├── plot_results.py               # 可视化
└── generate_phase2_report.py      # 报告生成
```

### 新增源代码

```
src/utils/
└── model_analysis.py              # 模型分析工具
```

### 结果文件

```
results/
├── model_comparison.csv           # 模型对比表
├── model_comparison.tex           # LaTeX 表格
├── speed_accuracy_curve.pdf       # 速度-准确率曲线
├── speed_accuracy_curve.png
├── temperature_analysis.pdf       # 温度分析图
└── temperature_analysis.png
```

### 文档

```
doc/
├── teacher_comparison.md          # Teacher 对比
├── data_statistics.md            # 数据统计
├── temperature_analysis.md        # 温度实验
└── phase2_results.md             # Phase 2 总结
```

---

## 课程要求对照

| 课程要求 | Phase 2 对应 | 文件 |
|----------|-------------|------|
| Student 使用 soft labels | Classical KD | `train_kd.py` |
| 不同温度 T 的影响 | 温度扫描实验 | Step 5 |
| 不同架构对比 | S2/S3/S4 对比 | Step 3-4 |
| KD vs No-KD 对比 | CE vs KD | Step 4 |
| 完整实验报告 | Phase 2 报告 | Step 6 |

---

## 成功标准

### 必须达到

1. **基础功能**:
   - [ ] 所有模型训练成功
   - [ ] WER/CER 正确计算
   - [ ] 无代码错误

2. **性能指标**:
   - [ ] KD 优于 CE (WER 提升 ≥1%)
   - [ ] S3/S4 优于 S2
   - [ ] 找到最优温度 T

3. **完整性**:
   - [ ] 4 个主要实验完成
   - [ ] 5 个温度实验完成
   - [ ] 所有图表生成
   - [ ] 报告撰写完成

### 加分项

- [ ] 尝试 whisper-base teacher
- [ ] 更细粒度的温度扫描
- [ ] 更多可视化分析
- [ ] 代码质量优秀

---

## 常见问题

### Q1: 100h 数据训练太慢怎么办？

**解决方案**:
1. 先在 10-20h 子集上验证
2. 确认代码无误后再跑全量
3. 使用 FP16 混合精度训练
4. 增大 batch size（如果显存允许）

### Q2: 显存不足怎么办？

**解决方案**:
1. 减小 batch size
2. 使用梯度累积
3. 只训练 S2 和 S3，跳过 S4
4. 减少数据量

### Q3: 温度实验时间不够？

**解决方案**:
1. 只跑 T={1, 2, 4, 8}（减少一个）
2. 减少训练 epochs
3. 在小数据集上快速验证趋势

---

## 总结

Phase 2 完成后，你将拥有：

### 实验成果
- ✅ 4 个主要模型（CE-S2, KD-S2, KD-S3, KD-S4）
- ✅ 5 个温度实验（T=1,2,4,6,8）
- ✅ 完整的性能对比数据

### 分析成果
- ✅ Speed-Accuracy Curve
- ✅ CE vs KD 对比
- ✅ 温度影响分析
- ✅ 模型性能表格

### 文档成果
- ✅ Phase 2 详细报告
- ✅ 所有图表和表格
- ✅ 清晰的实验结论

这些成果**已经足以支撑一个完整的课程项目报告**，满足所有课程要求！

Phase 2 之后，你可以选择：
1. **停在这里**: 完成报告和 presentation
2. **继续 Phase 3**: 实现 DKD，追求更高的创新性

无论哪种选择，Phase 2 都是一个坚实的基础！🚀
