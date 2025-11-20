# 快速开始指南

本指南帮助您快速运行 Phase 1 的 CE 和 KD 基线实验。

## 前置要求

- Python 3.10+
- CUDA-capable GPU（推荐）
- 至少 16GB GPU 显存

## 步骤 1：安装依赖

```bash
cd /home/zz359/workspace-whisper
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch & torchaudio
- HuggingFace transformers & datasets
- jiwer（用于 WER/CER 计算）

## 步骤 2：验证设置

运行测试脚本确保一切正常：

```bash
python test_setup.py
```

这个脚本会：
1. 检查设备（CPU/GPU）
2. 加载配置文件
3. 测试数据集加载（10 个样本）
4. 加载 teacher 和 student 模型
5. 测试前向传播和生成

如果所有测试通过，您会看到：
```
✓ All tests passed!
```

## 步骤 3：训练 CE Baseline

### 方法 1：使用 Python 模块

```bash
python -m src.training.train_ce --config configs/ce_tiny_s2.yaml
```

### 方法 2：使用脚本

```bash
bash scripts/train_ce.sh
```

### 训练过程

训练会进行以下操作：
- 加载 LibriSpeech 数据集（1000 个样本）
- 训练 10 个 epoch
- 每 20 步打印日志
- 每 200 步进行验证
- 每 500 步保存 checkpoint
- 保存最佳模型（按 WER）

### 输出位置

所有输出保存在 `outputs/ce_tiny_s2/`：
- `train.log`: 完整训练日志
- `best_model.pt`: 最佳模型
- `checkpoint_*.pt`: 定期保存的 checkpoint

### 预期时间

- 单 GPU (A100/V100): 约 30-45 分钟
- 单 GPU (RTX 3090): 约 45-60 分钟

## 步骤 4：训练 KD Baseline

### 方法 1：使用 Python 模块

```bash
python -m src.training.train_kd --config configs/kd_tiny_s2.yaml
```

### 方法 2：使用脚本

```bash
bash scripts/train_kd.sh
```

### KD 配置

默认 KD 参数（在 `configs/kd_tiny_s2.yaml` 中）：
- Temperature (T): 2.0
- Alpha (α): 0.5
- 总 Loss = 0.5 × CE + 0.5 × KD

### 输出位置

所有输出保存在 `outputs/kd_tiny_s2/`

## 步骤 5：查看结果

### 查看训练日志

```bash
# CE baseline
tail -f outputs/ce_tiny_s2/train.log

# KD baseline
tail -f outputs/kd_tiny_s2/train.log
```

### 提取 WER/CER

日志中搜索 "Validation" 或 "Epoch" 查看评估结果：

```bash
grep "Validation" outputs/ce_tiny_s2/train.log
grep "Validation" outputs/kd_tiny_s2/train.log
```

### 预期结果

在 1000 样本的小数据集上：

| 方法 | 预期 WER | 预期 CER |
|------|----------|----------|
| CE-only | 40-60% | 20-40% |
| KD (T=2, α=0.5) | 38-55% | 18-38% |

**注意**: KD 应该比 CE 略好，但在小数据集上提升可能不明显。

## 自定义配置

### 修改数据集大小

编辑 `configs/ce_tiny_s2.yaml` 或 `configs/kd_tiny_s2.yaml`：

```yaml
data:
  num_samples: 1000  # 修改这里
  batch_size: 8      # 根据 GPU 显存调整
```

### 修改模型大小

```yaml
model:
  student_decoder_layers: 2  # 增加层数
  student_decoder_heads: 4   # 调整注意力头数
```

### 修改训练超参数

```yaml
training:
  epochs: 10
  learning_rate: 1.0e-4
  warmup_steps: 100
```

### 修改 KD 参数

仅在 `configs/kd_tiny_s2.yaml` 中：

```yaml
kd:
  temperature: 2.0  # 增大 T 使分布更软
  alpha: 0.5        # 调整 CE/KD 权重
```

## 常见问题

### Q1: CUDA out of memory

**解决方案**:
1. 减小 batch size:
   ```yaml
   data:
     batch_size: 4  # 或更小
   ```
2. 减少样本数:
   ```yaml
   data:
     num_samples: 500
   ```

### Q2: 数据集下载慢

**解决方案**:
- LibriSpeech 会自动从 HuggingFace 下载
- 首次运行需要下载约 6GB 数据
- 后续运行会使用缓存

### Q3: WER 很高

**原因**: 
- 1000 样本太少，模型欠拟合
- 这是快速实验设置

**改进**:
1. 增加样本数:
   ```yaml
   data:
     num_samples: 5000  # 或 null（使用全部）
   ```
2. 增加训练轮数:
   ```yaml
   training:
     epochs: 20
   ```

### Q4: 训练太慢

**解决方案**:
1. 减少样本数（用于快速测试）
2. 减少验证频率:
   ```yaml
   training:
     eval_interval: 1000  # 原来是 200
   ```

## 下一步

完成 Phase 1 后，您可以：

1. **比较结果**: 对比 CE vs KD 的 WER/CER
2. **调整超参数**: 尝试不同的温度、alpha、学习率
3. **扩展数据**: 使用更多训练数据
4. **准备 Phase 2**: 在此基础上实现 DKD 等高级技术

## 进阶使用

### 从 checkpoint 恢复训练

修改训练脚本，加载 checkpoint：

```python
checkpoint = torch.load("outputs/ce_tiny_s2/checkpoint_1000.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### 使用自己的数据

修改 `src/data/dataset.py`，支持自定义数据源。

### 评估已训练模型

创建评估脚本：

```python
from src.models import TeacherWhisper, StudentASRModel
import torch

# 加载模型
checkpoint = torch.load("outputs/ce_tiny_s2/best_model.pt")
teacher = TeacherWhisper("openai/whisper-tiny")
model = StudentASRModel(teacher, ...)
model.load_state_dict(checkpoint['model_state_dict'])

# 推理
# ...
```

## 需要帮助？

1. 检查 `train.log` 文件查看详细错误信息
2. 运行 `python test_setup.py` 诊断问题
3. 查看 `STRUCTURE.md` 了解代码结构
4. 查看 `README.md` 了解项目概览

祝实验顺利！🚀

