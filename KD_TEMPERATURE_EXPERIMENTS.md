# KD Temperature 实验说明

## 🎯 实验目标

针对之前发现的 KD 训练效果不如 CE 的问题，进行系统的温度（Temperature）和学习率实验。

## 📊 问题背景

之前的实验结果：
- **small_ce_s2**: 23.29% WER ✅
- **small_kd_s2** (T=4, α=0.5): 25.95% WER ❌
- **small_ce_s3**: 18.88% WER ✅

**主要问题**：
1. Temperature=4.0 过高，导致 teacher 分布过于平滑
2. KD loss 不收敛（从 0.86 → 0.68，下降缓慢）
3. 训练不稳定，WER 剧烈波动（43% → 61% → 104%）

## 🔬 实验设计

### 实验变量

| 变量 | 取值 | 说明 |
|------|------|------|
| **Temperature (T)** | 1.0, 2.0 | 控制 softmax 平滑程度 |
| **Learning Rate** | 1e-4, 1e-5 | 标准 vs 0.1x |
| **模型层数** | 2, 3, 4 | S2/S3/S4 |
| **Alpha (α)** | 0.5 | CE 和 KD 权重（固定） |

### 实验配置

#### 组 1: Temperature = 1.0 (Sharp Distribution)
```
configs/small_kd_s2_t1.yaml  → outputs/small_kd_s2_t1/
configs/small_kd_s3_t1.yaml  → outputs/small_kd_s3_t1/
configs/small_kd_s4_t1.yaml  → outputs/small_kd_s4_t1/
```

#### 组 2: Temperature = 2.0 (Moderate Distribution)
```
configs/small_kd_s2_t2.yaml  → outputs/small_kd_s2_t2/
configs/small_kd_s3_t2.yaml  → outputs/small_kd_s3_t2/
configs/small_kd_s4_t2.yaml  → outputs/small_kd_s4_t2/
```

#### 组 3: Temperature = 2.0 + Lower LR (0.1x)
```
configs/small_kd_s2_t2_lr01.yaml  → outputs/small_kd_s2_t2_lr01/
configs/small_kd_s3_t2_lr01.yaml  → outputs/small_kd_s3_t2_lr01/
configs/small_kd_s4_t2_lr01.yaml  → outputs/small_kd_s4_t2_lr01/
```

## 🚀 运行实验

### 方式 1: 运行所有实验（批量）

```bash
# 运行所有 9 个实验配置
bash scripts/train_kd_temperature_sweep.sh
```

**预计时间**：每个模型约 2-3 小时，总计 18-27 小时

### 方式 2: 单独运行（推荐用于测试）

```bash
# S2 模型
bash scripts/train_kd_t1_s2.sh  # T=1.0, S2
bash scripts/train_kd_t2_s2.sh  # T=2.0, S2

# S3 模型
bash scripts/train_kd_t1_s3.sh  # T=1.0, S3
bash scripts/train_kd_t2_s3.sh  # T=2.0, S3

# S4 模型
bash scripts/train_kd_t1_s4.sh  # T=1.0, S4
bash scripts/train_kd_t2_s4.sh  # T=2.0, S4
```

### 方式 3: 直接调用 Python

```bash
# 示例
python -m src.training.train_kd --config configs/small_kd_s2_t1.yaml
```

## 📈 预期结果

### Temperature 影响

| Temperature | Softmax 效果 | 预期 KD 效果 |
|-------------|--------------|--------------|
| **T=1.0** | 保持原始分布 | 接近 CE，可能过于 sharp |
| **T=2.0** | 适度平滑 | **最佳** (文献推荐) |
| **T=4.0** | 过度平滑 | 信息损失，效果差 ❌ |

### 学习率影响

- **LR=1e-4**: 标准学习率，可能导致训练不稳定
- **LR=1e-5**: 更保守，训练更稳定但收敛慢

### 预期最佳配置

根据 KD 文献和之前的分析：
1. **T=2.0, LR=1e-4**: 最可能获得最佳性能
2. **T=1.0, LR=1e-4**: 可能接近 CE 效果
3. **T=2.0, LR=1e-5**: 更稳定但可能收敛不充分

## 📊 评估指标

训练完成后，关注以下指标：

1. **最佳 WER** (Best WER)
   - 目标：< 23.29% (超越 CE baseline)

2. **训练稳定性**
   - WER 是否单调下降
   - 是否出现剧烈波动

3. **KD Loss 收敛**
   - KD loss 是否有效下降
   - CE loss vs KD loss 的比例

4. **与 CE 对比**
   ```
   CE Baseline (已完成):
   - small_ce_s2: 23.29% WER
   - small_ce_s3: 18.88% WER
   - small_ce_s4: ? (待训练)
   ```

## 📝 结果记录

### 实验结果表格（待填写）

| 模型 | Temperature | LR | 最佳 WER | 最终 WER | 训练稳定性 | 备注 |
|------|-------------|----|---------|---------|-----------|----- |
| S2 | 1.0 | 1e-4 | ? | ? | ? | |
| S2 | 2.0 | 1e-4 | ? | ? | ? | |
| S2 | 2.0 | 1e-5 | ? | ? | ? | |
| S3 | 1.0 | 1e-4 | ? | ? | ? | |
| S3 | 2.0 | 1e-4 | ? | ? | ? | |
| S3 | 2.0 | 1e-5 | ? | ? | ? | |
| S4 | 1.0 | 1e-4 | ? | ? | ? | |
| S4 | 2.0 | 1e-4 | ? | ? | ? | |
| S4 | 2.0 | 1e-5 | ? | ? | ? | |

### 对比 Baseline

| 模型 | CE Baseline | 最佳 KD | 改进 |
|------|-------------|---------|------|
| S2 | 23.29% | ? | ? |
| S3 | 18.88% | ? | ? |
| S4 | ? | ? | ? |

## 🔍 分析要点

训练完成后，需要分析：

1. **Temperature 的影响**
   - T=1 vs T=2 哪个更好？
   - 是否符合文献预期（T=2 最佳）？

2. **学习率的影响**
   - 降低 LR 是否提高稳定性？
   - 是否影响最终性能？

3. **模型深度的影响**
   - S2/S3/S4 哪个受益最多？
   - 更深的模型是否更适合 KD？

4. **KD vs CE**
   - KD 是否最终超越 CE？
   - 如果没有，原因是什么？

## 📂 文件结构

```
workspace-whisper/
├── configs/
│   ├── small_kd_s2_t1.yaml          # T=1.0, S2
│   ├── small_kd_s2_t2.yaml          # T=2.0, S2
│   ├── small_kd_s2_t2_lr01.yaml     # T=2.0, S2, LR=0.1x
│   ├── small_kd_s3_t1.yaml          # T=1.0, S3
│   ├── small_kd_s3_t2.yaml          # T=2.0, S3
│   ├── small_kd_s3_t2_lr01.yaml     # T=2.0, S3, LR=0.1x
│   ├── small_kd_s4_t1.yaml          # T=1.0, S4
│   ├── small_kd_s4_t2.yaml          # T=2.0, S4
│   └── small_kd_s4_t2_lr01.yaml     # T=2.0, S4, LR=0.1x
├── scripts/
│   ├── train_kd_temperature_sweep.sh  # 批量训练脚本
│   ├── train_kd_t1_s2.sh             # 单独训练脚本
│   ├── train_kd_t2_s2.sh
│   ├── train_kd_t1_s3.sh
│   ├── train_kd_t2_s3.sh
│   ├── train_kd_t1_s4.sh
│   └── train_kd_t2_s4.sh
└── outputs/
    ├── small_kd_s2_t1/
    ├── small_kd_s2_t2/
    ├── small_kd_s2_t2_lr01/
    ├── small_kd_s3_t1/
    ├── small_kd_s3_t2/
    ├── small_kd_s3_t2_lr01/
    ├── small_kd_s4_t1/
    ├── small_kd_s4_t2/
    └── small_kd_s4_t2_lr01/
```

## 🎓 理论背景

### Knowledge Distillation 原理

```python
# Soft targets with temperature
student_probs = softmax(student_logits / T)
teacher_probs = softmax(teacher_logits / T)

# KL divergence loss
kd_loss = KL(student_probs || teacher_probs) * T^2

# Combined loss
loss = α * CE_loss + (1-α) * kd_loss
```

### Temperature 的作用

- **T=1**: 标准 softmax，分布 sharp
  ```
  logits = [10, 5, 2] → probs = [0.993, 0.007, 0.000]
  ```

- **T=2**: 适度平滑，保留"暗知识"
  ```
  logits = [10, 5, 2] → probs = [0.945, 0.052, 0.003]
  ```

- **T=4**: 过度平滑，信息损失
  ```
  logits = [10, 5, 2] → probs = [0.760, 0.210, 0.030]
  ```

### 为什么 T=2 通常最好？

1. **保留信息**: 不会过度平滑，保留 teacher 的知识
2. **稳定训练**: 不会太 sharp，梯度更稳定
3. **经验验证**: 大量文献验证 T=2-3 效果最好

## 📚 参考文献

1. Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
2. 推荐 Temperature: 2-3 for most tasks
3. Alpha: 0.5-0.7 (CE 略占主导)

## ✅ 下一步

1. **运行实验**: `bash scripts/train_kd_temperature_sweep.sh`
2. **监控训练**: 查看 `outputs/*/train.log`
3. **评估结果**: `python scripts/evaluate_all_models.py`
4. **分析对比**: 填写上面的结果表格
5. **撰写报告**: 总结最佳配置和发现

---

**创建时间**: 2025-11-21  
**实验状态**: 待运行  
**预计完成**: 18-27 小时后

