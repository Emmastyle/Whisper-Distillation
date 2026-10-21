# 🔬 KD Temperature 实验完整指南

## 📋 实验概览

**目标**: 测试不同 Temperature 和模型大小对 Knowledge Distillation 效果的影响

**配置矩阵**:
- **模型大小**: S2 (2层), S3 (3层), S4 (4层)
- **Temperature**: T=1.0, T=2.0
- **Alpha**: 0.1 (CE 权重)
- **Learning Rate**: 1e-4

**总配置数**: 6 个

## 📁 文件结构

```
workspace-whisper/
├── configs/
│   ├── small_kd_s2_t1_a01.yaml    # S2, T=1.0, α=0.1
│   ├── small_kd_s2_t2_a01.yaml    # S2, T=2.0, α=0.1
│   ├── small_kd_s3_t1_a01.yaml    # S3, T=1.0, α=0.1
│   ├── small_kd_s3_t2_a01.yaml    # S3, T=2.0, α=0.1
│   ├── small_kd_s4_t1_a01.yaml    # S4, T=1.0, α=0.1
│   └── small_kd_s4_t2_a01.yaml    # S4, T=2.0, α=0.1
│
├── scripts/
│   ├── train_kd_all_configs.sh    # 🔥 批量训练所有配置
│   ├── verify_configs.py          # ✅ 验证配置文件
│   └── list_kd_experiments.sh     # 📊 查看实验状态（如果存在）
│
└── src/
    ├── training/
    │   └── train_kd.py               # KD 训练脚本
    └── models/
        ├── student_asr_model.py      # Student 模型
        ├── teacher.py                # Teacher 模型
        └── student_decoder.py        # Student Decoder
```

## 🚀 快速开始

### 1. 验证配置

```bash
cd workspace-whisper
python scripts/verify_configs.py
```

**预期输出**:
```
✅ 所有配置文件验证通过！

| 模型 | T=1.0 | T=2.0 |
|------|-------|-------|
| S2   | α=0.1 | α=0.1 |
| S3   | α=0.1 | α=0.1 |
| S4   | α=0.1 | α=0.1 |
```

### 2. 开始训练

#### 方式 A: 批量训练所有配置（推荐）

```bash
bash scripts/train_kd_all_configs.sh
```

**特点**:
- ✅ 自动运行所有 6 个配置
- ✅ 显示进度和预计时间
- ✅ 自动保存日志
- ✅ 失败自动记录
- ✅ 完成后显示结果汇总

**预计时间**: 约 15 小时（6个模型 × 2.5小时/模型）

#### 方式 B: 单独训练（快速测试）

```bash
# 训练单个配置
python -m src.training.train_kd --config configs/small_kd_s2_t1_a01.yaml

# 或指定其他配置
python -m src.training.train_kd --config configs/small_kd_s3_t2_a01.yaml
```

### 3. 监控训练

#### 实时查看日志

```bash
# 查看训练日志
tail -f outputs/small_kd_s2_t1/train.log

# 或查看脚本日志
tail -f logs/train_S2-T1_*.log
```

#### 查看关键指标

```bash
# 查看最佳 WER
grep "Best WER" outputs/small_kd_s*/train.log

# 查看训练进度
grep "Epoch.*completed" outputs/small_kd_s2_t1/train.log

# 查看 loss 分解
grep "ce=.*kd=" outputs/small_kd_s2_t1/train.log | tail -20
```

## 📊 配置详情

### 配置参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| **teacher_model** | openai/whisper-small | Teacher 模型 |
| **student_decoder_layers** | 2/3/4 | Student 层数 |
| **student_decoder_heads** | 12 | 注意力头数 |
| **student_hidden_size** | 768 | 隐藏层大小 |
| **temperature** | 1.0/2.0 | KD 温度 |
| **alpha** | 0.1 | CE loss 权重 |
| **learning_rate** | 1e-4 | 学习率 |
| **batch_size** | 16 | 批次大小 |
| **epochs** | 15 | 训练轮数 |

### Loss 计算公式

```python
# Combined loss
loss = alpha * CE_loss + (1 - alpha) * KD_loss

# 对于 alpha=0.1:
loss = 0.1 * CE_loss + 0.9 * KD_loss

# KD loss (KL divergence)
KD_loss = KL(softmax(student_logits/T), softmax(teacher_logits/T)) * T^2
```

### 实验矩阵

| 配置 | 模型 | Temp | Alpha | LR | 输出目录 | 预计时间 |
|------|------|------|-------|----|---------|---------| 
| 1 | S2 | 1.0 | 0.1 | 1e-4 | `outputs/small_kd_s2_t1` | 2h |
| 2 | S2 | 2.0 | 0.1 | 1e-4 | `outputs/small_kd_s2_t2` | 2h |
| 3 | S3 | 1.0 | 0.1 | 1e-4 | `outputs/small_kd_s3_t1` | 2.5h |
| 4 | S3 | 2.0 | 0.1 | 1e-4 | `outputs/small_kd_s3_t2` | 2.5h |
| 5 | S4 | 1.0 | 0.1 | 1e-4 | `outputs/small_kd_s4_t1` | 3h |
| 6 | S4 | 2.0 | 0.1 | 1e-4 | `outputs/small_kd_s4_t2` | 3h |

## 🔍 代码验证清单

### ✅ 已验证的功能

1. **配置文件** ✅
   - 所有 6 个配置文件格式正确
   - Temperature、Alpha、LR 参数正确
   - 输出目录路径正确

2. **训练代码** ✅
   - KD loss 计算正确（KL divergence + T^2 scaling）
   - Combined loss 正确（alpha * CE + (1-alpha) * KD）
   - Teacher logits 正确获取（通过 `return_teacher_logits=True`）
   - Gradient 只更新 student decoder

3. **模型代码** ✅
   - Teacher encoder 正确冻结
   - Student decoder 正确初始化
   - Forward pass 正确实现
   - Generation 正确实现

4. **数据处理** ✅
   - Decoder input IDs 正确准备（shift right + BOS）
   - Labels 正确处理（padding = -100）
   - Mask 正确应用（忽略 padding）

### 关键代码片段

#### 1. KD Loss 计算

```python
def compute_kd_loss(student_logits, teacher_logits, temperature, labels):
    # Create mask for valid positions
    mask = (labels != -100).float()
    
    # Apply temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Compute KL divergence
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='none')
    kl_div = kl_div.sum(dim=-1)  # Sum over vocab
    
    # Apply mask and average
    kl_div = (kl_div * mask).sum() / mask.sum()
    
    # Scale by T^2
    return kl_div * (temperature ** 2)
```

#### 2. Combined Loss

```python
# In training loop
ce_loss, student_logits, teacher_logits = model(
    input_features=input_features,
    decoder_input_ids=decoder_input_ids,
    labels=labels,
    return_teacher_logits=True,
)

kd_loss = compute_kd_loss(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    temperature=config.kd.temperature,
    labels=labels,
)

# Combined: alpha * CE + (1 - alpha) * KD
loss = config.kd.alpha * ce_loss + (1 - config.kd.alpha) * kd_loss
```

#### 3. Teacher Logits 获取

```python
# In StudentASRModel.forward()
if return_teacher_logits:
    with torch.no_grad():
        teacher_logits = self.teacher.decode(encoder_outputs, decoder_input_ids)
    return loss, student_logits, teacher_logits
```

## 📈 预期结果

### Baseline 对比

**CE Baseline** (已完成):
- `small_ce_s2`: 23.29% WER
- `small_ce_s3`: 18.88% WER

**KD 目标**:
- `small_kd_s2_t2`: < 23% WER (期望最佳)
- `small_kd_s3_t2`: < 18% WER (期望最佳)
- `small_kd_s4_t2`: < 15% WER (期望最佳)

### Temperature 效果预测

根据 KD 文献：
1. **T=1.0**: 可能过于 sharp，效果接近 CE
2. **T=2.0**: 标准配置，期望最佳效果 ⭐
3. **Alpha=0.1**: 强调 KD loss（90%），期望学到更多 teacher 知识

## 🐛 故障排查

### 问题 1: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 方法 1: 降低 batch size
# 编辑配置文件，修改 batch_size: 16 → 8

# 方法 2: 使用梯度累积
# 在训练代码中添加 gradient_accumulation_steps
```

### 问题 2: 训练不稳定

**症状**: Loss 剧烈波动，WER 不下降

**可能原因**:
1. Temperature 过高
2. Learning rate 过高
3. Alpha 设置不合理

**解决方案**:
```bash
# 尝试降低 learning rate
# 或使用 T=1.0 的配置
```

### 问题 3: KD loss 为 NaN

**症状**: 训练日志显示 `kd=nan`

**可能原因**:
1. Temperature 过低导致数值不稳定
2. Logits 溢出

**解决方案**:
```python
# 在 compute_kd_loss 中添加数值稳定性检查
# 使用 torch.clamp 限制 logits 范围
```

### 问题 4: 配置文件找不到

**症状**: `FileNotFoundError: configs/small_kd_*.yaml`

**解决方案**:
```bash
# 确保在项目根目录运行
cd workspace-whisper

# 检查配置文件
ls -l configs/small_kd_s*_t*_a01.yaml

# 重新运行验证
python scripts/verify_configs.py
```

## 📝 实验记录模板

训练完成后，记录以下信息：

```markdown
### 实验结果 - [日期]

| 配置 | WER | CER | 训练时间 | 备注 |
|------|-----|-----|---------|------|
| S2-T1 | X.XX% | X.XX% | Xh | ... |
| S2-T2 | X.XX% | X.XX% | Xh | ... |
| S3-T1 | X.XX% | X.XX% | Xh | ... |
| S3-T2 | X.XX% | X.XX% | Xh | ... |
| S4-T1 | X.XX% | X.XX% | Xh | ... |
| S4-T2 | X.XX% | X.XX% | Xh | ... |

**最佳配置**: S?-T?
**最佳 WER**: X.XX%

**观察**:
1. Temperature 影响: ...
2. 模型大小影响: ...
3. 与 CE baseline 对比: ...

**结论**:
...
```

## 🎯 下一步

训练完成后：

1. **收集结果**
   ```bash
   grep "Best WER" outputs/small_kd_s*/train.log > results_summary.txt
   ```

2. **分析对比**
   - 对比不同 Temperature
   - 对比不同模型大小
   - 对比 KD vs CE baseline

3. **可选实验**
   - 测试其他 Alpha 值（0.3, 0.5）
   - 测试其他 Temperature（3.0, 4.0）
   - 测试不同 Learning Rate

4. **撰写报告**
   - 总结最佳配置
   - 分析 Temperature 影响
   - 提出改进建议

## 📚 参考资料

### KD 相关论文

1. **Distilling the Knowledge in a Neural Network** (Hinton et al., 2015)
   - 提出 Temperature scaling
   - 推荐 T=2-4

2. **DistilBERT** (Sanh et al., 2019)
   - 使用 α=0.5
   - 在 NLP 任务上验证有效

3. **Whisper 相关**
   - OpenAI Whisper paper
   - Encoder-Decoder 架构

### 代码参考

- `src/training/train_kd.py`: KD 训练主逻辑
- `src/models/student_asr_model.py`: Student 模型实现
- `configs/small_kd_s*_t*_a01.yaml`: 配置文件示例

---

**准备就绪！** 🚀

运行以下命令开始实验：

```bash
cd workspace-whisper
python scripts/verify_configs.py  # 验证配置
bash scripts/train_kd_all_configs.sh  # 开始训练
```

祝实验顺利！ 🎉

