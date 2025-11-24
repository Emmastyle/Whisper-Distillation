# 🔬 DKD 实验完整指南

## 📋 实验概览

**目标**: 实现和测试 DKD (Decoupled Knowledge Distillation) 方法

**DKD 方法**:
- 将 KD loss 解耦为两个部分：
  - **TCKD** (Target Class KD): 关注目标类别的知识
  - **NCKD** (Non-target Class KD): 关注非目标类别的分布
- 论文: "Decoupled Knowledge Distillation" (CVPR 2022)
- 链接: https://arxiv.org/abs/2203.08679

**配置矩阵**:
- **模型大小**: S2 (2层), S3 (3层), S4 (4层)
- **Temperature**: T=1.0, T=2.0
- **Alpha**: 0.8 (TCKD 权重)
- **Beta**: 1.2 (NCKD 权重)
- **Gamma**: 0.1 (CE 权重)
- **Learning Rate**: 1e-4

**总配置数**: 6 个

## 🧮 DKD Loss 公式

### 总 Loss

```
Total Loss = γ * CE_loss + (1 - γ) * DKD_loss
```

其中：
- `γ = 0.1`: CE loss 权重 (10%)
- `1 - γ = 0.9`: DKD loss 权重 (90%)

### DKD Loss

```
DKD_loss = α * TCKD + β * NCKD
```

其中：
- `α = 0.8`: TCKD 权重
- `β = 1.2`: NCKD 权重

### TCKD (Target Class KD)

```
TCKD = KL(p_t^T || p_t^S)
```

其中：
- `p_t^T`: Teacher 在目标类上的概率
- `p_t^S`: Student 在目标类上的概率
- 使用二分类 KL 散度 (target vs non-target)

### NCKD (Non-target Class KD)

```
NCKD = KL(P_nt^T || P_nt^S)
```

其中：
- `P_nt^T`: Teacher 在非目标类上的归一化分布
- `P_nt^S`: Student 在非目标类上的归一化分布

### 完整示例

对于 `γ=0.1, α=0.8, β=1.2`:

```
Loss = 0.1 * CE + 0.9 * (0.8 * TCKD + 1.2 * NCKD)
     = 0.1 * CE + 0.72 * TCKD + 1.08 * NCKD
```

## 📁 文件结构

```
workspace-whisper/
├── configs/
│   ├── small_dkd_s2_t1_a08.yaml    # S2, T=1.0
│   ├── small_dkd_s2_t2_a08.yaml    # S2, T=2.0
│   ├── small_dkd_s3_t1_a08.yaml    # S3, T=1.0
│   ├── small_dkd_s3_t2_a08.yaml    # S3, T=2.0
│   ├── small_dkd_s4_t1_a08.yaml    # S4, T=1.0
│   └── small_dkd_s4_t2_a08.yaml    # S4, T=2.0
│
├── scripts/
│   ├── train_dkd_all_configs.sh    # 🔥 批量训练所有配置
│   ├── verify_dkd_configs.py       # ✅ 验证配置文件
│   └── show_dkd_plan.sh            # 📊 显示实验计划
│
├── src/
│   └── training/
│       └── train_dkd.py            # DKD 训练脚本
│
└── DKD_EXPERIMENTS_GUIDE.md        # 本文档
```

## 🚀 快速开始

### 1. 验证配置

```bash
cd /home/zz359/workspace-whisper
python scripts/verify_dkd_configs.py
```

**预期输出**:
```
✅ 所有配置文件验证通过！

| 模型 | T=1.0 | T=2.0 |
|------|-------|-------|
| S2   | α=0.8, β=1.2, γ=0.1 | α=0.8, β=1.2, γ=0.1 |
| S3   | α=0.8, β=1.2, γ=0.1 | α=0.8, β=1.2, γ=0.1 |
| S4   | α=0.8, β=1.2, γ=0.1 | α=0.8, β=1.2, γ=0.1 |
```

### 2. 查看实验计划

```bash
bash scripts/show_dkd_plan.sh
```

### 3. 开始训练

#### 方式 A: 批量训练所有配置（推荐）

```bash
bash scripts/train_dkd_all_configs.sh
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
python -m src.training.train_dkd --config configs/small_dkd_s2_t2_a08.yaml

# 或指定其他配置
python -m src.training.train_dkd --config configs/small_dkd_s3_t1_a08.yaml
```

### 4. 监控训练

#### 实时查看日志

```bash
# 查看训练日志
tail -f outputs/small_dkd_s2_t2/train.log

# 或查看脚本日志
tail -f logs/train_S2-T2_*.log
```

#### 查看关键指标

```bash
# 查看最佳 WER
grep "Best WER" outputs/small_dkd_s*/train.log

# 查看训练进度
grep "Epoch.*completed" outputs/small_dkd_s2_t2/train.log

# 查看 loss 分解 (CE, DKD, TCKD, NCKD)
grep "ce=.*dkd=.*tckd=.*nckd=" outputs/small_dkd_s2_t2/train.log | tail -20
```

## 📊 配置详情

### 配置参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| **teacher_model** | openai/whisper-small | Teacher 模型 |
| **student_decoder_layers** | 2/3/4 | Student 层数 |
| **student_decoder_heads** | 12 | 注意力头数 |
| **student_hidden_size** | 768 | 隐藏层大小 |
| **temperature** | 1.0/2.0 | DKD 温度 |
| **alpha** | 0.8 | TCKD 权重 |
| **beta** | 1.2 | NCKD 权重 |
| **gamma** | 0.1 | CE loss 权重 |
| **learning_rate** | 1e-4 | 学习率 |
| **batch_size** | 16 | 批次大小 |
| **epochs** | 15 | 训练轮数 |

### 实验矩阵

| 配置 | 模型 | Temp | α | β | γ | LR | 输出目录 | 预计时间 |
|------|------|------|---|---|---|----|---------|---------|
| 1 | S2 | 1.0 | 0.8 | 1.2 | 0.1 | 1e-4 | `outputs/small_dkd_s2_t1` | 2h |
| 2 | S2 | 2.0 | 0.8 | 1.2 | 0.1 | 1e-4 | `outputs/small_dkd_s2_t2` | 2h |
| 3 | S3 | 1.0 | 0.8 | 1.2 | 0.1 | 1e-4 | `outputs/small_dkd_s3_t1` | 2.5h |
| 4 | S3 | 2.0 | 0.8 | 1.2 | 0.1 | 1e-4 | `outputs/small_dkd_s3_t2` | 2.5h |
| 5 | S4 | 1.0 | 0.8 | 1.2 | 0.1 | 1e-4 | `outputs/small_dkd_s4_t1` | 3h |
| 6 | S4 | 2.0 | 0.8 | 1.2 | 0.1 | 1e-4 | `outputs/small_dkd_s4_t2` | 3h |

## 🔍 代码实现细节

### DKD Loss 计算

```python
def compute_dkd_loss(student_logits, teacher_logits, labels, temperature, alpha, beta):
    """
    DKD = alpha * TCKD + beta * NCKD
    """
    # 1. 获取 softmax 概率（带温度）
    student_probs = F.softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # 2. 提取目标类概率
    target_mask = one_hot(labels)
    student_target = (student_probs * target_mask).sum(dim=-1)
    teacher_target = (teacher_probs * target_mask).sum(dim=-1)
    
    # 3. TCKD: 二分类 KL 散度 (target vs non-target)
    tckd = teacher_target * log(teacher_target / student_target) + \
           (1 - teacher_target) * log((1 - teacher_target) / (1 - student_target))
    
    # 4. NCKD: 非目标类分布的 KL 散度
    student_non_target = student_probs * (1 - target_mask)
    teacher_non_target = teacher_probs * (1 - target_mask)
    
    # 归一化非目标概率
    student_non_target_norm = student_non_target / student_non_target.sum()
    teacher_non_target_norm = teacher_non_target / teacher_non_target.sum()
    
    nckd = KL(teacher_non_target_norm || student_non_target_norm)
    
    # 5. 组合 TCKD 和 NCKD
    dkd_loss = alpha * tckd + beta * nckd
    
    # 6. 温度缩放
    dkd_loss = dkd_loss * (temperature ** 2)
    
    return dkd_loss, tckd, nckd
```

### 训练循环

```python
# Forward pass
ce_loss, student_logits, teacher_logits = model(
    input_features=input_features,
    decoder_input_ids=decoder_input_ids,
    labels=labels,
    return_teacher_logits=True,
)

# Compute DKD loss
dkd_loss, tckd_loss, nckd_loss = compute_dkd_loss(
    student_logits=student_logits,
    teacher_logits=teacher_logits,
    labels=labels,
    temperature=config.dkd.temperature,
    alpha=config.dkd.alpha,
    beta=config.dkd.beta,
)

# Combined loss
loss = config.dkd.gamma * ce_loss + (1 - config.dkd.gamma) * dkd_loss

# Backward
loss.backward()
optimizer.step()
```

## 📈 预期结果

### Baseline 对比

**CE Baseline** (已完成):
- `small_ce_s2`: 23.29% WER
- `small_ce_s3`: 18.88% WER

**KD Baseline** (如果已完成):
- 查看 `outputs/small_kd_s*/train.log`

**DKD 目标**:
- `small_dkd_s2_t2`: < 23% WER (期望优于 CE 和 KD) ⭐
- `small_dkd_s3_t2`: < 18% WER (期望优于 CE 和 KD) ⭐
- `small_dkd_s4_t2`: < 15% WER (期望最佳) ⭐

### DKD vs KD 的优势

根据论文，DKD 相比标准 KD 的优势：

1. **解耦目标和非目标知识**
   - TCKD 专注于目标类别
   - NCKD 专注于类间关系

2. **更灵活的权重控制**
   - 可以独立调整 α 和 β
   - 标准 KD 只有一个权重

3. **更好的性能**
   - 在 ImageNet、CIFAR 等任务上优于标准 KD
   - 期望在 ASR 任务上也有提升

## 🔬 分析方法

### 1. 对比不同方法

```bash
# 查看所有结果
echo "CE Baseline:"
grep "Best WER" outputs/small_ce_s*/train.log

echo "KD Baseline:"
grep "Best WER" outputs/small_kd_s*/train.log

echo "DKD (本次):"
grep "Best WER" outputs/small_dkd_s*/train.log
```

### 2. 分析 TCKD vs NCKD

```bash
# 查看 TCKD 和 NCKD 的值
grep "tckd=.*nckd=" outputs/small_dkd_s2_t2/train.log | tail -50

# 计算平均值
grep "tckd=" outputs/small_dkd_s2_t2/train.log | \
  awk '{print $5, $6}' | \
  awk -F'[=,]' '{tckd+=$2; nckd+=$4; n++} END {print "Avg TCKD:", tckd/n, "Avg NCKD:", nckd/n}'
```

### 3. 分析 Temperature 影响

```bash
# 对比 T=1 vs T=2
for model in s2 s3 s4; do
    echo "Model: $model"
    echo "  T=1.0: $(grep 'Best WER' outputs/small_dkd_${model}_t1/train.log)"
    echo "  T=2.0: $(grep 'Best WER' outputs/small_dkd_${model}_t2/train.log)"
done
```

### 4. 分析模型大小影响

```bash
# 对比不同模型大小
for temp in t1 t2; do
    echo "Temperature: $temp"
    echo "  S2: $(grep 'Best WER' outputs/small_dkd_s2_${temp}/train.log)"
    echo "  S3: $(grep 'Best WER' outputs/small_dkd_s3_${temp}/train.log)"
    echo "  S4: $(grep 'Best WER' outputs/small_dkd_s4_${temp}/train.log)"
done
```

## 🐛 故障排查

### 问题 1: TCKD 或 NCKD 为 NaN

**症状**: 训练日志显示 `tckd=nan` 或 `nckd=nan`

**可能原因**:
1. 概率值为 0 导致 log(0)
2. 除以 0

**解决方案**:
- 代码中已添加 `torch.clamp(min=1e-7)` 防止数值不稳定
- 如果仍有问题，可以增加 clamp 的最小值

### 问题 2: DKD loss 过大或过小

**症状**: DKD loss 数值异常

**可能原因**:
1. Alpha/Beta 权重不合理
2. Temperature 设置不当

**解决方案**:
```bash
# 尝试调整权重
# 编辑配置文件，修改 alpha, beta, gamma
```

### 问题 3: 训练不稳定

**症状**: Loss 剧烈波动

**解决方案**:
1. 降低 learning rate
2. 调整 alpha/beta 权重
3. 使用更小的 temperature

## 📝 实验记录模板

训练完成后，记录以下信息：

```markdown
### DKD 实验结果 - [日期]

#### 配置

| 配置 | WER | CER | 训练时间 | TCKD | NCKD | 备注 |
|------|-----|-----|---------|------|------|------|
| S2-T1 | X.XX% | X.XX% | Xh | X.XX | X.XX | ... |
| S2-T2 | X.XX% | X.XX% | Xh | X.XX | X.XX | ... |
| S3-T1 | X.XX% | X.XX% | Xh | X.XX | X.XX | ... |
| S3-T2 | X.XX% | X.XX% | Xh | X.XX | X.XX | ... |
| S4-T1 | X.XX% | X.XX% | Xh | X.XX | X.XX | ... |
| S4-T2 | X.XX% | X.XX% | Xh | X.XX | X.XX | ... |

**最佳配置**: S?-T?
**最佳 WER**: X.XX%

#### 与 Baseline 对比

| 方法 | S2 WER | S3 WER | S4 WER |
|------|--------|--------|--------|
| CE   | 23.29% | 18.88% | -      |
| KD   | X.XX%  | X.XX%  | X.XX%  |
| DKD  | X.XX%  | X.XX%  | X.XX%  |

#### 观察

1. **DKD vs KD**:
   - DKD 相比 KD 的改进: ...
   - TCKD vs NCKD 的贡献: ...

2. **Temperature 影响**:
   - T=1 vs T=2: ...
   - 最佳温度: ...

3. **模型大小影响**:
   - S2 vs S3 vs S4: ...
   - 性能/参数 trade-off: ...

#### 结论

...

#### 下一步

...
```

## 🎯 下一步

训练完成后：

1. **收集结果**
   ```bash
   grep "Best WER" outputs/small_dkd_s*/train.log > dkd_results_summary.txt
   ```

2. **分析对比**
   - 对比 DKD vs KD vs CE
   - 分析 TCKD vs NCKD 的贡献
   - 分析 Temperature 影响
   - 分析模型大小影响

3. **可选实验**
   - 测试其他 Alpha/Beta 组合
   - 测试其他 Gamma 值
   - 测试其他 Temperature

4. **撰写报告**
   - 总结 DKD 的效果
   - 分析最佳配置
   - 提出改进建议

## 📚 参考资料

### DKD 论文

**Decoupled Knowledge Distillation** (CVPR 2022)
- 论文: https://arxiv.org/abs/2203.08679
- 代码: https://github.com/megvii-research/mdistiller

**主要贡献**:
1. 将 KD loss 解耦为 TCKD 和 NCKD
2. TCKD 关注目标类，NCKD 关注类间关系
3. 在 ImageNet、CIFAR 等任务上优于标准 KD

### 相关论文

1. **Hinton et al. (2015)**: Distilling the Knowledge in a Neural Network
   - 提出 Knowledge Distillation
   - Temperature scaling

2. **Sanh et al. (2019)**: DistilBERT
   - 在 NLP 任务上应用 KD

3. **Zhao et al. (2022)**: Decoupled Knowledge Distillation
   - DKD 方法

### 代码参考

- `src/training/train_dkd.py`: DKD 训练主逻辑
- `configs/small_dkd_s*_t*_a08.yaml`: 配置文件示例

---

**准备就绪！** 🚀

运行以下命令开始实验：

```bash
cd /home/zz359/workspace-whisper
python scripts/verify_dkd_configs.py  # 验证配置
bash scripts/train_dkd_all_configs.sh  # 开始训练
```

祝实验顺利！ 🎉

