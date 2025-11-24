# KD vs DKD 对比指南

## 📊 方法对比

### 标准 KD (Knowledge Distillation)

**Loss 公式**:
```
Loss = α * CE + (1 - α) * KD
KD = KL(Teacher || Student) * T²
```

**参数**:
- `α = 0.1`: CE 权重
- `T = 1.0 or 2.0`: Temperature

**特点**:
- ✅ 简单直接
- ✅ 广泛使用
- ❌ 目标类和非目标类混在一起
- ❌ 只有一个权重参数

### DKD (Decoupled Knowledge Distillation)

**Loss 公式**:
```
Loss = γ * CE + (1 - γ) * DKD
DKD = α * TCKD + β * NCKD
```

**参数**:
- `γ = 0.1`: CE 权重
- `α = 0.8`: TCKD 权重
- `β = 1.2`: NCKD 权重
- `T = 1.0 or 2.0`: Temperature

**特点**:
- ✅ 解耦目标类和非目标类
- ✅ 更灵活的权重控制
- ✅ 理论上更优
- ❌ 稍微复杂
- ❌ 需要调整更多参数

## 🔬 技术细节对比

### KD Loss 计算

```python
# 标准 KD
student_log_probs = F.log_softmax(student_logits / T, dim=-1)
teacher_probs = F.softmax(teacher_logits / T, dim=-1)
kd_loss = F.kl_div(student_log_probs, teacher_probs) * (T ** 2)
```

### DKD Loss 计算

```python
# DKD
student_probs = F.softmax(student_logits / T, dim=-1)
teacher_probs = F.softmax(teacher_logits / T, dim=-1)

# TCKD: Binary KL on target class
student_target = (student_probs * target_mask).sum()
teacher_target = (teacher_probs * target_mask).sum()
tckd = binary_kl(teacher_target, student_target)

# NCKD: KL on non-target distribution
student_non_target = normalize(student_probs * (1 - target_mask))
teacher_non_target = normalize(teacher_probs * (1 - target_mask))
nckd = kl_div(teacher_non_target, student_non_target)

# Combine
dkd_loss = (alpha * tckd + beta * nckd) * (T ** 2)
```

## 📁 文件对比

### KD 文件

```
configs/
├── small_kd_s2_t1_a01.yaml
├── small_kd_s2_t2_a01.yaml
├── small_kd_s3_t1_a01.yaml
├── small_kd_s3_t2_a01.yaml
├── small_kd_s4_t1_a01.yaml
└── small_kd_s4_t2_a01.yaml

scripts/
├── train_kd_all_configs.sh
└── verify_configs.py

src/training/
└── train_kd.py
```

### DKD 文件

```
configs/
├── small_dkd_s2_t1_a08.yaml
├── small_dkd_s2_t2_a08.yaml
├── small_dkd_s3_t1_a08.yaml
├── small_dkd_s3_t2_a08.yaml
├── small_dkd_s4_t1_a08.yaml
└── small_dkd_s4_t2_a08.yaml

scripts/
├── train_dkd_all_configs.sh
├── verify_dkd_configs.py
└── show_dkd_plan.sh

src/training/
└── train_dkd.py
```

## 🚀 运行对比

### 训练 KD

```bash
# 验证配置
python scripts/verify_configs.py

# 批量训练
bash scripts/train_kd_all_configs.sh

# 单独训练
python -m src.training.train_kd --config configs/small_kd_s2_t2_a01.yaml
```

### 训练 DKD

```bash
# 验证配置
python scripts/verify_dkd_configs.py

# 批量训练
bash scripts/train_dkd_all_configs.sh

# 单独训练
python -m src.training.train_dkd --config configs/small_dkd_s2_t2_a08.yaml
```

## 📊 配置对比

### KD 配置

```yaml
kd:
  temperature: 2.0  # T
  alpha: 0.1        # CE weight
```

**含义**:
- `Loss = 0.1 * CE + 0.9 * KD`
- KD 占 90%

### DKD 配置

```yaml
dkd:
  temperature: 2.0  # T
  alpha: 0.8        # TCKD weight
  beta: 1.2         # NCKD weight
  gamma: 0.1        # CE weight
```

**含义**:
- `Loss = 0.1 * CE + 0.9 * DKD`
- `DKD = 0.8 * TCKD + 1.2 * NCKD`
- 展开: `Loss = 0.1 * CE + 0.72 * TCKD + 1.08 * NCKD`

## 📈 预期结果对比

### Baseline

| 方法 | S2 | S3 | S4 |
|------|----|----|-----|
| CE   | 23.29% | 18.88% | - |

### KD 预期

| 配置 | S2 | S3 | S4 |
|------|----|----|-----|
| T=1.0 | ~23% | ~18% | ~15% |
| T=2.0 | **< 23%** ⭐ | **< 18%** ⭐ | **< 15%** ⭐ |

### DKD 预期

| 配置 | S2 | S3 | S4 |
|------|----|----|-----|
| T=1.0 | ~22% | ~17% | ~14% |
| T=2.0 | **< 22%** ⭐⭐ | **< 17%** ⭐⭐ | **< 14%** ⭐⭐ |

**期望**: DKD 优于 KD 优于 CE

## 🔍 监控对比

### KD 日志

```bash
# 查看 KD loss
grep "kd=" outputs/small_kd_s2_t2/train.log | tail -20

# 输出示例:
# Step 1000: loss=2.3456, ce=2.1234, kd=2.5678, lr=1.00e-04
```

### DKD 日志

```bash
# 查看 DKD loss (包含 TCKD 和 NCKD)
grep "dkd=.*tckd=.*nckd=" outputs/small_dkd_s2_t2/train.log | tail -20

# 输出示例:
# Step 1000: loss=2.3456, ce=2.1234, dkd=2.5678 (tckd=1.2345, nckd=1.3333), lr=1.00e-04
```

## 📊 结果分析

### 收集所有结果

```bash
# KD 结果
echo "=== KD Results ==="
grep "Best WER" outputs/small_kd_s*/train.log

# DKD 结果
echo "=== DKD Results ==="
grep "Best WER" outputs/small_dkd_s*/train.log
```

### 对比脚本

创建 `scripts/compare_kd_dkd.sh`:

```bash
#!/bin/bash
echo "KD vs DKD Comparison"
echo "===================="
echo ""

for model in s2 s3 s4; do
    echo "Model: $model"
    
    for temp in t1 t2; do
        kd_wer=$(grep "Best WER" outputs/small_kd_${model}_${temp}/train.log 2>/dev/null | grep -oP '\d+\.\d+' || echo "N/A")
        dkd_wer=$(grep "Best WER" outputs/small_dkd_${model}_${temp}/train.log 2>/dev/null | grep -oP '\d+\.\d+' || echo "N/A")
        
        echo "  T=${temp#t}.0:"
        echo "    KD:  ${kd_wer}%"
        echo "    DKD: ${dkd_wer}%"
        
        if [ "$kd_wer" != "N/A" ] && [ "$dkd_wer" != "N/A" ]; then
            improvement=$(echo "$kd_wer - $dkd_wer" | bc)
            echo "    Improvement: ${improvement}%"
        fi
    done
    echo ""
done
```

## 🎯 实验建议

### 推荐顺序

1. **先运行 KD** (如果还没运行)
   ```bash
   bash scripts/train_kd_all_configs.sh
   ```

2. **再运行 DKD**
   ```bash
   bash scripts/train_dkd_all_configs.sh
   ```

3. **对比结果**
   ```bash
   bash scripts/compare_kd_dkd.sh
   ```

### 快速测试

如果时间有限，可以只测试最关键的配置：

```bash
# 1. KD S2-T2 (最快，约2小时)
python -m src.training.train_kd --config configs/small_kd_s2_t2_a01.yaml

# 2. DKD S2-T2 (最快，约2小时)
python -m src.training.train_dkd --config configs/small_dkd_s2_t2_a08.yaml

# 3. 对比结果
grep "Best WER" outputs/small_kd_s2_t2/train.log
grep "Best WER" outputs/small_dkd_s2_t2/train.log
```

## 📝 结果记录模板

```markdown
### KD vs DKD 实验结果

#### 配置

| 方法 | 模型 | Temp | WER | CER | 训练时间 |
|------|------|------|-----|-----|---------|
| CE   | S2   | -    | 23.29% | - | - |
| CE   | S3   | -    | 18.88% | - | - |
| KD   | S2   | 1.0  | X.XX% | X.XX% | Xh |
| KD   | S2   | 2.0  | X.XX% | X.XX% | Xh |
| KD   | S3   | 1.0  | X.XX% | X.XX% | Xh |
| KD   | S3   | 2.0  | X.XX% | X.XX% | Xh |
| DKD  | S2   | 1.0  | X.XX% | X.XX% | Xh |
| DKD  | S2   | 2.0  | X.XX% | X.XX% | Xh |
| DKD  | S3   | 1.0  | X.XX% | X.XX% | Xh |
| DKD  | S3   | 2.0  | X.XX% | X.XX% | Xh |

#### 最佳结果

- **KD 最佳**: S?-T? = X.XX% WER
- **DKD 最佳**: S?-T? = X.XX% WER
- **改进**: X.XX%

#### 分析

1. **DKD vs KD**:
   - DKD 平均改进: ...
   - 最大改进: ...
   - 最小改进: ...

2. **TCKD vs NCKD**:
   - TCKD 平均值: ...
   - NCKD 平均值: ...
   - 贡献比例: ...

3. **Temperature 影响**:
   - T=1 vs T=2: ...
   - 最佳温度: ...

#### 结论

...
```

## 🔬 深入分析

### 分析 TCKD 和 NCKD 的贡献

```bash
# 提取 TCKD 和 NCKD 的值
grep "tckd=.*nckd=" outputs/small_dkd_s2_t2/train.log | \
  awk -F'[=,()]' '{print $8, $10}' > tckd_nckd.txt

# 使用 Python 分析
python -c "
import numpy as np
data = np.loadtxt('tckd_nckd.txt')
print(f'TCKD: mean={data[:,0].mean():.4f}, std={data[:,0].std():.4f}')
print(f'NCKD: mean={data[:,1].mean():.4f}, std={data[:,1].std():.4f}')
print(f'Ratio: {data[:,0].mean() / data[:,1].mean():.4f}')
"
```

### 可视化对比

如果有 matplotlib:

```python
import matplotlib.pyplot as plt
import re

# 读取 KD 和 DKD 的 loss
def parse_log(log_file):
    losses = []
    with open(log_file) as f:
        for line in f:
            if 'Step' in line and 'loss=' in line:
                loss = float(re.search(r'loss=([\d.]+)', line).group(1))
                losses.append(loss)
    return losses

kd_losses = parse_log('outputs/small_kd_s2_t2/train.log')
dkd_losses = parse_log('outputs/small_dkd_s2_t2/train.log')

plt.figure(figsize=(10, 6))
plt.plot(kd_losses, label='KD', alpha=0.7)
plt.plot(dkd_losses, label='DKD', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('KD vs DKD Training Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('kd_vs_dkd_loss.png')
print('Saved to kd_vs_dkd_loss.png')
```

## 📚 参考资料

### KD 论文
- Hinton et al. (2015): Distilling the Knowledge in a Neural Network
- https://arxiv.org/abs/1503.02531

### DKD 论文
- Zhao et al. (2022): Decoupled Knowledge Distillation
- https://arxiv.org/abs/2203.08679

### 实现参考
- KD: `src/training/train_kd.py`
- DKD: `src/training/train_dkd.py`

---

**总结**: DKD 是 KD 的改进版本，通过解耦目标类和非目标类的知识，理论上应该获得更好的性能。实验将验证这一假设。

祝实验顺利！ 🎉

