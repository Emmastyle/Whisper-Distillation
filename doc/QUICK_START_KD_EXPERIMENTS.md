# 🚀 KD Temperature 实验快速启动指南

## ✅ 已创建的文件

### 📄 配置文件 (9个)

```
configs/
├── small_kd_s2_t1.yaml         # S2, T=1.0, LR=1e-4
├── small_kd_s2_t2.yaml         # S2, T=2.0, LR=1e-4
├── small_kd_s2_t2_lr01.yaml    # S2, T=2.0, LR=1e-5 ⭐
├── small_kd_s3_t1.yaml         # S3, T=1.0, LR=1e-4
├── small_kd_s3_t2.yaml         # S3, T=2.0, LR=1e-4
├── small_kd_s3_t2_lr01.yaml    # S3, T=2.0, LR=1e-5 ⭐
├── small_kd_s4_t1.yaml         # S4, T=1.0, LR=1e-4
├── small_kd_s4_t2.yaml         # S4, T=2.0, LR=1e-4
└── small_kd_s4_t2_lr01.yaml    # S4, T=2.0, LR=1e-5 ⭐
```

### 🔧 训练脚本 (7个)

```
scripts/
├── train_kd_temperature_sweep.sh  # 批量运行所有实验
├── train_kd_t1_s2.sh             # 单独运行: S2, T=1
├── train_kd_t2_s2.sh             # 单独运行: S2, T=2
├── train_kd_t1_s3.sh             # 单独运行: S3, T=1
├── train_kd_t2_s3.sh             # 单独运行: S3, T=2
├── train_kd_t1_s4.sh             # 单独运行: S4, T=1
└── train_kd_t2_s4.sh             # 单独运行: S4, T=2
```

### 📚 文档

```
├── KD_TEMPERATURE_EXPERIMENTS.md    # 详细实验说明
└── QUICK_START_KD_EXPERIMENTS.md    # 本文件（快速启动）
```

## 🎯 实验目标

解决之前发现的问题：
- ❌ **small_kd_s2** (T=4): 25.95% WER（不如 CE 的 23.29%）
- ✅ 测试 **T=1, T=2** 是否能改善效果
- ✅ 测试 **降低学习率** (lr=0.1) 是否能提高稳定性

## 🚀 如何运行

### 方式 1: 批量运行所有实验（推荐用于完整实验）

```bash
cd workspace-whisper
bash scripts/train_kd_temperature_sweep.sh
```

**预计时间**: 18-27 小时（9个模型 × 2-3小时/模型）

### 方式 2: 单独运行（推荐用于快速测试）

```bash
cd workspace-whisper

# 先测试 S2 模型（最快）
bash scripts/train_kd_t1_s2.sh  # T=1.0, 约 2 小时
bash scripts/train_kd_t2_s2.sh  # T=2.0, 约 2 小时

# 如果 S2 效果好，再运行 S3
bash scripts/train_kd_t1_s3.sh  # T=1.0, 约 2.5 小时
bash scripts/train_kd_t2_s3.sh  # T=2.0, 约 2.5 小时

# 最后运行 S4
bash scripts/train_kd_t1_s4.sh  # T=1.0, 约 3 小时
bash scripts/train_kd_t2_s4.sh  # T=2.0, 约 3 小时
```

### 方式 3: 直接调用 Python

```bash
cd workspace-whisper

# 示例：运行 S2, T=2.0
python -m src.training.train_kd --config configs/small_kd_s2_t2.yaml

# 示例：运行 S3, T=1.0
python -m src.training.train_kd --config configs/small_kd_s3_t1.yaml
```

## 📊 实验矩阵

| 模型 | Temperature | LR | 配置文件 | 脚本 | 预计时间 |
|------|-------------|----|---------|----- |---------|
| S2 | 1.0 | 1e-4 | `small_kd_s2_t1.yaml` | `train_kd_t1_s2.sh` | 2h |
| S2 | 2.0 | 1e-4 | `small_kd_s2_t2.yaml` | `train_kd_t2_s2.sh` | 2h |
| S2 | 2.0 | 1e-5 | `small_kd_s2_t2_lr01.yaml` | - | 2h |
| S3 | 1.0 | 1e-4 | `small_kd_s3_t1.yaml` | `train_kd_t1_s3.sh` | 2.5h |
| S3 | 2.0 | 1e-4 | `small_kd_s3_t2.yaml` | `train_kd_t2_s3.sh` | 2.5h |
| S3 | 2.0 | 1e-5 | `small_kd_s3_t2_lr01.yaml` | - | 2.5h |
| S4 | 1.0 | 1e-4 | `small_kd_s4_t1.yaml` | `train_kd_t1_s4.sh` | 3h |
| S4 | 2.0 | 1e-4 | `small_kd_s4_t2.yaml` | `train_kd_t2_s4.sh` | 3h |
| S4 | 2.0 | 1e-5 | `small_kd_s4_t2_lr01.yaml` | - | 3h |

## 📈 监控训练

### 查看训练日志

```bash
# 实时查看训练进度
tail -f outputs/small_kd_s2_t1/train.log

# 查看最后 50 行
tail -50 outputs/small_kd_s2_t1/train.log

# 搜索最佳 WER
grep "Best WER" outputs/small_kd_s2_t1/train.log
```

### 查看所有实验状态

```bash
bash scripts/list_kd_experiments.sh
```

### 关键指标

训练过程中关注：
1. **WER 趋势**: 是否单调下降？
2. **KD Loss**: 是否有效收敛？
3. **训练稳定性**: 是否有剧烈波动？

```bash
# 查看 WER 变化
grep "Val Loss:" outputs/small_kd_s2_t1/train.log

# 查看 loss 组成
grep "Step.*ce=.*kd=" outputs/small_kd_s2_t1/train.log
```

## 🎯 预期结果

### 最佳配置预测

根据 KD 文献和之前的分析：

1. **T=2.0, LR=1e-4** ⭐⭐⭐
   - 最可能获得最佳性能
   - 文献推荐的标准配置

2. **T=1.0, LR=1e-4** ⭐⭐
   - 可能接近 CE 效果
   - 但可能过于 sharp

3. **T=2.0, LR=1e-5** ⭐
   - 更稳定但可能收敛不充分
   - 适合训练不稳定的情况

### 成功标准

- **目标 1**: WER < 23.29% (超越 CE baseline)
- **目标 2**: 训练稳定，无剧烈波动
- **目标 3**: KD loss 有效下降

### 对比 Baseline

```
已完成的 CE Baseline:
- small_ce_s2: 23.29% WER ✅
- small_ce_s3: 18.88% WER ✅

期望的 KD 结果:
- small_kd_s2_t2: < 23% WER? 🎯
- small_kd_s3_t2: < 18% WER? 🎯
```

## 🔍 结果分析

训练完成后，运行：

```bash
# 查看所有结果
python scripts/evaluate_all_models.py

# 或手动对比
grep "Best WER" outputs/small_kd_s*/train.log
```

## 📝 推荐的运行顺序

### 阶段 1: 快速验证（4-5 小时）

```bash
# 先跑 S2 的两个主要配置
bash scripts/train_kd_t1_s2.sh  # 2h
bash scripts/train_kd_t2_s2.sh  # 2h
```

**决策点**: 如果 T=2 明显优于 T=1，则后续优先跑 T=2

### 阶段 2: 深度模型（5-6 小时）

```bash
# 根据阶段1结果，选择最佳 temperature
bash scripts/train_kd_t2_s3.sh  # 2.5h（假设 T=2 更好）
bash scripts/train_kd_t2_s4.sh  # 3h
```

### 阶段 3: 完整实验（可选）

```bash
# 如果想要完整的对比数据
bash scripts/train_kd_temperature_sweep.sh  # 运行所有配置
```

## ⚠️ 注意事项

1. **GPU 使用**: 默认自动选择可用设备；也可通过 `--device cuda:N` 显式指定
   - 如需修改，编辑脚本中的 GPU 设置

2. **磁盘空间**: 每个模型约占用 500MB
   - 9 个模型需要约 5GB 空间

3. **训练中断**: 如果训练中断
   ```bash
   # 检查最后保存的 checkpoint
   ls -lh outputs/small_kd_s2_t1/checkpoint_*.pt
   
   # 可以从 checkpoint 恢复（需要修改代码支持）
   ```

4. **并行训练**: 如果有多个 GPU
   ```bash
   # Terminal 1 (GPU 0)
   python -m src.training.train_kd --config configs/small_kd_s2_t1_a01.yaml --device cuda:N &
   
   # Terminal 2 (GPU 1)
   python -m src.training.train_kd --config configs/small_kd_s2_t2_a01.yaml --device cuda:M &
   ```

## 📞 问题排查

### 问题 1: CUDA Out of Memory

```bash
# 解决方案：降低 batch size
# 编辑配置文件，修改 batch_size: 16 → 8
```

### 问题 2: 训练不稳定

```bash
# 尝试降低学习率的配置
python -m src.training.train_kd --config configs/small_kd_s2_t2_lr01.yaml
```

### 问题 3: 效果仍不如 CE

可能的原因：
1. Alpha 权重不合理（当前 0.5）
2. Teacher 模型质量限制
3. 数据集不适合 KD

## 📚 更多信息

- **详细实验设计**: 查看 `KD_TEMPERATURE_EXPERIMENTS.md`
- **理论背景**: 查看文档中的"理论背景"章节
- **之前的分析**: 查看之前的对话记录

## ✅ 检查清单

开始实验前：
- [ ] 确认 GPU 可用: `nvidia-smi`
- [ ] 确认环境激活: `conda activate whisper`
- [ ] 确认磁盘空间: `df -h`
- [ ] 查看配置列表: `bash scripts/list_kd_experiments.sh`

开始训练：
- [ ] 选择运行方式（批量 or 单独）
- [ ] 启动训练脚本
- [ ] 监控训练日志

训练完成后：
- [ ] 评估所有模型
- [ ] 对比 CE baseline
- [ ] 记录结果到 `KD_TEMPERATURE_EXPERIMENTS.md`
- [ ] 分析最佳配置

---

**准备就绪！** 🚀

运行以下命令开始实验：

```bash
cd workspace-whisper
bash scripts/train_kd_temperature_sweep.sh
```

或者先测试单个模型：

```bash
bash scripts/train_kd_t2_s2.sh
```
