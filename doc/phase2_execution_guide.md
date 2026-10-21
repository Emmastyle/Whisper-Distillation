# Phase 2 执行指南

本文档提供 Phase 2 的详细执行步骤和命令。

## 前置条件

1. 已完成 Phase 1 的所有设置
2. 已安装所有依赖（参考 `requirements.txt`）
3. 激活 `whisper` conda 环境
4. 确保有足够的 GPU 显存（建议 ≥16GB）
5. 确保有足够的磁盘空间用于存储模型和日志

## 执行步骤

### Step 1: 验证 Phase 2 设置

```bash
cd workspace-whisper
source ~/miniconda3/etc/profile.d/conda.sh
conda activate whisper
python test_phase2_setup.py
```

**预期输出**: 所有测试通过 ✓

### Step 2: 数据分析（已完成）

```bash
python scripts/analyze_data_100h.py
```

**输出文件**:
- `outputs/data_100h_statistics.txt`

**关键统计**:
- 有效样本数: 5,366（≤10秒）
- 训练集时长: 8.23 小时
- 验证集时长: 0.49 小时
- 测试集时长: 0.45 小时

### Step 3: Teacher Model 基线测试

```bash
python scripts/test_teacher_small.py --model openai/whisper-small --num_samples 100
```

**输出文件**:
- `outputs/teacher_small_report.txt` - 性能报告
- `outputs/teacher_small_samples.txt` - 预测样本

**预期指标**:
- 模型参数量: ~244M
- WER: ~5-8%（LibriSpeech test-clean）
- RTF: <1.0（实时）
- GPU显存: ~1-2GB

**注意**: 首次运行会下载 whisper-small 模型（~967MB）

### Step 4: CE-only 基线训练

依次训练 S2/S3/S4 三个 student decoder 配置：

```bash
# S2 (2层 decoder)
bash scripts/train_ce.sh configs/small_ce_s2.yaml

# S3 (3层 decoder)
bash scripts/train_ce.sh configs/small_ce_s3.yaml

# S4 (4层 decoder)
bash scripts/train_ce.sh configs/small_ce_s4.yaml
```

**或使用批量训练脚本的 CE 部分**（需手动注释掉 KD 部分）：

```bash
bash scripts/train_phase2_all.sh
```

**预期训练时间**: 每个配置约 2-4 小时（取决于硬件）

**输出目录**:
- `outputs/small_ce_s2/`
  - `best_model.pt`
  - `checkpoint_*.pt`
  - `train.log`
- `outputs/small_ce_s3/`
- `outputs/small_ce_s4/`

### Step 5: KD 训练（T=4）

依次训练 S2/S3/S4 的 KD 版本：

```bash
# S2 KD
bash scripts/train_kd.sh configs/small_kd_s2.yaml

# S3 KD
bash scripts/train_kd.sh configs/small_kd_s3.yaml

# S4 KD
bash scripts/train_kd.sh configs/small_kd_s4.yaml
```

**预期训练时间**: 每个配置约 2-4 小时

**输出目录**:
- `outputs/small_kd_s2/`
- `outputs/small_kd_s3/`
- `outputs/small_kd_s4/`

### Step 6: 温度扫描实验（S3）

测试不同温度对 S3 student 的影响：

```bash
bash scripts/train_temperature_sweep.sh
```

**测试温度**: T = 2, 3, 4, 5, 6, 8

**预期训练时间**: 总计约 12-24 小时（6个模型）

**输出目录**:
- `outputs/small_kd_s3_t2/`
- `outputs/small_kd_s3_t3/`
- `outputs/small_kd_s3_t4/`（与 Step 5 的 S3 KD 相同）
- `outputs/small_kd_s3_t5/`
- `outputs/small_kd_s3_t6/`
- `outputs/small_kd_s3_t8/`

### Step 7: 统一评估

评估所有训练好的模型：

```bash
python scripts/evaluate_all_models.py --num_samples 100
```

**输出文件**:
- `results/model_comparison.csv` - 模型对比表格（CSV）
- `results/model_comparison.tex` - 模型对比表格（LaTeX）
- `results/speed_accuracy_curve.png` - 速度-准确率曲线（PNG）
- `results/speed_accuracy_curve.pdf` - 速度-准确率曲线（PDF）
- `results/temperature_comparison.png` - 温度对比图（PNG）
- `results/temperature_comparison.pdf` - 温度对比图（PDF）

**评估指标**:
- 参数量（Params）
- 计算量（FLOPs）
- 实时因子（RTF）
- 词错误率（WER）
- 字符错误率（CER）
- 推理速度（Speed，相对实时）

### Step 8: 结果分析与文档

1. 查看模型对比表格：
   ```bash
   cat results/model_comparison.csv
   ```

2. 查看图表：
   - 使用图片查看器打开 `results/*.png`
   - 或使用 PDF 阅读器打开 `results/*.pdf`

3. 撰写 Phase 2 结果报告：
   - 基于评估结果填写 `doc/phase2_results.md`
   - 总结关键发现
   - 对比 CE vs KD
   - 分析温度影响
   - 绘制结论

## 训练监控

### 实时查看训练日志

```bash
# 查看最新的训练日志
tail -f logs/*.log

# 或查看特定模型的日志
tail -f logs/small_ce_s2_*.log
```

### 检查 GPU 使用情况

```bash
watch -n 1 nvidia-smi
```

### 检查磁盘空间

```bash
df -h
du -sh outputs/
```

## 故障排除

### 问题 1: 显存不足（OOM）

**症状**: `CUDA out of memory`

**解决方案**:
1. 减小 batch_size（在配置文件中修改）
2. 使用梯度累积
3. 启用 fp16 混合精度训练（已在配置中启用）

### 问题 2: 数据加载错误

**症状**: 音频解码失败

**解决方案**:
1. 确保已安装 `datasets` 库
2. 检查 FFmpeg 是否正确安装
3. 或使用 `librosa` 作为备选（参考 Phase 1 的修复）

### 问题 3: 训练中断

**症状**: 训练意外停止

**解决方案**:
1. 从最近的 checkpoint 恢复训练
2. 修改训练脚本添加断点续训逻辑：
   ```python
   if os.path.exists(resume_checkpoint):
       model.load_state_dict(torch.load(resume_checkpoint))
   ```

### 问题 4: 模型收敛慢

**症状**: loss 下降缓慢或停滞

**解决方案**:
1. 调整学习率
2. 增加 warmup steps
3. 检查数据质量
4. 对比 teacher 的输出是否合理

## 时间估算

基于典型硬件配置（单张 V100/A100 GPU）：

| 步骤 | 模型数 | 单模型时间 | 总时间 |
|-----|-------|----------|--------|
| Teacher 验证 | 1 | 10 min | 10 min |
| CE 训练 (S2/S3/S4) | 3 | 2-4 h | 6-12 h |
| KD 训练 (S2/S3/S4) | 3 | 2-4 h | 6-12 h |
| 温度扫描 (6个T) | 6 | 2-4 h | 12-24 h |
| 评估 | 所有 | 30 min | 30 min |
| **总计** | - | - | **~25-49 h** |

**建议**: 
- 使用后台运行或 tmux/screen
- 分批次执行（例如先完成 CE，再完成 KD）
- 使用批量训练脚本并设置在夜间运行

## 快速开始（最小化版本）

如果时间或资源有限，可以只运行核心实验：

```bash
# 1. 验证设置
python test_phase2_setup.py

# 2. 只训练 S3（最平衡的配置）
bash scripts/train_ce.sh configs/small_ce_s3.yaml
bash scripts/train_kd.sh configs/small_kd_s3.yaml

# 3. 温度扫描（S3, T=2,4,6）
# 手动运行 train_temperature_sweep.sh 中的部分温度

# 4. 评估
python scripts/evaluate_all_models.py --num_samples 50
```

**最小化时间估算**: ~8-12 小时

## 下一步

Phase 2 完成后，根据结果决定：

1. **Phase 3**: 实现 DKD（Decoupled Knowledge Distillation）
   - TCKD (Target-Class KD)
   - NCKD (Non-Target-Class KD)
   - 进一步优化性能

2. **优化方向**:
   - 模型量化（INT8/FP16）
   - 模型剪枝
   - 更大的数据集
   - 更先进的蒸馏技术

3. **论文/报告撰写**:
   - 整理实验结果
   - 绘制对比图表
   - 撰写方法论和结论

## 参考资料

- [Whisper 官方文档](https://github.com/openai/whisper)
- [LibriSpeech 数据集](http://www.openslr.org/12/)
- [Knowledge Distillation 综述](https://arxiv.org/abs/2006.05525)
- 项目文档：`doc/Project Plan Phase 2.md`

