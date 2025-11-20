# Phase 2 实验结果报告

## 概述

本报告总结了 Phase 2（Standard KD Complete Version）的实验结果，包括：
- Teacher 从 whisper-tiny 升级到 whisper-small
- 数据量从 10 样本扩展到 ~9 小时（5,366 有效样本）
- S2/S3/S4 三种 decoder 配置的 CE 和 KD 训练
- 温度参数优化实验

## 1. 数据集统计

### LibriSpeech train-clean-100

| 指标 | 数值 |
|-----|-----|
| 原始样本数 | 28,539 |
| 有效样本数（≤10s） | 5,366 |
| 过滤比例 | 81.20% |
| 总时长 | 9.17 小时 |
| 训练集时长 | 8.23 小时（4,829 样本） |
| 验证集时长 | 0.49 小时（268 样本） |
| 测试集时长 | 0.45 小时（269 样本） |
| 平均音频时长 | 6.15 秒 |
| 平均文本长度 | 90 字符 |

**说明**: 由于设置了 `max_duration=10s` 的限制，大部分长音频被过滤。如需使用完整100小时数据，需调整此参数或移除时长限制。

## 2. Teacher Model 基线

### whisper-small 性能

| 指标 | 数值 |
|-----|-----|
| 模型名称 | openai/whisper-small |
| 总参数量 | 244M |
| Encoder 层数 | 12 |
| Decoder 层数 | 12 |
| Hidden Size | 768 |
| WER (test-clean) | *待填写* % |
| CER (test-clean) | *待填写* % |
| RTF | *待填写* |
| 推理速度 | *待填写* x 实时 |
| GPU 显存占用 | *待填写* MB |

**运行命令**: `python scripts/test_teacher_small.py --num_samples 100`

## 3. Student Models 性能对比

### 3.1 CE-only Baseline

| Model | Layers | Params (M) | FLOPs (G) | WER (%) | CER (%) | RTF | Speed (x) |
|-------|--------|-----------|----------|---------|---------|-----|-----------|
| S2 CE | 2 | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* |
| S3 CE | 3 | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* |
| S4 CE | 4 | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* |

### 3.2 Knowledge Distillation (T=4)

| Model | Layers | Params (M) | FLOPs (G) | WER (%) | CER (%) | RTF | Speed (x) |
|-------|--------|-----------|----------|---------|---------|-----|-----------|
| S2 KD | 2 | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* |
| S3 KD | 3 | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* |
| S4 KD | 4 | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* | *待填写* |

### 3.3 CE vs KD 提升

| Model | CE WER (%) | KD WER (%) | 相对提升 (%) | 绝对提升 (%) |
|-------|-----------|-----------|------------|------------|
| S2 | *待填写* | *待填写* | *待填写* | *待填写* |
| S3 | *待填写* | *待填写* | *待填写* | *待填写* |
| S4 | *待填写* | *待填写* | *待填写* | *待填写* |

**运行命令**: `python scripts/evaluate_all_models.py --num_samples 100`

## 4. 温度参数优化实验

### S3 Model 不同温度的性能

| Temperature (T) | WER (%) | CER (%) | RTF | 训练 Loss |
|----------------|---------|---------|-----|----------|
| T=2 | *待填写* | *待填写* | *待填写* | *待填写* |
| T=3 | *待填写* | *待填写* | *待填写* | *待填写* |
| **T=4** | *待填写* | *待填写* | *待填写* | *待填写* |
| T=5 | *待填写* | *待填写* | *待填写* | *待填写* |
| T=6 | *待填写* | *待填写* | *待填写* | *待填写* |
| T=8 | *待填写* | *待填写* | *待填写* | *待填写* |

### 温度分析

- **最优温度**: T=*待填写*
- **WER 最低**: *待填写* % (T=*待填写*)
- **趋势观察**: *待填写*

**运行命令**: `bash scripts/train_temperature_sweep.sh`

## 5. 速度-准确率权衡分析

### 模型效率对比

| Model | Params (M) | Speedup vs Teacher | WER Gap vs Teacher (%) |
|-------|-----------|-------------------|----------------------|
| Teacher | 244 | 1.0x | 0.0% |
| S4 KD | *待填写* | *待填写* x | *待填写* % |
| S3 KD | *待填写* | *待填写* x | *待填写* % |
| S2 KD | *待填写* | *待填写* x | *待填写* % |

### 关键发现

1. **参数压缩比**: 
   - S2: *待填写* x 压缩
   - S3: *待填写* x 压缩
   - S4: *待填写* x 压缩

2. **速度提升**:
   - S2: *待填写* x 加速
   - S3: *待填写* x 加速
   - S4: *待填写* x 加速

3. **性能保持**:
   - S4 KD 相对 Teacher 的 WER 差距: *待填写* %
   - S3 KD 相对 Teacher 的 WER 差距: *待填写* %
   - S2 KD 相对 Teacher 的 WER 差距: *待填写* %

## 6. 可视化结果

### 6.1 速度-准确率曲线

![Speed-Accuracy Curve](../results/speed_accuracy_curve.png)

**观察**:
- *待填写*

### 6.2 参数量-准确率关系

**观察**:
- *待填写*

### 6.3 温度对比图

![Temperature Comparison](../results/temperature_comparison.png)

**观察**:
- *待填写*

## 7. 训练细节

### 训练配置

| 参数 | 数值 |
|-----|-----|
| Batch Size | 16 |
| Learning Rate | 1e-4 |
| Weight Decay | 0.01 |
| Warmup Steps | 500 |
| Epochs | 15 |
| FP16 | 启用 |
| α (KD weight) | 0.5 |

### 训练时间

| Model | 训练时间 | 硬件 |
|-------|---------|-----|
| S2 CE | *待填写* | *待填写* |
| S2 KD | *待填写* | *待填写* |
| S3 CE | *待填写* | *待填写* |
| S3 KD | *待填写* | *待填写* |
| S4 CE | *待填写* | *待填写* |
| S4 KD | *待填写* | *待填写* |
| 温度扫描 (6个) | *待填写* | *待填写* |

## 8. 结论与讨论

### 主要结论

1. **KD 的有效性**:
   - *待填写*

2. **最佳配置**:
   - *待填写*

3. **温度影响**:
   - *待填写*

4. **效率分析**:
   - *待填写*

### 与 Phase 1 对比

| 指标 | Phase 1 (tiny) | Phase 2 (small) | 变化 |
|-----|---------------|----------------|-----|
| Teacher Params | 39M | 244M | +6.3x |
| 数据量 | 10 samples | 5,366 samples | +536x |
| 训练时长 | ~1 min | ~8.23 h | +500x |
| WER | *待填写* | *待填写* | *待填写* |

### 局限性

1. **数据集**:
   - 由于 `max_duration=10s` 限制，实际只使用了约9小时数据
   - 建议：移除或调整时长限制以使用完整100小时

2. **评估**:
   - 当前仅在 LibriSpeech test-clean 上评估
   - 建议：增加 test-other 和其他数据集

3. **模型架构**:
   - 仅压缩 decoder，encoder 保持不变
   - 建议：探索 encoder 压缩或完整模型蒸馏

### 改进建议

1. **数据扩展**:
   - 使用完整100小时数据
   - 添加数据增强

2. **架构优化**:
   - 尝试不同的 decoder hidden size
   - 探索其他压缩技术（剪枝、量化）

3. **训练策略**:
   - 更长的训练周期
   - 学习率调度优化
   - 尝试其他 α 权重

4. **评估完善**:
   - 多数据集评估
   - 实际推理延迟测试
   - 用户体验评估

## 9. Phase 3 展望

基于 Phase 2 的结果，Phase 3 将探索更高级的知识蒸馏技术：

### DKD (Decoupled Knowledge Distillation)

1. **TCKD (Target-Class KD)**:
   - 关注目标类别的知识传递
   - 提升准确率

2. **NCKD (Non-Target-Class KD)**:
   - 利用非目标类的关系信息
   - 增强泛化能力

3. **预期提升**:
   - 相对标准 KD 进一步降低 WER
   - 更好的知识传递效率

## 10. 附录

### A. 配置文件列表

- `configs/data_100h.yaml`
- `configs/teacher_small.yaml`
- `configs/small_ce_s2.yaml`
- `configs/small_ce_s3.yaml`
- `configs/small_ce_s4.yaml`
- `configs/small_kd_s2.yaml`
- `configs/small_kd_s3.yaml`
- `configs/small_kd_s4.yaml`
- `configs/temp/small_kd_s3_t*.yaml`

### B. 输出文件

- 数据统计: `outputs/data_100h_statistics.txt`
- Teacher 报告: `outputs/teacher_small_report.txt`
- 训练模型: `outputs/small_*/best_model.pt`
- 评估结果: `results/model_comparison.csv`
- 可视化: `results/*.png`, `results/*.pdf`

### C. 脚本清单

- 数据分析: `scripts/analyze_data_100h.py`
- Teacher 测试: `scripts/test_teacher_small.py`
- 批量训练: `scripts/train_phase2_all.sh`
- 温度扫描: `scripts/train_temperature_sweep.sh`
- 统一评估: `scripts/evaluate_all_models.py`

### D. 复现步骤

详见 `doc/phase2_execution_guide.md`

## 参考文献

1. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper)
2. Hinton, G., et al. "Distilling the Knowledge in a Neural Network"
3. Zhao, B., et al. "Decoupled Knowledge Distillation"
4. LibriSpeech ASR corpus: http://www.openslr.org/12/

---

**报告日期**: *待填写*  
**实验环境**: *待填写*  
**作者**: *待填写*

