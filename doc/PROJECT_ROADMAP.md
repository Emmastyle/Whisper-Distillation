# Whisper Knowledge Distillation 项目路线图

## 总体概览

本项目遵循 “Mini 验证 → 标准 KD 扩展 → DKD 创新” 的三步推进。

```
Phase 1 (已完成) → Phase 2 (标准 KD 扩展) → Phase 3 (DKD 研究) → Final Report
    ↓                    ↓                        ↓
 基线验证            Classical KD 完整版       DKD 创新实验
```

---

## Phase 1: 基线系统 ✅ (已完成)

### 目标
建立最小可运行版本，验证技术路线可行性

### 完成内容

#### 1.1 基础设施
- ✅ 项目目录结构
- ✅ 依赖管理 (requirements.txt)
- ✅ 配置系统 (YAML 配置)
- ✅ 日志和监控系统

#### 1.2 数据管线
- ✅ LibriSpeech 数据加载
- ✅ Whisper 特征提取 (log-mel)
- ✅ 数据批处理和 padding
- ✅ 小规模数据集 (1000 samples)

#### 1.3 模型实现
- ✅ Teacher: whisper-tiny 封装
- ✅ Student: 2层 Transformer decoder
- ✅ 冻结 encoder，训练 decoder

#### 1.4 训练系统
- ✅ CE-only baseline
- ✅ Classical KD (Hinton)
- ✅ 训练/验证/评估循环
- ✅ WER/CER 计算

#### 1.5 文档
- ✅ README
- ✅ 项目结构文档
- ✅ 快速开始指南
- ✅ Phase 1 完成报告

### 关键成果

| 指标 | 数值 |
|------|------|
| 代码文件 | 14 个 Python 文件 |
| 代码行数 | ~1,554 行 |
| 配置文件 | 2 个 (CE + KD) |
| 文档 | 8 个 Markdown |
| Linter 错误 | 0 |

### 验证结果

在 1000 样本的小数据集上：
- CE baseline: 可训练，收敛
- KD baseline: 可训练，预期优于 CE
- 代码质量: 无错误，模块化设计

---

## Phase 2: 标准 KD 扩展 🚧 (进行中)

### 目标
在 Phase 1 的基础上放大规模，完成 Classical KD 的完整实验，输出课程所需的全部指标：

1. Teacher：tiny → small/base（更高质量的 logits）
2. 数据：1000 samples → 50–100 小时（train-clean-100）
3. Student：S2 / S3 / S4（2/3/4 层 decoder）
4. 训练：CE-only vs KD & 温度 T 最优
5. 指标：参数量、FLOPs、RTF、WER/CER + 速度–准确率曲线

### 实施计划

#### 2.1 基础扩展 (Week 4)

**数据扩展**:
- [ ] 使用 LibriSpeech train-clean-100 (≈100h)
- [ ] 数据划分：train/val/test = 90/5/5
- [ ] 数据统计与质量分析

**Teacher 升级**:
- [ ] whisper-tiny → whisper-small（必要时尝试 whisper-base）
- [ ] 记录 teacher WER/CER、RTF、显存占用

**Student 变体**:
- [ ] S2: 2-layer decoder（轻量、速度快）
- [ ] S3: 3-layer decoder（性能/速度平衡）
- [ ] S4: 4-layer decoder（追求最佳性能）

**交付物**:
- `configs/data_100h.yaml`
- `configs/teacher_small.yaml`
- S2/S3/S4 训练配置
- Teacher 对比报告

#### 2.2 Classical KD 训练 (Week 4-5)

**实验矩阵**:
| Teacher | Student | Loss | 目的 |
|---------|---------|------|------|
| small | S2 | CE-only | baseline |
| small | S2 | KD | 对比 CE vs KD |
| small | S3 | KD | 性能/速度平衡 |
| small | S4 | KD | 追求最佳准确率 |

**任务**:
- [ ] 训练 CE-only baseline (S2)
- [ ] 训练 KD (S2/S3/S4)
- [ ] 保存最佳模型、记录训练日志

**交付物**:
- 批量训练脚本 (`scripts/train_phase2_all.sh`)
- `outputs/*/train.log`、`best_model.pt`
- 初步 WER/CER 统计

#### 2.3 温度实验 (Week 5)

**设置**:
- Teacher = whisper-small，Student = S3
- T ∈ {1, 2, 4, 6, 8}

**任务**:
- [ ] 运行 5 组温度实验
- [ ] 分析温度对 WER/CER、收敛速度的影响
- [ ] 选择最优温度 T*

**交付物**:
- `configs/temperature/*.yaml`
- 温度分析脚本 (`scripts/analyze_temperature.py`)
- `results/temperature_analysis.{png,pdf}`

#### 2.4 评估与可视化 (Week 6)

**任务**:
- [ ] 统一评估所有模型（测试集 WER/CER）
- [ ] 统计参数量、FLOPs、RTF
- [ ] 绘制 Speed-Accuracy Curve、CE vs KD 对比
- [ ] 汇总表格（CSV + LaTeX）
- [ ] 生成 Phase 2 结果报告

**交付物**:
- `results/model_comparison.{csv,tex}`
- `results/speed_accuracy_curve.{png,pdf}`
- `doc/phase2_results.md`

#### 2.5 (可选) 规模再扩展

若时间/算力允许，可额外尝试：
- Teacher = whisper-base
- 数据 = train-clean-360 子集
- 仅跑 CE vs KD，验证 scaling 趋势

---

## 时间线总览

```
Week 1-2 (已完成): Phase 1 基础设施
├── 数据管线
├── 模型实现
├── CE/KD baseline
└── 文档

Week 3 (已完成): Phase 1 收尾
├── 运行 test_setup
├── 完成 Phase 1 文档
└── 规划 Phase 2

Week 4: Phase 2 扩展
├── 数据扩展到 100h
├── Teacher 升级为 small
├── S2/S3/S4 配置与测试
└── 启动 CE/KD 训练

Week 5: Phase 2 深入
├── 完成所有 CE/KD 训练
├── 温度实验 (T=1/2/4/6/8)
├── 收集初步指标
└── 准备中期展示

Week 6: Phase 2 完成
├── 统一评估 + Speed-Accuracy 曲线
├── 生成对比表格/图
├── 撰写 Phase 2 报告
└── 准备 poster / slides

Week 7+: Phase 3 (DKD 研究)
├── Loss 设计 (TCKD/NCKD)
├── Ablation Study
└── CE/KD/DKD 对比
```

---

## 技术栈总览

### 核心框架
- **PyTorch**: 深度学习框架
- **HuggingFace**: Whisper 模型 + datasets
- **torchaudio**: 音频处理

### 数据处理
- **datasets**: LibriSpeech 加载
- **librosa**: 音频重采样
- **soundfile**: 音频读取

### 评估与可视化
- **jiwer**: WER/CER 计算
- **matplotlib**: 绘图
- **pandas**: 数据分析
- **numpy**: 数值计算

### 工程工具
- **PyYAML**: 配置管理
- **tqdm**: 进度条
- **logging**: 日志系统

---

## 课程要求对照表

| 课程要求 | Phase 2 对应 | 状态 | 文件位置 |
|----------|-------------|------|---------|
| Student 使用 soft labels | Classical KD（S2/S3/S4） | 🚧 进行中 | `src/training/train_kd.py` |
| 不同温度 T 的影响 | T={1,2,4,6,8} 扫描 | ⏳ 即将开始 | `doc/temperature_analysis.md` (计划) |
| 不同架构对比 | S2 vs S3 vs S4 | 🚧 进行中 | `doc/student_comparison.md` (计划) |
| KD vs No-KD 对比 | CE-only vs KD (S2) | 🚧 进行中 | `doc/phase2_results.md` (计划) |
| 完整实验报告 | Phase 2 结果 + poster | ⏳ Week 6 | `doc/phase2_results.md` / `doc/final_report.md` |
| 代码实现 | Phase 1 基础设施 | ✅ 完成 | `src/*` |
| Baseline 训练 | CE + classical KD (tiny) | ✅ 完成 | `src/training/train_{ce,kd}.py` |

---

## 预期成果

### 定量指标

**模型对比（Phase 2 预计）**:
| 模型 | Params | WER | RTF | 说明 |
|------|--------|-----|-----|------|
| Teacher (small) | 244M | ~5% | 0.2 | 上限参考 |
| Student S2 (CE) | 1.5M | ~12% | 0.05 | Baseline |
| Student S2 (KD) | 1.5M | ~10% | 0.05 | KD 带来 +2% |
| Student S3 (KD) | 2.3M | ~8% | 0.08 | 性能/速度平衡 |
| Student S4 (KD) | 3.1M | ~7.3% | 0.11 | 精度优先 |

**CE vs KD 对比（S2 示例）**:
| Loss | 预期 WER | CER | 备注 |
|------|---------|-----|------|
| CE-only | 12.0% | 6.0% | Hard labels |
| CE + KD (T=4) | 10.0% | 5.2% | Soft labels, 与课程要求一一对应 |

**温度分析预期**:
- T=1: 接近 hard labels
- T=2-4: 最优区间
- T=8-16: 过度平滑

> Phase 3（DKD）计划在 Phase 2 完成后启动，目标是在上述 KD 结果基础上再降低 0.5–1.0% WER。

### 定性成果

**代码质量**:
- ✅ 模块化设计
- ✅ 完整注释
- ✅ 可复现性强
- ✅ 文档完善

**实验完整性**:
- ✅ 多种 baseline 对比
- ✅ 完整 ablation study
- ✅ 超参数分析
- ✅ 可视化丰富

**创新点**:
- ✅ Decoder-only distillation（冻结 encoder + 轻量 decoder）
- ✅ 速度–准确率权衡分析
- ✅ Phase 3 计划引入 DKD（TCKD/NCKD）作为额外创新

---

## 风险与应对

### 风险 1: 计算资源不足

**影响**: 无法完成所有大规模实验

**应对**:
1. 优先完成核心实验 (ablation + 温度)
2. 使用梯度累积减少显存占用
3. 减少数据量但保证实验完整性

### 风险 2: 温度/超参选择不佳

**影响**: KD 改善不足或训练不稳定

**应对**:
1. 进行系统的 T 扫描（1/2/4/6/8）
2. 结合验证集 WER + loss 曲线共同判断
3. 适当调整 α（CE/KD 权重）与 batch size

### 风险 3: 时间不够

**影响**: 无法完成所有实验

**应对**:
1. 聚焦核心实验 (S3 + ablation)
2. S2/S4 作为补充
3. 大规模实验标记为 optional

---

## 里程碑

### Milestone 1: Phase 1 完成 ✅
- 日期: Week 3
- 标志: CE/KD baseline 可运行
- 交付: Phase 1 完成报告

### Milestone 2: 数据扩展完成
- 日期: Week 4 Day 3
- 标志: 100h 数据加载成功
- 交付: 数据统计报告

### Milestone 3: S2/S3/S4 训练完成
- 日期: Week 4 Day 7
- 标志: 三种 student 都训练完成
- 交付: Student 对比报告

### Milestone 4: 温度实验完成
- 日期: Week 5 Day 5
- 标志: T=1/2/4/6/8 实验全部结束
- 交付: Temperature 分析图/表

### Milestone 5: Speed-Accuracy 曲线完成
- 日期: Week 6 Day 3
- 标志: S2/S3/S4 的 WER、RTF、FLOPs 全部统计
- 交付: `results/model_comparison.*`、`speed_accuracy_curve.*`

### Milestone 6: Phase 2 报告提交
- 日期: Week 6 Day 7
- 标志: Phase 2 结果、图表、poster 草稿齐备
- 交付: `doc/phase2_results.md`、slides/poster 草稿

### Milestone 7: Phase 3 启动（可选）
- 日期: Week 7+
- 标志: DKD 设计评审通过
- 交付: Phase 3 计划（DKD loss、实验矩阵）

---

## 资源需求

### 计算资源
- **GPU**: A100 / V100 / 3090 或同等
- **显存**: 16GB+ (推荐 24GB+)
- **存储**: 100GB+ (数据集 + 模型 + 日志)

### 时间估算
- **数据准备**: 2-3 小时
- **S2/S3/S4 训练 (CE/KD)**: 每个 6-8 小时（共 ~24 小时）
- **温度实验 (5 组)**: 30-40 小时
- **评估 & 可视化**: 10-15 小时
- **总计**: ~80-100 GPU 小时

### 人力投入
- **编码**: 40-60 小时
- **实验**: 30-40 小时 (等待时间)
- **分析**: 20-30 小时
- **报告**: 20-30 小时
- **总计**: ~110-160 人时

---

## 下一步行动

### 立即执行 (本周)
1. [ ] 扩展数据到 100h，完成统计与划分
2. [ ] 升级 teacher = whisper-small，跑通推理
3. [ ] 整理 S2/S3/S4 配置与脚本

### 短期 (Week 4)
1. [ ] 训练 CE-only (S2)
2. [ ] 训练 KD-S2/S3/S4
3. [ ] 记录训练日志与初步指标

### 中期 (Week 5)
1. [ ] 运行温度实验 (T=1/2/4/6/8)
2. [ ] 汇总 CE vs KD 对比
3. [ ] 准备中期展示材料

### 长期 (Week 6)
1. [ ] 统一评估 + Speed-Accuracy 曲线
2. [ ] 生成所有图表/表格与 Phase2 报告
3. [ ] 准备 poster / slides
4. [ ] (可选) 规划 Phase 3 DKD 研究

---

## 总结

这是一个**研究级别的 ASR 知识蒸馏项目**，涵盖：

✅ **Phase 1** (已完成):
- 基础设施完善
- CE/KD baseline 验证
- 代码质量高

🚧 **Phase 2** (进行中):
- 规模扩展 (small teacher + 100h)
- Classical KD 全量实验（S2/S3/S4 + 温度）
- 速度–准确率曲线 & Phase2 报告

🧪 **Phase 3** (规划中):
- DKD Loss（TCKD/NCKD）实现
- CE/KD/DKD 全面对比

📊 **预期成果**:
- 论文级别的图表和表格
- 深入的 ablation 分析
- 超越课程基本要求

让我们开始 Phase 2 的实施吧！🚀

