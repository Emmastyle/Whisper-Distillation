# Whisper Knowledge Distillation 项目路线图

## 总体概览

本项目分为两个主要阶段，从小规模验证到大规模 DKD 实验。

```
Phase 1 (已完成) → Phase 2 (待实施) → Final Report & Presentation
    ↓                    ↓                           ↓
 基线验证            DKD 核心实现              论文级成果
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

## Phase 2: DKD 核心实现 🚧 (待实施)

### 目标
扩展到大规模，实现 DKD，进行完整的 ablation study

### 实施计划

#### 2.1 扩展基础 (Week 4)

**数据扩展**:
- [ ] 使用完整 LibriSpeech train-clean-100 (100h)
- [ ] 优化数据加载效率
- [ ] 数据统计和分析

**Teacher 升级**:
- [ ] whisper-tiny (39M) → whisper-small (244M)
- [ ] 测试 teacher 性能 (WER 上限)
- [ ] 显存和速度优化

**多种 Student**:
- [ ] S2: 2层 decoder (~1.5M params)
- [ ] S3: 3层 decoder (~2.3M params)
- [ ] S4: 4层 decoder (~3.1M params)

**交付物**:
- 大规模配置文件
- Teacher 性能测试报告
- Student 对比配置

#### 2.2 Classical KD 扩展 (Week 4)

**任务**:
- [ ] 在 S2/S3/S4 上分别训练 classical KD
- [ ] 收集 WER/CER、参数量、RTF
- [ ] 绘制 speed-accuracy 曲线

**交付物**:
- 批量训练脚本
- 性能评估脚本
- Student 对比报告

#### 2.3 DKD 实现 (Week 5)

**核心任务**:
- [ ] 实现 TCKD Loss
- [ ] 实现 NCKD Loss
- [ ] 实现 DKD 训练脚本
- [ ] 配置超参数 (α, β, T)

**Ablation Study** (5个实验):
1. [ ] CE-only
2. [ ] CE + Classical KD
3. [ ] CE + TCKD only
4. [ ] CE + NCKD only
5. [ ] CE + DKD (TCKD + NCKD)

**交付物**:
- DKD Loss 实现
- DKD 训练脚本
- Ablation 配置文件 (5个)
- Ablation 结果报告

#### 2.4 温度实验 (Week 5)

**任务**:
- [ ] 在 T={1,2,4,8,16} 上测试 classical KD
- [ ] 在 T={1,2,4,8,16} 上测试 DKD
- [ ] 分析温度对性能的影响
- [ ] 找到最优温度

**交付物**:
- 温度扫描配置 (10个)
- 温度分析脚本
- 温度实验报告

#### 2.5 评估与可视化 (Week 6)

**任务**:
- [ ] 统一评估框架
- [ ] 性能分析 (FLOPs, RTF)
- [ ] 生成所有图表
- [ ] 生成所有表格
- [ ] 汇总实验结果

**图表**:
1. Speed-Accuracy Curve (WER vs params, WER vs RTF)
2. Ablation Comparison (柱状图)
3. Temperature Analysis (T vs WER)
4. Training Curves (loss, WER 随 epoch)
5. Loss Decomposition (CE/TCKD/NCKD)

**交付物**:
- 评估框架代码
- 所有图表 (PDF/PNG)
- 所有表格 (LaTeX/Markdown)
- 最终结果报告

#### 2.6 (可选) 大规模实验 (Week 6-7)

如果时间允许:
- [ ] whisper-base teacher
- [ ] train-clean-360 (360h)
- [ ] 验证 scaling 效果

---

## 时间线总览

```
Week 1-2 (已完成): Phase 1 基础设施
├── 数据管线
├── 模型实现
├── CE/KD baseline
└── 文档

Week 3 (已完成): Phase 1 完善
├── 测试验证
├── 文档完善
└── Phase 2 计划

Week 4: Phase 2 扩展
├── 数据扩展 (100h)
├── Teacher 升级 (small)
├── S2/S3/S4 实现
└── Classical KD 训练

Week 5: Phase 2 核心
├── DKD 实现
├── Ablation Study (5个实验)
├── 温度实验
└── 结果初步分析

Week 6: Phase 2 完善
├── 完整评估
├── 可视化
├── 结果汇总
└── 报告撰写

Week 7 (可选): 扩展实验
├── 更大规模
├── 更多对比
└── 论文润色
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

| 课程要求 | 实现方式 | 状态 | 文件位置 |
|----------|---------|------|---------|
| Student 使用 soft labels | DKD (TCKD+NCKD) | 🚧 Phase 2 | `src/training/losses.py` |
| 不同温度 T 的影响 | T={1,2,4,8,16} 扫描 | 🚧 Phase 2 | `doc/temperature_analysis.md` |
| 不同架构对比 | S2/S3/S4 对比 | 🚧 Phase 2 | `doc/student_comparison.md` |
| KD vs No-KD 对比 | Ablation study | 🚧 Phase 2 | `doc/dkd_ablation_results.md` |
| 完整实验报告 | 最终报告 + figures | 🚧 Phase 2 | `doc/final_report.md` |
| 代码实现 | 完整 codebase | ✅ Phase 1 | `src/*` |
| Baseline 训练 | CE + classical KD | ✅ Phase 1 | `src/training/train_{ce,kd}.py` |

---

## 预期成果

### 定量指标

**模型对比**:
| 模型 | Params | WER | RTF | 说明 |
|------|--------|-----|-----|------|
| Teacher (small) | 244M | ~5% | 0.2 | 上限参考 |
| Student S2 (CE) | 1.5M | ~12% | 0.05 | Baseline |
| Student S2 (KD) | 1.5M | ~10% | 0.05 | Classical KD |
| Student S3 (KD) | 2.3M | ~8% | 0.08 | 平衡点 |
| Student S3 (DKD) | 2.3M | ~7.5% | 0.08 | **Best** |
| Student S4 (DKD) | 3.1M | ~7% | 0.10 | 更强性能 |

**Ablation 预期**:
| 方法 | WER | 相对提升 |
|------|-----|---------|
| CE-only | 12.0% | Baseline |
| CE+KD | 10.0% | +2.0% |
| CE+TCKD | 11.0% | +1.0% |
| CE+NCKD | 9.5% | +2.5% |
| CE+DKD | 9.0% | +3.0% |

**温度分析预期**:
- T=1: 接近 hard labels
- T=2-4: 最优区间
- T=8-16: 过度平滑

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
- ✅ DKD 在 ASR 上的应用
- ✅ Decoder-only distillation
- ✅ 系统的实验设计

---

## 风险与应对

### 风险 1: 计算资源不足

**影响**: 无法完成所有大规模实验

**应对**:
1. 优先完成核心实验 (ablation + 温度)
2. 使用梯度累积减少显存占用
3. 减少数据量但保证实验完整性

### 风险 2: DKD 效果不明显

**影响**: 与 classical KD 差异小

**应对**:
1. 调整超参数 (α, β, T)
2. 尝试不同数据规模
3. 深入分析 TCKD vs NCKD 的作用

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

### Milestone 4: DKD 实现完成
- 日期: Week 5 Day 2
- 标志: DKD Loss 实现并通过测试
- 交付: DKD 实现代码

### Milestone 5: Ablation 完成
- 日期: Week 5 Day 5
- 标志: 5个 ablation 实验完成
- 交付: Ablation 结果报告

### Milestone 6: 温度实验完成
- 日期: Week 5 Day 7
- 标志: 温度扫描实验完成
- 交付: 温度分析报告

### Milestone 7: Phase 2 完成
- 日期: Week 6 Day 7
- 标志: 所有图表和报告生成
- 交付: 完整项目报告

---

## 资源需求

### 计算资源
- **GPU**: A100 / V100 / 3090 或同等
- **显存**: 16GB+ (推荐 24GB+)
- **存储**: 100GB+ (数据集 + 模型 + 日志)

### 时间估算
- **数据准备**: 2-3 小时
- **S2/S3/S4 训练**: 每个 6-8 小时
- **Ablation (5个)**: 30-40 小时
- **温度实验 (10个)**: 60-80 小时
- **总计**: ~150-200 GPU 小时

### 人力投入
- **编码**: 40-60 小时
- **实验**: 30-40 小时 (等待时间)
- **分析**: 20-30 小时
- **报告**: 20-30 小时
- **总计**: ~110-160 人时

---

## 下一步行动

### 立即执行 (本周)
1. [ ] 阅读 DKD 论文，理解 TCKD/NCKD
2. [ ] 设计 DKD Loss 接口
3. [ ] 准备数据扩展到 100h

### 短期 (Week 4)
1. [ ] 实现数据扩展
2. [ ] 升级到 whisper-small
3. [ ] 实现 S3/S4
4. [ ] 训练 classical KD baseline

### 中期 (Week 5)
1. [ ] 实现 DKD Loss
2. [ ] 运行 ablation study
3. [ ] 运行温度实验
4. [ ] 初步结果分析

### 长期 (Week 6-7)
1. [ ] 完整评估
2. [ ] 生成所有图表
3. [ ] 撰写最终报告
4. [ ] 准备 presentation

---

## 总结

这是一个**研究级别的 ASR 知识蒸馏项目**，涵盖：

✅ **Phase 1** (已完成):
- 基础设施完善
- CE/KD baseline 验证
- 代码质量高

🚧 **Phase 2** (进行中):
- 规模扩展 (small teacher + 100h)
- DKD 创新实现
- 完整实验设计

📊 **预期成果**:
- 论文级别的图表和表格
- 深入的 ablation 分析
- 超越课程基本要求

让我们开始 Phase 2 的实施吧！🚀

