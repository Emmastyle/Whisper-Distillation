# Phase 2 实施进度

## 已完成

### ✓ Step 1: 数据与 Teacher 扩展准备
- [x] 创建 100h 数据配置 (`configs/data_100h.yaml`)
- [x] 创建 Teacher 配置 (`configs/teacher_small.yaml`)
- [x] 数据分析完成 (`outputs/data_100h_statistics.txt`)
  - 原始样本数: 28,539
  - 有效样本数（≤10s）: 5,366
  - 总时长: 9.17 小时
  - 训练集: 8.23 小时（4,829 样本）
  - 验证集: 0.49 小时（268 样本）
  - 测试集: 0.45 小时（269 样本）

### ✓ Step 2: 配置文件创建
- [x] CE 配置：`small_ce_s2/s3/s4.yaml`
- [x] KD 配置：`small_kd_s2/s3/s4.yaml`（T=4）
- [x] 批量训练脚本：`train_phase2_all.sh`
- [x] 温度扫描脚本：`train_temperature_sweep.sh`（T=2,3,4,5,6,8）
- [x] 评估脚本：`evaluate_all_models.py`

### ✓ 辅助脚本
- [x] 数据分析脚本：`analyze_data_100h.py`
- [x] Teacher 验证脚本：`test_teacher_small.py`
- [x] Phase 2 设置验证：`test_phase2_setup.py`

### ✓ 依赖更新
- [x] 添加 `pandas`, `matplotlib`, `seaborn` 到 `requirements.txt`
- [x] 修复 `TeacherWhisperModel` 别名问题

## 进行中

### ⏳ Teacher Model 验证
- [ ] 运行 `test_teacher_small.py`
- [ ] 生成 `outputs/teacher_small_report.txt`

## 待完成

### 📋 Step 2: Student 模型训练
- [ ] 运行 CE-only 训练（S2/S3/S4）
- [ ] 运行 KD 训练（S2/S3/S4, T=4）

### 📋 Step 3: 温度扫描实验
- [ ] S3 模型温度扫描（T=2,3,4,5,6,8）
- [ ] 分析温度对 WER 的影响

### 📋 Step 4: 统一评估
- [ ] 评估所有模型
- [ ] 生成对比表格（CSV + LaTeX）
- [ ] 生成速度-准确率曲线
- [ ] 生成温度对比图

### 📋 Step 5: 文档与交付
- [ ] 撰写 `doc/phase2_results.md`
- [ ] 更新 `PROJECT_ROADMAP.md`
- [ ] 准备 poster/slides 素材

## 配置文件列表

### 数据配置
- `configs/data_100h.yaml` - LibriSpeech train-clean-100

### Teacher 配置
- `configs/teacher_small.yaml` - whisper-small (244M params)

### Student 配置 (CE-only)
- `configs/small_ce_s2.yaml` - 2层 decoder
- `configs/small_ce_s3.yaml` - 3层 decoder
- `configs/small_ce_s4.yaml` - 4层 decoder

### Student 配置 (KD, T=4)
- `configs/small_kd_s2.yaml` - 2层 decoder, T=4
- `configs/small_kd_s3.yaml` - 3层 decoder, T=4
- `configs/small_kd_s4.yaml` - 4层 decoder, T=4

### 温度扫描配置（自动生成）
- `configs/temp/small_kd_s3_t{2,3,4,5,6,8}.yaml`

## 脚本列表

### 训练脚本
- `scripts/train_phase2_all.sh` - 批量训练所有模型
- `scripts/train_temperature_sweep.sh` - 温度扫描实验

### 评估脚本
- `scripts/evaluate_all_models.py` - 统一评估所有模型
- `scripts/test_teacher_small.py` - Teacher model 基线测试

### 分析脚本
- `scripts/analyze_data_100h.py` - 数据集统计分析

### 验证脚本
- `test_phase2_setup.py` - Phase 2 设置验证

## 输出文件

### 数据统计
- `outputs/data_100h_statistics.txt` ✓

### Teacher 报告
- `outputs/teacher_small_report.txt` (待生成)
- `outputs/teacher_small_samples.txt` (待生成)

### 训练输出
- `outputs/small_ce_s2/` (待训练)
- `outputs/small_ce_s3/` (待训练)
- `outputs/small_ce_s4/` (待训练)
- `outputs/small_kd_s2/` (待训练)
- `outputs/small_kd_s3/` (待训练)
- `outputs/small_kd_s4/` (待训练)
- `outputs/small_kd_s3_t{2,3,5,6,8}/` (待训练)

### 评估结果
- `results/model_comparison.csv` (待生成)
- `results/model_comparison.tex` (待生成)
- `results/speed_accuracy_curve.png` (待生成)
- `results/speed_accuracy_curve.pdf` (待生成)
- `results/temperature_comparison.png` (待生成)
- `results/temperature_comparison.pdf` (待生成)

## 注意事项

1. **数据量限制**：由于 `max_duration=10s` 限制，实际使用数据约9.17小时，不是完整的100小时
2. **显存需求**：whisper-small 需要较大显存，batch_size 设为 16
3. **训练时间**：完整训练所有模型预计需要较长时间
4. **温度扫描**：先完成基础训练再进行温度扫描实验

