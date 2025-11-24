# Whisper-Large T=1 训练指南

本指南说明如何运行 whisper-large 作为教师、温度 T=1 的 6 个实验。

## 📋 实验清单

### Classical KD (3 个实验)
1. **large_kd_s2_t1** - S2 学生 (2层 decoder)
2. **large_kd_s3_t1** - S3 学生 (3层 decoder)
3. **large_kd_s4_t1** - S4 学生 (4层 decoder)

### DKD (3 个实验)
4. **large_dkd_s2_t1** - S2 学生 (2层 decoder)
5. **large_dkd_s3_t1** - S3 学生 (3层 decoder)
6. **large_dkd_s4_t1** - S4 学生 (4层 decoder)

## 🚀 运行方式

### 方式 1: 批量运行所有实验（推荐）

```bash
cd /home/zz359/workspace-whisper
bash scripts/train_large_t1_all.sh
```

**特点：**
- ✅ 自动按顺序运行全部 6 个实验
- ✅ 每个实验的日志单独保存到 `logs/` 目录
- ✅ 整体进度保存到批量日志文件
- ✅ 如果某个实验失败，会记录并继续下一个
- ⏱️ 预计总耗时：**60-90 小时**（每个实验 10-15 小时）

**日志位置：**
- 批量日志: `logs/batch_large_t1_YYYYMMDD_HHMMSS.log`
- 单个实验日志: `logs/large_kd_s2_t1_YYYYMMDD_HHMMSS.log` 等

### 方式 2: 单独运行每个实验

#### Classical KD 实验

```bash
cd /home/zz359/workspace-whisper

# S2 学生
python src/training/train_kd.py --config configs/large_kd_s2_t1.yaml

# S3 学生
python src/training/train_kd.py --config configs/large_kd_s3_t1.yaml

# S4 学生
python src/training/train_kd.py --config configs/large_kd_s4_t1.yaml
```

#### DKD 实验

```bash
cd /home/zz359/workspace-whisper

# S2 学生
python src/training/train_dkd.py --config configs/large_dkd_s2_t1.yaml

# S3 学生
python src/training/train_dkd.py --config configs/large_dkd_s3_t1.yaml

# S4 学生
python src/training/train_dkd.py --config configs/large_dkd_s4_t1.yaml
```

### 方式 3: 后台运行（适合长时间训练）

```bash
cd /home/zz359/workspace-whisper

# 使用 nohup 在后台运行，即使断开 SSH 也继续
nohup bash scripts/train_large_t1_all.sh > logs/nohup_large_t1.log 2>&1 &

# 查看进程
ps aux | grep train_large_t1_all.sh

# 实时查看日志
tail -f logs/nohup_large_t1.log
```

### 方式 4: 使用 tmux/screen（推荐远程训练）

```bash
# 创建新的 tmux 会话
tmux new -s large_t1_training

# 在 tmux 中运行训练
cd /home/zz359/workspace-whisper
bash scripts/train_large_t1_all.sh

# 断开 tmux（训练继续）: Ctrl+B 然后按 D

# 重新连接
tmux attach -t large_t1_training

# 查看所有会话
tmux ls
```

## 📊 查看训练进度

### 实时查看当前实验日志

```bash
# 找到最新的日志文件
ls -lt logs/large_*_*.log | head -1

# 实时查看
tail -f logs/large_kd_s2_t1_20251122_123456.log
```

### 查看批量训练进度

```bash
# 查看批量日志
tail -f logs/batch_large_t1_*.log
```

### 检查训练输出

```bash
# 查看所有 large 实验的输出目录
ls -lh outputs/large_*/

# 查看某个实验的 checkpoint
ls -lh outputs/large_kd_s2_t1/
```

## 📁 输出位置

每个实验的输出保存在：

```
outputs/
├── large_kd_s2_t1/
│   ├── best_model.pt          # 最佳模型 (WER 最低)
│   ├── checkpoint_2000.pt     # Step 2000 的 checkpoint
│   ├── checkpoint_4000.pt     # Step 4000 的 checkpoint
│   └── train.log              # 训练日志副本
├── large_kd_s3_t1/
├── large_kd_s4_t1/
├── large_dkd_s2_t1/
├── large_dkd_s3_t1/
└── large_dkd_s4_t1/
```

## 🔧 配置说明

所有实验使用相同的超参数：

### 模型配置
- **教师模型**: openai/whisper-large (1550M 参数)
- **学生 hidden size**: 1024
- **学生 attention heads**: 16
- **Dropout**: 0.1

### 数据配置
- **数据集**: LibriSpeech ASR clean train.100
- **最大音频长度**: 10 秒
- **Batch size**: 16
- **训练/验证/测试**: 90%/5%/5%

### 训练配置
- **Epochs**: 15
- **Learning rate**: 1e-4
- **Weight decay**: 0.01
- **Warmup steps**: 500
- **混合精度**: FP16
- **评估间隔**: 每 1000 steps
- **保存间隔**: 每 2000 steps

### KD 超参数
- **温度 (T)**: 1.0
- **Classical KD alpha**: 0.5 (CE 和 KD 各占 50%)
- **DKD alpha**: 2.0 (TCKD 权重)
- **DKD beta**: 0.3 (NCKD 权重)
- **DKD gamma**: 0.6 (CE 权重)
- **DKD top-k**: 64 (非目标类保留数量)

## ⚠️ 注意事项

### GPU 内存要求
- **whisper-large** 比 whisper-small 大很多 (~1550M vs ~244M)
- **建议 GPU 显存**: ≥ 24GB (如 RTX 3090, A5000, V100)
- 如果显存不足，可以：
  - 减小 `batch_size` (从 16 改为 8 或 4)
  - 关闭 FP16 混合精度 (改为 `fp16: false`)

### 训练时间估计
- **每个实验**: 约 10-15 小时 (取决于 GPU)
- **全部 6 个实验**: 约 60-90 小时
- **建议**: 使用 tmux/screen 或后台运行

### 磁盘空间
- 每个模型的 checkpoint: ~300-500MB
- 每个实验保存 3 个文件 (best + 2 个 checkpoint)
- **总共需要**: 约 10-15GB

## 🔍 故障排查

### CUDA Out of Memory

```bash
# 修改配置文件中的 batch_size
# 例如: configs/large_kd_s2_t1.yaml
data:
  batch_size: 8  # 从 16 改为 8
```

### 检查 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者使用 gpustat
pip install gpustat
gpustat -i 1
```

### 查看失败日志

```bash
# 查看批量训练日志中的错误信息
grep -i "error\|failed\|exception" logs/batch_large_t1_*.log

# 查看具体实验的错误
grep -i "error\|traceback" logs/large_kd_s2_t1_*.log
```

## 📈 训练完成后

### 查看所有结果

运行解析脚本（之前创建的）来汇总所有实验结果：

```bash
cd /home/zz359/workspace-whisper
python - <<'PY'
# (使用之前的解析脚本)
# 会自动生成:
# - analysis/runs_table.md
# - analysis/runs_table.csv
# - analysis/runs_data.json
PY
```

### 比较 small vs large 教师

对比 whisper-small 和 whisper-large 的效果：

```bash
# 过滤 large 实验
grep "large_" analysis/runs_table.csv

# 过滤 small 实验（之前的）
grep "small_" analysis/runs_table.csv
```

## 🎯 预期结果

根据 Phase 2 计划，预期：
- ✅ S2/S3/S4 随着层数增加，WER 应该降低
- ✅ KD 应该比 CE-only 效果更好
- ✅ Large 教师应该比 Small 教师提供更好的知识迁移
- ✅ T=1 是经典设置，可以作为后续温度扫描的基准

祝训练顺利！🚀

