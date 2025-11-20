## 一、Phase 1 的任务计划（人类视角，一步一步）

**Phase 1 总目标：**
在小数据集上搭建一个最小可跑的代码库，完成：

1. Whisper-tiny 作为 **teacher encoder + decoder**（只推理，不训练）。
2. Student：复用 teacher encoder（冻结），自己定义一个小的 **decoder-only student**。
3. 先做 **CE-only baseline**，能训练、能评估 WER/CER。
4. 然后加上 **standard KD（Hinton）**，验证 KD 在小数据集上的提升。
5. 所有东西封装成一个清晰的 repo 结构，后面 Phase 2 / DKD 可以直接在此基础上改。

下面是按“任务顺序”的 Phase 1 计划：

---

### Step 0：技术栈 & 基本假设

* 使用 **Python 3.10+**
* 深度学习框架：**PyTorch**
* 使用 **HuggingFace transformers + datasets + torchaudio**
* 使用单卡 GPU（A100 / V100 / 3090 之类）为假设环境
* 目标训练时间：在小数据集上，**1–2 小时内能跑完若干 epoch + 出结果**

---

### Step 1：仓库初始化 & 依赖

**任务内容：**

1. 创建一个 repo 结构（大致）：

   * `configs/`：yaml/json 配置文件
   * `src/`

     * `data/`：数据集与 DataLoader
     * `models/`：teacher wrapper + student decoder 定义
     * `training/`：训练循环（CE / KD）、损失函数
     * `utils/`：日志、metrics（WER/CER）、配置解析
   * `scripts/`：一键运行脚本（train_ce.sh, train_kd.sh）
   * `requirements.txt` 或 `pyproject.toml`

2. 依赖（大概）：

   * `torch`, `torchaudio`
   * `transformers`
   * `datasets`
   * `jiwer`（算 WER/CER）
   * `pyyaml` / `omegaconf`
   * `tqdm`

**Phase 1 输出：**

* `README.md` 简要描述：

  * 如何安装依赖
  * 如何运行 CE baseline / KD baseline

---

### Step 2：数据集 & 特征 Pipeline

**目标：**在 5–10 小时的 LibriSpeech 子集上构建 dataloader。

**任务内容：**

1. 使用 `datasets` 加载 LibriSpeech：

   * 默认使用 `train.clean.100`，再从里面抽样若干小时（比如前 N 条）。
2. 对每条样本：

   * 使用 torchaudio 读音频并 resample 到 Whisper 需要的采样率（16k）。
   * 使用 HF 的 Whisper feature extractor 计算 **log-mel spectrogram**。
3. 文本处理：

   * 使用 Whisper 的 tokenizer 对 transcript 做 tokenization。
   * 处理特殊 token（BOS, EOS, PAD 等）。
4. 封装成：

   * `WhisperASRDataset`
   * `DataLoader`（支持 `collate_fn`，padding 到 batch max length）。

**Phase 1 输出：**

* `src/data/dataset.py`：定义 dataset 与 collate
* 一个 config：控制

  * 采样多少条数据
  * 最大时长（比如 ≤ 10s）
  * batch size

---

### Step 3：Teacher & Student 模型定义

**目标：**
实现一个“冻结 encoder + 可训练 student decoder”的模型结构。

**任务内容：**

1. **Teacher wrapper：**

   * 使用 `WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")`
   * 封装成 `TeacherWhisper` 类：

     * 提供 `encode(audio_features)` → encoder hidden states
     * 提供 `decode(encoder_outputs, decoder_input_ids)` → logits（推理用，不做梯度）

2. **Student 模型：**

   * 复用 teacher encoder 输出：

     * 直接传入 teacher encoder 的 `encoder_outputs` 作为 cross-attention 的 memory。
   * 定义一个自制的 **小 decoder**：

     * 例如 2-layer TransformerDecoder：

       * hidden size 与 teacher 保持兼容（方便 logits 投影）
       * 使用与 teacher 相同的 vocab size 和 embedding（甚至直接复制 embedding 权重）
   * 输出：

     * `logits` (batch, seq_len, vocab_size)
     * 支持 teacher forcing（输入为右移的 ground-truth tokens）

3. **CE-only 整体模型：**

   * 类似 `StudentASRModel`：

     * `forward(audio_features, decoder_input_ids, labels=None, teacher_outputs=None)`
     * 若 `labels` 给出，则计算 CE loss。

**Phase 1 输出：**

* `src/models/teacher.py`
* `src/models/student_decoder.py`
* `src/models/student_asr_model.py`

---

### Step 4：训练循环 – CE-only Baseline

**目标：**
在小数据集上跑通 student（只用 CE），能输出 WER/CER。

**任务内容：**

1. 实现 `train_ce.py`：

   * 读取配置（路径、超参）
   * 初始化：

     * teacher 模型（只用来 encode）
     * student 模型（encoder 复用 teacher，冻结；decoder 训练）
   * 优化器：AdamW
   * scheduler：可选 linear / cosine
   * loss：`nn.CrossEntropyLoss(ignore_index=pad_token_id)`

2. 训练 loop：

   * 对每个 batch：

     * `with torch.no_grad()` 调用 teacher 的 encoder 得到 `encoder_outputs`
     * 把 `encoder_outputs` 与 `decoder_input_ids` 传入 student，得到 logits
     * 计算 CE loss，反向传播、更新 decoder 参数。
   * 每 N step：

     * 打 log（loss）
     * 保存 checkpoint

3. 验证：

   * 实现 `evaluate.py` 或在 `train_ce.py` 中集成：

     * greedy decode（或 beam search 簡易版）
     * 把生成的 transcript 与 ground-truth 对比，用 `jiwer` 计算 WER/CER。

**Phase 1 输出：**

* 可以运行的命令：
  `python -m src.training.train_ce --config configs/ce_tiny_s2.yaml`
* 一次完整训练 + 验证，输出 WER/CER。

---

### Step 5：训练循环 – classical KD baseline（小规模）

**目标：**
在同一小数据集上增加一个 **KD 版本**，证明 distillation 有效果。

**任务内容：**

1. 实现 KD loss：

   * 温度 T
   * teacher logits `z_T` 和 student logits `z_S`
   * KD loss = KL(softmax(z_T/T) || softmax(z_S/T)) * T²（经典做法）
   * 总 loss： `L = alpha * CE + (1 - alpha) * KD`

2. 实现 `train_kd.py`：

   * 对每个 batch：

     * teacher：获得 encoder_outputs 和 decoder logits（teacher decoder 前向）
     * student：用相同输入做前向
     * CE loss（对 student logits vs ground truth）
     * KD loss（对 teacher vs student logits）
   * 组合 loss，训练 student decoder。

3. 小实验：

   * 固定一组超参（例如：

     * T = 2
     * alpha = 0.5
   * 与 CE-only baseline 对比：

     * 验证集 WER/CER
     * loss 收敛情况（可选）

**Phase 1 输出：**

* `src/training/train_kd.py`
* config：`configs/kd_tiny_s2.yaml`
* 报告一小节：**“CE vs KD on tiny + small dataset”** 的结果（Phase 1 结束标志）

---

