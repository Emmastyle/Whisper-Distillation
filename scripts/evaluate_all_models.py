#!/usr/bin/env python3
"""
统一评估脚本：评估所有训练好的模型
输出：
- 参数量
- FLOPs（理论计算量）
- RTF（实时因子）
- WER/CER
- 对比表格与速度-准确率曲线
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import logging
from src.models.teacher import TeacherWhisperModel
from src.models.student_asr_model import StudentASRModel
from src.utils.metrics import calculate_wer, calculate_cer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, config):
    """估算 FLOPs（简化版本）"""
    # 这是一个简化的估算，实际 FLOPs 需要更复杂的分析
    # 基于模型配置估算每个前向传播的 FLOPs
    
    if 'student_decoder_layers' in config['model']:
        # Student model
        hidden_size = config['model']['student_hidden_size']
        num_layers = config['model']['student_decoder_layers']
        vocab_size = 51865  # Whisper vocab size
        
        # 估算：每层的 self-attention + FFN
        # Self-attention: 4 * hidden_size^2 (Q,K,V,O projections)
        # FFN: 2 * hidden_size * (4 * hidden_size)
        flops_per_layer = 4 * hidden_size**2 + 2 * hidden_size * (4 * hidden_size)
        decoder_flops = num_layers * flops_per_layer
        
        # Encoder (固定，来自 teacher)
        encoder_hidden = 768  # whisper-small encoder
        encoder_layers = 12
        encoder_flops = encoder_layers * (4 * encoder_hidden**2 + 2 * encoder_hidden * (4 * encoder_hidden))
        
        # Output projection
        output_flops = hidden_size * vocab_size
        
        total_flops = encoder_flops + decoder_flops + output_flops
        
    else:
        # Teacher model (whisper-small)
        hidden_size = 768
        num_layers = 12  # both encoder and decoder
        vocab_size = 51865
        
        flops_per_layer = 4 * hidden_size**2 + 2 * hidden_size * (4 * hidden_size)
        total_flops = 2 * num_layers * flops_per_layer + hidden_size * vocab_size
    
    return total_flops


def load_student_model(checkpoint_path, config):
    """加载 student model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    teacher = TeacherWhisperModel(config['model']['teacher_model'])
    student = StudentASRModel(
        teacher=teacher,
        student_decoder_layers=config['model']['student_decoder_layers'],
        student_decoder_heads=config['model']['student_decoder_heads'],
        student_hidden_size=config['model']['student_hidden_size'],
        dropout=config['model'].get('dropout', 0.1)
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        student.load_state_dict(checkpoint['model_state_dict'])
    else:
        student.load_state_dict(checkpoint)
    
    student.to(device)
    student.eval()
    
    return student, teacher.processor


def evaluate_model(model, processor, dataset, device):
    """评估单个模型"""
    predictions = []
    references = []
    audio_durations = []
    inference_times = []
    
    logger.info("开始推理...")
    with torch.no_grad():
        for item in tqdm(dataset, desc="Inference"):
            audio = item['audio']['array']
            sr = item['audio']['sampling_rate']
            text = item['text']
            
            # 音频时长
            duration = len(audio) / sr
            audio_durations.append(duration)
            
            # 预处理
            inputs = processor(
                audio,
                sampling_rate=sr,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(device)
            
            # 推理计时
            start_time = time.time()
            if hasattr(model, 'generate'):
                # Student model
                generated_ids = model.generate(input_features)
            else:
                # Teacher model
                generated_ids = model.generate(input_features)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 解码
            transcription = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            predictions.append(transcription)
            references.append(text)
    
    # 计算指标
    wer = calculate_wer(predictions, references)
    cer = calculate_cer(predictions, references)
    
    total_audio_duration = sum(audio_durations)
    total_inference_time = sum(inference_times)
    rtf = total_inference_time / total_audio_duration
    
    return {
        'wer': wer,
        'cer': cer,
        'rtf': rtf,
        'avg_inference_time': sum(inference_times) / len(inference_times),
        'avg_audio_duration': sum(audio_durations) / len(audio_durations),
    }


def evaluate_all_models(num_test_samples=100):
    """评估所有模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载测试数据
    logger.info("加载测试数据集...")
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test.clean",
        trust_remote_code=True
    )
    dataset = dataset.select(range(min(num_test_samples, len(dataset))))
    logger.info(f"测试样本数: {len(dataset)}")
    
    # 查找所有训练好的模型
    outputs_dir = Path("outputs")
    model_results = []
    
    # 1. 评估 Teacher Model (whisper-small)
    logger.info("\n" + "="*60)
    logger.info("评估 Teacher Model: whisper-small")
    logger.info("="*60)
    
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    teacher_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    teacher_model.to(device)
    teacher_model.eval()
    
    total_params, _ = count_parameters(teacher_model)
    teacher_config = {'model': {}}
    teacher_flops = estimate_flops(teacher_model, teacher_config)
    
    teacher_results = evaluate_model(teacher_model, processor, dataset, device)
    
    model_results.append({
        'Model': 'Teacher (whisper-small)',
        'Type': 'Teacher',
        'Layers': 12,
        'Params (M)': total_params / 1e6,
        'FLOPs (G)': teacher_flops / 1e9,
        'WER (%)': teacher_results['wer'] * 100,
        'CER (%)': teacher_results['cer'] * 100,
        'RTF': teacher_results['rtf'],
        'Speed (x)': 1 / teacher_results['rtf'],
    })
    
    del teacher_model  # 释放显存
    torch.cuda.empty_cache()
    
    # 2. 评估所有 Student Models
    for model_dir in sorted(outputs_dir.glob("small_*")):
        if not model_dir.is_dir():
            continue
        
        checkpoint_path = model_dir / "best_model.pt"
        if not checkpoint_path.exists():
            logger.warning(f"未找到模型文件: {checkpoint_path}")
            continue
        
        # 查找对应的配置文件
        model_name = model_dir.name
        config_path = Path(f"configs/{model_name}.yaml")
        
        # 如果是温度实验的模型，配置在 temp 目录
        if not config_path.exists():
            config_path = Path(f"configs/temp/{model_name}.yaml")
        
        if not config_path.exists():
            logger.warning(f"未找到配置文件: {config_path}")
            continue
        
        logger.info("\n" + "="*60)
        logger.info(f"评估模型: {model_name}")
        logger.info("="*60)
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 加载模型
        try:
            model, processor = load_student_model(checkpoint_path, config)
            total_params, _ = count_parameters(model)
            model_flops = estimate_flops(model, config)
            
            # 评估
            results = evaluate_model(model, processor, dataset, device)
            
            # 解析模型类型和层数
            model_type = "CE-only" if "_ce_" in model_name else "KD"
            layers = config['model']['student_decoder_layers']
            
            # 提取温度信息（如果有）
            temp_info = ""
            if "_t" in model_name and model_type == "KD":
                temp = model_name.split("_t")[-1]
                temp_info = f" (T={temp})"
            
            model_results.append({
                'Model': f"S{layers} {model_type}{temp_info}",
                'Type': model_type,
                'Layers': layers,
                'Params (M)': total_params / 1e6,
                'FLOPs (G)': model_flops / 1e9,
                'WER (%)': results['wer'] * 100,
                'CER (%)': results['cer'] * 100,
                'RTF': results['rtf'],
                'Speed (x)': 1 / results['rtf'],
            })
            
            logger.info(f"✓ {model_name} 评估完成")
            logger.info(f"  WER: {results['wer']:.2f}%")
            logger.info(f"  RTF: {results['rtf']:.4f}")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"✗ {model_name} 评估失败: {str(e)}")
            continue
    
    # 创建结果表格
    df = pd.DataFrame(model_results)
    
    # 排序：先按层数，再按类型
    df['Sort_Key'] = df['Layers'].astype(str) + '_' + df['Type']
    df = df.sort_values(['Layers', 'Type'])
    df = df.drop('Sort_Key', axis=1)
    
    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # CSV 格式
    csv_path = results_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ 结果已保存至: {csv_path}")
    
    # LaTeX 表格
    latex_path = results_dir / "model_comparison.tex"
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.2f"))
    logger.info(f"✓ LaTeX 表格已保存至: {latex_path}")
    
    # 打印表格
    print("\n" + "="*80)
    print("模型对比表格")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # 绘制速度-准确率曲线
    plot_speed_accuracy_curve(df, results_dir)
    
    return df


def plot_speed_accuracy_curve(df, output_dir):
    """绘制速度-准确率曲线"""
    
    # 只选择 S2/S3/S4 的 CE 和 KD（不包括温度扫描）
    df_main = df[~df['Model'].str.contains(r'\(T=', regex=True)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. WER vs Speed
    ax = axes[0]
    for model_type in ['CE-only', 'KD']:
        subset = df_main[df_main['Type'] == model_type]
        ax.plot(
            subset['Speed (x)'],
            subset['WER (%)'],
            marker='o',
            linewidth=2,
            markersize=10,
            label=model_type
        )
        
        # 添加层数标签
        for _, row in subset.iterrows():
            ax.annotate(
                f"S{row['Layers']}",
                (row['Speed (x)'], row['WER (%)']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10
            )
    
    # Teacher baseline
    teacher = df[df['Type'] == 'Teacher'].iloc[0]
    ax.scatter(
        teacher['Speed (x)'],
        teacher['WER (%)'],
        marker='*',
        s=300,
        c='red',
        label='Teacher',
        zorder=5
    )
    ax.annotate(
        'Teacher',
        (teacher['Speed (x)'], teacher['WER (%)']),
        textcoords="offset points",
        xytext=(0, -15),
        ha='center',
        fontsize=10
    )
    
    ax.set_xlabel('Speed (x Real-time)', fontsize=14)
    ax.set_ylabel('WER (%)', fontsize=14)
    ax.set_title('Speed vs Accuracy Trade-off', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 2. Params vs WER
    ax = axes[1]
    for model_type in ['CE-only', 'KD']:
        subset = df_main[df_main['Type'] == model_type]
        ax.plot(
            subset['Params (M)'],
            subset['WER (%)'],
            marker='o',
            linewidth=2,
            markersize=10,
            label=model_type
        )
        
        for _, row in subset.iterrows():
            ax.annotate(
                f"S{row['Layers']}",
                (row['Params (M)'], row['WER (%)']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=10
            )
    
    ax.scatter(
        teacher['Params (M)'],
        teacher['WER (%)'],
        marker='*',
        s=300,
        c='red',
        label='Teacher',
        zorder=5
    )
    ax.annotate(
        'Teacher',
        (teacher['Params (M)'], teacher['WER (%)']),
        textcoords="offset points",
        xytext=(0, -15),
        ha='center',
        fontsize=10
    )
    
    ax.set_xlabel('Parameters (M)', fontsize=14)
    ax.set_ylabel('WER (%)', fontsize=14)
    ax.set_title('Model Size vs Accuracy', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    png_path = output_dir / "speed_accuracy_curve.png"
    pdf_path = output_dir / "speed_accuracy_curve.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    logger.info(f"✓ 速度-准确率曲线已保存至:")
    logger.info(f"  - {png_path}")
    logger.info(f"  - {pdf_path}")
    
    # 如果有温度扫描结果，绘制温度对比图
    df_temp = df[df['Model'].str.contains(r'\(T=', regex=True)]
    if len(df_temp) > 0:
        plot_temperature_comparison(df_temp, output_dir)


def plot_temperature_comparison(df_temp, output_dir):
    """绘制温度对比图"""
    
    # 提取温度值
    df_temp = df_temp.copy()
    df_temp['Temperature'] = df_temp['Model'].str.extract(r'\(T=(\d+)\)')[0].astype(int)
    df_temp = df_temp.sort_values('Temperature')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. WER vs Temperature
    ax = axes[0]
    ax.plot(
        df_temp['Temperature'],
        df_temp['WER (%)'],
        marker='o',
        linewidth=2,
        markersize=10,
        color='blue'
    )
    ax.set_xlabel('Temperature (T)', fontsize=14)
    ax.set_ylabel('WER (%)', fontsize=14)
    ax.set_title('WER vs KD Temperature (S3 Student)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. RTF vs Temperature
    ax = axes[1]
    ax.plot(
        df_temp['Temperature'],
        df_temp['RTF'],
        marker='o',
        linewidth=2,
        markersize=10,
        color='green'
    )
    ax.set_xlabel('Temperature (T)', fontsize=14)
    ax.set_ylabel('RTF (Real-Time Factor)', fontsize=14)
    ax.set_title('RTF vs KD Temperature (S3 Student)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    png_path = output_dir / "temperature_comparison.png"
    pdf_path = output_dir / "temperature_comparison.pdf"
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    logger.info(f"✓ 温度对比图已保存至:")
    logger.info(f"  - {png_path}")
    logger.info(f"  - {pdf_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估所有训练好的模型")
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='测试样本数'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Phase 2 统一评估脚本")
    logger.info("="*60)
    
    df = evaluate_all_models(args.num_samples)
    
    logger.info("\n✓ 所有模型评估完成！")
    logger.info("结果文件:")
    logger.info("  - results/model_comparison.csv")
    logger.info("  - results/model_comparison.tex")
    logger.info("  - results/speed_accuracy_curve.png/pdf")
    if len(df[df['Model'].str.contains(r'\(T=', regex=True)]) > 0:
        logger.info("  - results/temperature_comparison.png/pdf")

