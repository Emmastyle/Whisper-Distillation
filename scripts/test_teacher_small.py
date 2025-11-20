#!/usr/bin/env python3
"""
Teacher Model 验证脚本：测试 whisper-small
输出：
- 模型参数量
- 推理速度（RTF）
- 显存占用
- WER/CER 基线
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import yaml
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm
import logging
from src.utils.metrics import calculate_wer, calculate_cer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_gpu_memory():
    """获取 GPU 显存占用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def test_teacher_model(model_name: str = "openai/whisper-small", num_samples: int = 100):
    """测试 teacher model"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型和处理器
    logger.info(f"加载 Teacher Model: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # 统计参数量
    total_params, trainable_params = count_parameters(model)
    logger.info(f"总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    
    # 加载测试数据
    logger.info("加载测试数据集...")
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        trust_remote_code=True
    )
    
    # 限制样本数
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    logger.info(f"测试样本数: {len(dataset)}")
    
    # 重置显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # 推理
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
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_audio_duration = sum(audio_durations) / len(audio_durations)
    
    gpu_memory = get_gpu_memory()
    
    # 输出报告
    logger.info("\n" + "="*60)
    logger.info(f"Teacher Model 验证报告 - {model_name}")
    logger.info("="*60)
    logger.info(f"\n【模型信息】")
    logger.info(f"  模型名称: {model_name}")
    logger.info(f"  总参数量: {total_params / 1e6:.2f}M")
    logger.info(f"  设备: {device}")
    
    logger.info(f"\n【性能指标】")
    logger.info(f"  WER: {wer:.2f}%")
    logger.info(f"  CER: {cer:.2f}%")
    
    logger.info(f"\n【速度指标】")
    logger.info(f"  平均推理时间: {avg_inference_time:.4f}s / 样本")
    logger.info(f"  平均音频时长: {avg_audio_duration:.4f}s / 样本")
    logger.info(f"  RTF (Real-Time Factor): {rtf:.4f}")
    logger.info(f"  速度: {'实时' if rtf < 1 else '非实时'} ({1/rtf:.2f}x)")
    
    logger.info(f"\n【资源占用】")
    logger.info(f"  GPU 显存峰值: {gpu_memory:.2f} MB")
    logger.info(f"  测试样本数: {len(dataset)}")
    logger.info("="*60 + "\n")
    
    # 保存报告
    output_file = "outputs/teacher_small_report.txt"
    os.makedirs("outputs", exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"Teacher Model 验证报告 - {model_name}\n")
        f.write("="*60 + "\n\n")
        f.write(f"【模型信息】\n")
        f.write(f"  模型名称: {model_name}\n")
        f.write(f"  总参数量: {total_params / 1e6:.2f}M\n")
        f.write(f"  设备: {device}\n\n")
        f.write(f"【性能指标】\n")
        f.write(f"  WER: {wer:.2f}%\n")
        f.write(f"  CER: {cer:.2f}%\n\n")
        f.write(f"【速度指标】\n")
        f.write(f"  平均推理时间: {avg_inference_time:.4f}s / 样本\n")
        f.write(f"  平均音频时长: {avg_audio_duration:.4f}s / 样本\n")
        f.write(f"  RTF (Real-Time Factor): {rtf:.4f}\n")
        f.write(f"  速度: {'实时' if rtf < 1 else '非实时'} ({1/rtf:.2f}x)\n\n")
        f.write(f"【资源占用】\n")
        f.write(f"  GPU 显存峰值: {gpu_memory:.2f} MB\n")
        f.write(f"  测试样本数: {len(dataset)}\n")
        f.write("="*60 + "\n")
    
    logger.info(f"验证报告已保存至: {output_file}")
    
    # 保存部分预测结果用于对比
    sample_output_file = "outputs/teacher_small_samples.txt"
    with open(sample_output_file, 'w') as f:
        f.write("Teacher Model 预测样本\n")
        f.write("="*60 + "\n\n")
        for i in range(min(10, len(predictions))):
            f.write(f"样本 {i+1}:\n")
            f.write(f"  参考: {references[i]}\n")
            f.write(f"  预测: {predictions[i]}\n")
            f.write(f"  WER: {calculate_wer([predictions[i]], [references[i]]):.4f}\n\n")
    
    logger.info(f"预测样本已保存至: {sample_output_file}")
    
    return {
        'model_name': model_name,
        'total_params': total_params,
        'wer': wer,
        'cer': cer,
        'rtf': rtf,
        'gpu_memory_mb': gpu_memory,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 Teacher Model")
    parser.add_argument(
        '--model',
        type=str,
        default='openai/whisper-small',
        help='Teacher model name'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='测试样本数'
    )
    
    args = parser.parse_args()
    
    results = test_teacher_model(args.model, args.num_samples)
    logger.info(f"\n✓ Teacher Model 验证完成！")

