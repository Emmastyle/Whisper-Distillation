#!/usr/bin/env python3
"""
数据统计脚本：分析 LibriSpeech train-clean-100 数据集
输出统计报告，包括：
- 总样本数
- 时长分布
- 文本长度分布
- 历史训练/验证划分统计
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_dataset(config_path: str):
    """分析数据集并生成统计报告"""
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    logger.info(f"加载数据集: {data_config['dataset_name']}")
    
    # 加载数据集
    dataset = load_dataset(
        data_config['dataset_name'],
        data_config['dataset_config'],
        split=data_config['split'],
        trust_remote_code=True
    )
    
    total_samples = len(dataset)
    logger.info(f"总样本数: {total_samples}")
    
    # 统计时长和文本长度
    durations = []
    text_lengths = []
    max_duration = data_config['max_duration']
    valid_samples = 0
    
    logger.info("分析音频时长和文本长度...")
    for item in tqdm(dataset, desc="Processing"):
        # 获取音频时长（秒）
        audio = item['audio']
        duration = len(audio['array']) / audio['sampling_rate']
        
        if duration <= max_duration:
            durations.append(duration)
            text_lengths.append(len(item['text']))
            valid_samples += 1
    
    durations = np.array(durations)
    text_lengths = np.array(text_lengths)
    
    # 计算统计量
    logger.info("\n" + "="*60)
    logger.info("数据集统计报告 - LibriSpeech train-clean-100")
    logger.info("="*60)
    logger.info(f"\n【样本统计】")
    logger.info(f"  原始样本数: {total_samples}")
    logger.info(f"  有效样本数（≤{max_duration}s）: {valid_samples}")
    logger.info(f"  过滤比例: {(1 - valid_samples/total_samples)*100:.2f}%")
    
    logger.info(f"\n【音频时长统计（秒）】")
    logger.info(f"  总时长: {durations.sum()/3600:.2f} 小时")
    logger.info(f"  平均时长: {durations.mean():.2f}s")
    logger.info(f"  中位数时长: {np.median(durations):.2f}s")
    logger.info(f"  标准差: {durations.std():.2f}s")
    logger.info(f"  最小值: {durations.min():.2f}s")
    logger.info(f"  最大值: {durations.max():.2f}s")
    logger.info(f"  25% 分位数: {np.percentile(durations, 25):.2f}s")
    logger.info(f"  75% 分位数: {np.percentile(durations, 75):.2f}s")
    
    logger.info(f"\n【文本长度统计（字符数）】")
    logger.info(f"  平均长度: {text_lengths.mean():.1f}")
    logger.info(f"  中位数长度: {np.median(text_lengths):.1f}")
    logger.info(f"  标准差: {text_lengths.std():.1f}")
    logger.info(f"  最小值: {text_lengths.min()}")
    logger.info(f"  最大值: {text_lengths.max()}")
    
    # 与训练入口一致：验证集最多 100 条，其余作为训练集
    validation_size = data_config.get('validation_size', 100)
    val_size = max(1, min(validation_size, valid_samples // 10))
    train_size = valid_samples - val_size
    
    logger.info(f"\n【数据划分】")
    logger.info(f"  训练集: {train_size} 样本")
    logger.info(f"  验证集: {val_size} 样本")
    logger.info("  测试集: 未创建（历史实验协议）")
    
    indices = torch.randperm(
        valid_samples,
        generator=torch.Generator().manual_seed(42),
    ).numpy()
    train_hours = durations[indices[:train_size]].sum() / 3600
    val_hours = durations[indices[train_size:]].sum() / 3600
    
    logger.info(f"\n【训练集时长】: {train_hours:.2f} 小时")
    logger.info(f"【验证集时长】: {val_hours:.2f} 小时")
    logger.info("="*60 + "\n")
    
    # 保存报告到文件
    output_file = "outputs/data_100h_statistics.txt"
    os.makedirs("outputs", exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("数据集统计报告 - LibriSpeech train-clean-100\n")
        f.write("="*60 + "\n\n")
        f.write(f"【样本统计】\n")
        f.write(f"  原始样本数: {total_samples}\n")
        f.write(f"  有效样本数（≤{max_duration}s）: {valid_samples}\n")
        f.write(f"  过滤比例: {(1 - valid_samples/total_samples)*100:.2f}%\n\n")
        f.write(f"【音频时长统计（秒）】\n")
        f.write(f"  总时长: {durations.sum()/3600:.2f} 小时\n")
        f.write(f"  平均时长: {durations.mean():.2f}s\n")
        f.write(f"  中位数时长: {np.median(durations):.2f}s\n")
        f.write(f"  标准差: {durations.std():.2f}s\n")
        f.write(f"  最小值: {durations.min():.2f}s\n")
        f.write(f"  最大值: {durations.max():.2f}s\n")
        f.write(f"  25% 分位数: {np.percentile(durations, 25):.2f}s\n")
        f.write(f"  75% 分位数: {np.percentile(durations, 75):.2f}s\n\n")
        f.write(f"【文本长度统计（字符数）】\n")
        f.write(f"  平均长度: {text_lengths.mean():.1f}\n")
        f.write(f"  中位数长度: {np.median(text_lengths):.1f}\n")
        f.write(f"  标准差: {text_lengths.std():.1f}\n")
        f.write(f"  最小值: {text_lengths.min()}\n")
        f.write(f"  最大值: {text_lengths.max()}\n\n")
        f.write(f"【数据划分】\n")
        f.write(f"  训练集: {train_size} 样本\n")
        f.write(f"  验证集: {val_size} 样本\n")
        f.write("  测试集: 未创建（历史实验协议）\n\n")
        f.write(f"【训练集时长】: {train_hours:.2f} 小时\n")
        f.write(f"【验证集时长】: {val_hours:.2f} 小时\n")
        f.write("="*60 + "\n")
    
    logger.info(f"统计报告已保存至: {output_file}")
    
    return {
        'total_samples': total_samples,
        'valid_samples': valid_samples,
        'total_hours': durations.sum() / 3600,
        'train_hours': train_hours,
        'val_hours': val_hours,
        'test_hours': test_hours,
    }


if __name__ == "__main__":
    config_path = "configs/data_100h.yaml"
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
    
    stats = analyze_dataset(config_path)
    logger.info(f"\n✓ 数据分析完成！")
