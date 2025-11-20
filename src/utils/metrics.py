"""Metrics calculation utilities with Whisper-style normalization"""
import jiwer
import logging
from typing import List, Tuple
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

logger = logging.getLogger(__name__)

# 创建 Whisper 官方的文本规范化器（用于 ASR 评估）
# 功能：转小写、去除标点、去除多余空格、数字规范化、去除特殊字符
normalizer = BasicTextNormalizer()


def normalize_text(text: str) -> str:
    """
    使用 Whisper 官方的规范化器处理文本
    
    处理内容：
    - 转小写
    - 去除标点
    - 去除多余空格
    - 数字规范化 (23 <-> twenty three)
    - 去除特殊字符
    
    Args:
        text: 原始文本
        
    Returns:
        规范化后的文本
    """
    if not text or not text.strip():
        return ""
    return normalizer(text).strip()


def debug_normalization(text: str) -> None:
    """
    打印规范化前后的对比，用于调试
    使用 repr() 可以看出隐藏的空格、换行等字符
    
    Args:
        text: 要调试的文本
    """
    print("RAW: ", repr(text))
    print("NORM:", repr(normalize_text(text)))


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate with Whisper normalization
    
    使用 Whisper 官方的文本规范化器处理 prediction 和 reference，
    然后计算 WER。这是 ASR 评估的标准做法。
    
    Args:
        predictions: 模型预测的文本列表
        references: 真实标签文本列表
        
    Returns:
        WER 百分比 (0-100+)
    """
    # 先规范化，再过滤空字符串
    normalized_pairs = [
        (normalize_text(p), normalize_text(r)) 
        for p, r in zip(predictions, references)
    ]
    valid_pairs = [(p, r) for p, r in normalized_pairs if r]
    
    # 统计有效样本数
    num_total = len(predictions)
    num_valid = len(valid_pairs)
    if num_valid < num_total:
        logger.warning(
            f"WER eval: {num_valid}/{num_total} samples used after normalization "
            f"({num_total - num_valid} samples filtered out due to empty references)"
        )
    
    if not valid_pairs:
        logger.warning("No valid samples for WER calculation, returning 0.0")
        return 0.0
    
    preds, refs = zip(*valid_pairs)
    wer = jiwer.wer(list(refs), list(preds))
    return wer * 100  # Return as percentage


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate with Whisper normalization
    
    使用 Whisper 官方的文本规范化器处理 prediction 和 reference，
    然后计算 CER。这是 ASR 评估的标准做法。
    
    Args:
        predictions: 模型预测的文本列表
        references: 真实标签文本列表
        
    Returns:
        CER 百分比 (0-100+)
    """
    # 先规范化，再过滤空字符串
    normalized_pairs = [
        (normalize_text(p), normalize_text(r)) 
        for p, r in zip(predictions, references)
    ]
    valid_pairs = [(p, r) for p, r in normalized_pairs if r]
    
    # 统计有效样本数
    num_total = len(predictions)
    num_valid = len(valid_pairs)
    if num_valid < num_total:
        logger.warning(
            f"CER eval: {num_valid}/{num_total} samples used after normalization "
            f"({num_total - num_valid} samples filtered out due to empty references)"
        )
    
    if not valid_pairs:
        logger.warning("No valid samples for CER calculation, returning 0.0")
        return 0.0
    
    preds, refs = zip(*valid_pairs)
    cer = jiwer.cer(list(refs), list(preds))
    return cer * 100  # Return as percentage

