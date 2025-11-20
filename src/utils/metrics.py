"""Metrics calculation utilities"""
import jiwer
from typing import List


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """Calculate Word Error Rate"""
    # Filter out empty strings
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not valid_pairs:
        return 0.0
    
    preds, refs = zip(*valid_pairs)
    wer = jiwer.wer(list(refs), list(preds))
    return wer * 100  # Return as percentage


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """Calculate Character Error Rate"""
    # Filter out empty strings
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not valid_pairs:
        return 0.0
    
    preds, refs = zip(*valid_pairs)
    cer = jiwer.cer(list(refs), list(preds))
    return cer * 100  # Return as percentage

