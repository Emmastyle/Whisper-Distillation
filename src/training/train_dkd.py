"""Training script for DKD (Decoupled Knowledge Distillation)

DKD decouples the KD loss into two parts:
1. TCKD (Target Class KD): Knowledge about the target class
2. NCKD (Non-target Class KD): Knowledge about non-target classes

Reference: "Decoupled Knowledge Distillation" (CVPR 2022)
https://arxiv.org/abs/2203.08679
"""
import os
import argparse
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from src.data import WhisperASRDataset, whisper_collate_fn
from src.models import TeacherWhisper, StudentASRModel
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_wer, calculate_cer
from transformers import WhisperTokenizer, WhisperFeatureExtractor, get_linear_schedule_with_warmup


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_decoder_input_ids(
    labels: torch.Tensor,
    bos_token_id: int,
    pad_token_id: Optional[int] = None,
):
    """
    Prepare decoder input ids by shifting labels right
    and prepending BOS token
    
    Args:
        labels: (batch, seq_len) target labels
        bos_token_id: BOS token id
        pad_token_id: Padding token id (fallbacks to BOS when None)
    
    Returns:
        decoder_input_ids: (batch, seq_len) shifted labels
    """
    # Create decoder input by shifting right and adding BOS
    batch_size, seq_len = labels.shape
    if pad_token_id is None:
        pad_token_id = bos_token_id

    decoder_input_ids = torch.full(
        (batch_size, seq_len), pad_token_id, dtype=torch.long, device=labels.device
    )
    
    # Shift labels to right
    shifted_labels = labels[:, :-1].clone()
    shifted_labels = shifted_labels.masked_fill(shifted_labels == -100, pad_token_id)

    decoder_input_ids[:, 1:] = shifted_labels
    decoder_input_ids[:, 0] = bos_token_id
    
    return decoder_input_ids


def compute_dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
    beta: float,
    topk: Optional[int] = None,
):
    """
    Compute Decoupled Knowledge Distillation (DKD) loss
    
    DKD = alpha * TCKD + beta * NCKD
    
    TCKD (Target Class KD): Binary KD on target vs non-target
    NCKD (Non-target Class KD): KD on distribution of non-target classes
    
    Args:
        student_logits: (batch, seq_len, vocab_size)
        teacher_logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len) target labels
        temperature: Temperature for softening
        alpha: Weight for TCKD
        beta: Weight for NCKD
        
    Returns:
        dkd_loss: DKD loss
        tckd_loss: TCKD component
        nckd_loss: NCKD component
    """
    # Create mask for valid positions (ignore padding)
    mask = (labels != -100).float()  # (batch, seq_len)
    
    batch_size, seq_len, vocab_size = student_logits.shape
    
    # Flatten for easier computation
    student_logits_flat = student_logits.view(-1, vocab_size)  # (B*T, V)
    teacher_logits_flat = teacher_logits.view(-1, vocab_size)  # (B*T, V)
    labels_flat = labels.view(-1)  # (B*T,)
    mask_flat = mask.view(-1)  # (B*T,)
    
    # Get softmax probabilities with temperature
    student_probs = F.softmax(student_logits_flat / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits_flat / temperature, dim=-1)
    
    # Get target class probabilities
    # Create one-hot for target class
    target_mask = torch.zeros_like(student_probs)  # (B*T, V)
    valid_positions = labels_flat != -100
    target_mask[valid_positions, labels_flat[valid_positions]] = 1.0
    
    # Extract target class probabilities
    student_target = (student_probs * target_mask).sum(dim=-1)  # (B*T,)
    teacher_target = (teacher_probs * target_mask).sum(dim=-1)  # (B*T,)
    
    # ========== TCKD: Target Class Knowledge Distillation ==========
    # Binary classification: target vs non-target
    # Use binary cross-entropy on target class probability
    
    # Clamp to avoid log(0)
    student_target = torch.clamp(student_target, min=1e-7, max=1.0 - 1e-7)
    teacher_target = torch.clamp(teacher_target, min=1e-7, max=1.0 - 1e-7)
    
    # Binary KL divergence for target class
    tckd = teacher_target * torch.log(teacher_target / student_target) + \
           (1 - teacher_target) * torch.log((1 - teacher_target) / (1 - student_target))
    
    # Apply mask and average
    tckd_loss = (tckd * mask_flat).sum() / mask_flat.sum()
    
    # ========== NCKD: Non-target Class Knowledge Distillation ==========
    # KL divergence on non-target classes
    
    # Normalize non-target probabilities
    student_non_target = student_probs * (1 - target_mask)  # Zero out target
    teacher_non_target = teacher_probs * (1 - target_mask)
    
    # Normalize to sum to 1 (excluding target)
    student_non_target_sum = student_non_target.sum(dim=-1, keepdim=True)
    teacher_non_target_sum = teacher_non_target.sum(dim=-1, keepdim=True)
    
    # Avoid division by zero
    student_non_target_sum = torch.clamp(student_non_target_sum, min=1e-7)
    teacher_non_target_sum = torch.clamp(teacher_non_target_sum, min=1e-7)
    
    student_non_target_norm = student_non_target / student_non_target_sum
    teacher_non_target_norm = teacher_non_target / teacher_non_target_sum

    if topk is not None and topk > 0:
        max_non_target = vocab_size - 1
        k = min(topk, max_non_target)
        if k > 0:
            _, indices = torch.topk(
                teacher_non_target_norm,
                k,
                dim=-1,
            )
            topk_mask = torch.zeros_like(teacher_non_target_norm)
            topk_mask.scatter_(dim=-1, index=indices, value=1.0)
            teacher_non_target_norm = teacher_non_target_norm * topk_mask
            student_non_target_norm = student_non_target_norm * topk_mask

            teacher_non_target_norm = teacher_non_target_norm / torch.clamp(
                teacher_non_target_norm.sum(dim=-1, keepdim=True),
                min=1e-7,
            )
            student_non_target_norm = student_non_target_norm / torch.clamp(
                student_non_target_norm.sum(dim=-1, keepdim=True),
                min=1e-7,
            )
    
    # KL divergence on non-target distribution
    # KL(teacher || student) for non-target classes
    student_log_probs = torch.log(torch.clamp(student_non_target_norm, min=1e-7))
    nckd = teacher_non_target_norm * (
        torch.log(torch.clamp(teacher_non_target_norm, min=1e-7)) - student_log_probs
    )
    nckd = nckd.sum(dim=-1)  # Sum over vocab
    
    # Apply mask and average
    nckd_loss = (nckd * mask_flat).sum() / mask_flat.sum()
    
    # ========== Combine TCKD and NCKD ==========
    # Scale by T^2 (standard in KD)
    tckd_loss = tckd_loss * (temperature ** 2)
    nckd_loss = nckd_loss * (temperature ** 2)
    
    # Weighted combination
    dkd_loss = alpha * tckd_loss + beta * nckd_loss
    
    return dkd_loss, tckd_loss, nckd_loss


def evaluate(
    model,
    dataloader,
    tokenizer,
    device,
    pad_token_id: Optional[int] = None,
    max_samples: int = 100,
):
    """
    Evaluate model on validation set
    
    Args:
        model: Student ASR model
        dataloader: Validation dataloader
        tokenizer: Whisper tokenizer
        device: Device to use
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        avg_loss: Average loss
        wer: Word Error Rate
        cer: Character Error Rate
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    predictions = []
    references = []
    
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.bos_token_id

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx * dataloader.batch_size >= max_samples:
                break
            
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['texts']
            
            # Prepare decoder input
            decoder_input_ids = prepare_decoder_input_ids(
                labels,
                tokenizer.bos_token_id,
                pad_token_id,
            )
            
            # Forward pass
            loss, logits = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
            )
            
            total_loss += loss.item()
            num_batches += 1
            
            # Generate predictions for metrics
            generated_ids = model.generate(
                input_features=input_features,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode predictions
            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(pred_texts)
            references.extend(texts)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    wer = calculate_wer(predictions, references)
    cer = calculate_cer(predictions, references)
    
    model.train()
    return avg_loss, wer, cer


def train(args):
    """Main training function"""
    # Load config
    config = load_config(args.config)
    
    if config.dkd is None:
        raise ValueError("DKD config must include a 'dkd' section.")

    # Setup output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("train_dkd", str(output_dir / "train.log"))
    logger.info(f"Starting DKD training with config: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    # Set seed
    set_seed(config.training.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {config.model.teacher_model}")
    tokenizer = WhisperTokenizer.from_pretrained(config.model.teacher_model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.bos_token_id
    logger.info(f"Tokenizer pad token id: {pad_token_id}")
    
    # Load teacher model
    logger.info(f"Loading teacher model: {config.model.teacher_model}")
    teacher = TeacherWhisper(model_name=config.model.teacher_model)
    teacher.disable_dropout()
    teacher = teacher.to(device)
    teacher.eval()
    logger.info("Teacher dropout disabled for deterministic logits")
    
    # Create student model
    logger.info("Creating student model")
    logger.info(f"  Decoder layers: {config.model.student_decoder_layers}")
    logger.info(f"  Decoder heads: {config.model.student_decoder_heads}")
    
    model = StudentASRModel(
        teacher=teacher,
        student_decoder_layers=config.model.student_decoder_layers,
        student_decoder_heads=config.model.student_decoder_heads,
        dropout=config.model.dropout,
    )
    model = model.to(device)
    
    # Count parameters
    student_params = sum(p.numel() for p in model.student_decoder.parameters())
    logger.info(f"Student decoder parameters: {student_params:,}")
    
    # Load datasets
    logger.info("Loading datasets")
    logger.info(f"  Dataset: {config.data.dataset_name}")
    logger.info(f"  Config: {config.data.dataset_config}")
    logger.info(f"  Split: {config.data.split}")
    
    # Create feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(config.model.teacher_model)
    
    # Load training dataset
    train_dataset = WhisperASRDataset(
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        split=config.data.split,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_duration=config.data.max_duration,
        num_samples=config.data.num_samples,
        sample_rate=config.data.sample_rate,
    )
    
    # For validation, use a small subset (10% of training data or 500 samples, whichever is smaller)
    val_samples = min(500, len(train_dataset) // 10)
    logger.info(f"  Using {val_samples} samples for validation")
    
    val_dataset = WhisperASRDataset(
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        split=config.data.split,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_duration=config.data.max_duration,
        num_samples=val_samples,
        sample_rate=config.data.sample_rate,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=whisper_collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=whisper_collate_fn,
        pin_memory=True,
    )
    
    # Optimizer
    optimizer = AdamW(
        model.student_decoder.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.training.epochs
    warmup_steps = config.training.warmup_steps
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # DKD parameters
    logger.info("DKD parameters:")
    logger.info(f"  Temperature: {config.dkd.temperature}")
    logger.info(f"  Alpha (TCKD weight): {config.dkd.alpha}")
    logger.info(f"  Beta (NCKD weight): {config.dkd.beta}")
    logger.info(f"  Gamma (CE weight): {config.dkd.gamma}")
    logger.info(f"  NCKD top-k: {config.dkd.topk if config.dkd.topk is not None else 'all'}")
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.fp16)
    logger.info(f"Mixed precision training: {config.training.fp16}")
    
    # Training loop
    logger.info("Starting training...")
    
    global_step = 0
    best_wer = float('inf')
    
    for epoch in range(config.training.epochs):
        model.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_dkd_loss = 0
        epoch_tckd_loss = 0
        epoch_nckd_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Prepare decoder input
            decoder_input_ids = prepare_decoder_input_ids(
                labels,
                tokenizer.bos_token_id,
                pad_token_id,
            )
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=config.training.fp16):
                # Forward pass with teacher logits
                ce_loss, student_logits, teacher_logits = model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    return_teacher_logits=True,
                )
                
                # Compute DKD loss
                dkd_loss, tckd_loss, nckd_loss = compute_dkd_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    temperature=config.dkd.temperature,
                    alpha=config.dkd.alpha,
                    beta=config.dkd.beta,
                    topk=config.dkd.topk,
                )
                
                # Combined loss: gamma * CE + (1 - gamma) * DKD
                loss = config.dkd.gamma * ce_loss + (1 - config.dkd.gamma) * dkd_loss
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.student_decoder.parameters(),
                config.training.gradient_clip
            )
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_dkd_loss += dkd_loss.item()
            epoch_tckd_loss += tckd_loss.item()
            epoch_nckd_loss += nckd_loss.item()
            global_step += 1
            
            if global_step % config.training.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_ce = epoch_ce_loss / (batch_idx + 1)
                avg_dkd = epoch_dkd_loss / (batch_idx + 1)
                avg_tckd = epoch_tckd_loss / (batch_idx + 1)
                avg_nckd = epoch_nckd_loss / (batch_idx + 1)
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce': f'{ce_loss.item():.4f}',
                    'dkd': f'{dkd_loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f}, "
                    f"ce={ce_loss.item():.4f}, dkd={dkd_loss.item():.4f} "
                    f"(tckd={tckd_loss.item():.4f}, nckd={nckd_loss.item():.4f}), "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
            
            # Evaluation
            if global_step % config.training.eval_interval == 0:
                logger.info(f"Running evaluation at step {global_step}...")
                val_loss, wer, cer = evaluate(
                    model,
                    val_loader,
                    tokenizer,
                    device,
                    pad_token_id=pad_token_id,
                )
                logger.info(
                    f"Step {global_step} - Val Loss: {val_loss:.4f}, "
                    f"WER: {wer:.2f}%, CER: {cer:.2f}%"
                )
                
                # Save best model
                if wer < best_wer:
                    best_wer = wer
                    save_path = output_dir / "best_model.pt"
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.student_decoder.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'wer': wer,
                        'cer': cer,
                        'config': config,
                    }, save_path)

                    logger.info(f"Saved best model (WER: {wer:.2f}%) to {save_path}")
            
            # Save checkpoint
            if global_step % config.training.save_interval == 0:
                save_path = output_dir / f"checkpoint_{global_step}.pt"
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.student_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")
        
        # End of epoch evaluation
        logger.info(f"Epoch {epoch+1} completed. Running full evaluation...")
        val_loss, wer, cer = evaluate(
            model,
            val_loader,
            tokenizer,
            device,
            pad_token_id=pad_token_id,
            max_samples=1000,
        )
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_ce = epoch_ce_loss / len(train_loader)
        avg_epoch_dkd = epoch_dkd_loss / len(train_loader)
        avg_epoch_tckd = epoch_tckd_loss / len(train_loader)
        avg_epoch_nckd = epoch_nckd_loss / len(train_loader)
        
        logger.info(
            f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.4f} "
            f"(CE: {avg_epoch_ce:.4f}, DKD: {avg_epoch_dkd:.4f}, "
            f"TCKD: {avg_epoch_tckd:.4f}, NCKD: {avg_epoch_nckd:.4f})"
        )
        logger.info(
            f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, WER: {wer:.2f}%, CER: {cer:.2f}%"
        )
        
        # Update best model if epoch evaluation is better
        if wer < best_wer:
            best_wer = wer
            save_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.student_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'wer': wer,
                'cer': cer,
                'config': config,
            }, save_path)
            logger.info(f"Saved best model (WER: {wer:.2f}%) to {save_path}")
    
    logger.info("Training completed!")
    logger.info(f"Best WER: {best_wer:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DKD model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    
    args = parser.parse_args()
    train(args)

