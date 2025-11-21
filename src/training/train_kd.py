"""Training script for KD (Knowledge Distillation) baseline"""
import os
import argparse
import random
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
from transformers import WhisperTokenizer, get_linear_schedule_with_warmup


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_decoder_input_ids(labels: torch.Tensor, bos_token_id: int):
    """
    Prepare decoder input ids by shifting labels right
    and prepending BOS token
    
    Args:
        labels: (batch, seq_len) target labels
        bos_token_id: BOS token id
    
    Returns:
        decoder_input_ids: (batch, seq_len) shifted labels
    """
    # Create decoder input by shifting right and adding BOS
    batch_size, seq_len = labels.shape
    decoder_input_ids = torch.full(
        (batch_size, seq_len), -100, dtype=torch.long, device=labels.device
    )
    
    # Shift labels to right
    decoder_input_ids[:, 1:] = labels[:, :-1].clone()
    decoder_input_ids[:, 0] = bos_token_id
    
    # Replace -100 (padding) with pad token
    # For Whisper, we use the EOS token as padding
    decoder_input_ids = decoder_input_ids.masked_fill(decoder_input_ids == -100, bos_token_id)
    
    return decoder_input_ids


def compute_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    labels: torch.Tensor,
):
    """
    Compute KL divergence loss for knowledge distillation
    
    Args:
        student_logits: (batch, seq_len, vocab_size)
        teacher_logits: (batch, seq_len, vocab_size)
        temperature: Temperature for softening
        labels: (batch, seq_len) for masking padding
        
    Returns:
        kd_loss: KL divergence loss
    """
    # Create mask for valid positions (ignore padding)
    mask = (labels != -100).float()  # (batch, seq_len)
    
    # Apply temperature
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Compute KL divergence
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none',
        log_target=False,
    )  # (batch, seq_len, vocab_size)
    
    # Sum over vocabulary, then average over valid positions
    kl_div = kl_div.sum(dim=-1)  # (batch, seq_len)
    
    # Apply mask and compute mean
    kl_div = (kl_div * mask).sum() / mask.sum()
    
    # Scale by T^2 (standard in KD literature)
    kd_loss = kl_div * (temperature ** 2)
    
    return kd_loss


def evaluate(model, dataloader, tokenizer, device, max_samples=100):
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
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx * dataloader.batch_size >= max_samples:
                break
            
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['texts']
            
            # Prepare decoder input
            decoder_input_ids = prepare_decoder_input_ids(labels, tokenizer.bos_token_id)
            
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
    
    # Check that KD config exists
    if config.kd is None:
        raise ValueError("KD config not found in config file. Please add 'kd' section.")
    
    # Setup output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger("train_kd", str(output_dir / "train.log"))
    logger.info(f"Starting KD training with config: {args.config}")
    logger.info(f"KD parameters - Temperature: {config.kd.temperature}, Alpha: {config.kd.alpha}")
    
    # Set seed
    set_seed(config.training.seed)
    logger.info(f"Random seed: {config.training.seed}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(config.model.teacher_model)
    
    # Create datasets
    logger.info("Loading training dataset...")
    train_dataset = WhisperASRDataset(
        dataset_name=config.data.dataset_name,
        dataset_config=config.data.dataset_config,
        split=config.data.split,
        max_duration=config.data.max_duration,
        num_samples=config.data.num_samples,
        sample_rate=config.data.sample_rate,
    )
    
    # Create validation split (use last 10% of training data)
    val_size = min(100, len(train_dataset) // 10)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
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
    
    # Create models
    logger.info("Loading models...")
    teacher = TeacherWhisper(config.model.teacher_model).to(device)
    
    model = StudentASRModel(
        teacher=teacher,
        student_decoder_layers=config.model.student_decoder_layers,
        student_decoder_heads=config.model.student_decoder_heads,
        dropout=config.model.dropout,
    ).to(device)
    
    # Count parameters
    student_params = sum(p.numel() for p in model.student_decoder.parameters())
    logger.info(f"Student decoder parameters: {student_params:,}")
    
    # Optimizer
    optimizer = AdamW(
        model.student_decoder.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    # Scheduler
    total_steps = len(train_loader) * config.training.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_wer = float('inf')
    
    for epoch in range(config.training.epochs):
        model.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_kd_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Prepare decoder input
            decoder_input_ids = prepare_decoder_input_ids(labels, tokenizer.bos_token_id)
            
            # Forward pass with teacher logits
            ce_loss, student_logits, teacher_logits = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                return_teacher_logits=True,
            )
            
            # Compute KD loss
            kd_loss = compute_kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temperature=config.kd.temperature,
                labels=labels,
            )
            
            # Combined loss: (1 - alpha) * CE + alpha * KD
            loss = (1 - config.kd.alpha) * ce_loss + config.kd.alpha * kd_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.student_decoder.parameters(),
                config.training.gradient_clip
            )
            
            optimizer.step()
            scheduler.step()
            
            # Logging
            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss.item()
            epoch_kd_loss += kd_loss.item()
            global_step += 1
            
            if global_step % config.training.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_ce = epoch_ce_loss / (batch_idx + 1)
                avg_kd = epoch_kd_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce': f'{ce_loss.item():.4f}',
                    'kd': f'{kd_loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                logger.info(
                    f"Step {global_step}: loss={loss.item():.4f}, "
                    f"ce={ce_loss.item():.4f}, kd={kd_loss.item():.4f}, "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
            
            # Evaluation
            if global_step % config.training.eval_interval == 0:
                logger.info("Running evaluation...")
                val_loss, wer, cer = evaluate(model, val_loader, tokenizer, device)
                logger.info(
                    f"Validation - Loss: {val_loss:.4f}, WER: {wer:.2f}%, CER: {cer:.2f}%"
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
        val_loss, wer, cer = evaluate(model, val_loader, tokenizer, device, max_samples=1000)
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_ce = epoch_ce_loss / len(train_loader)
        avg_epoch_kd = epoch_kd_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1} - Train Loss: {avg_epoch_loss:.4f} "
            f"(CE: {avg_epoch_ce:.4f}, KD: {avg_epoch_kd:.4f})"
        )
        logger.info(
            f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}, WER: {wer:.2f}%, CER: {cer:.2f}%"
        )
    
    logger.info("Training completed!")
    logger.info(f"Best WER: {best_wer:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KD baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    train(args)

