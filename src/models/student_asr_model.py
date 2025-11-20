"""Student ASR Model combining frozen encoder and trainable decoder"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .teacher import TeacherWhisper
from .student_decoder import StudentDecoder


class StudentASRModel(nn.Module):
    """
    Student ASR model that:
    1. Uses frozen teacher encoder
    2. Trains a lightweight student decoder
    """
    
    def __init__(
        self,
        teacher: TeacherWhisper,
        student_decoder_layers: int = 2,
        student_decoder_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            teacher: Teacher Whisper model (frozen)
            student_decoder_layers: Number of decoder layers for student
            student_decoder_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.teacher = teacher
        
        # Student decoder
        self.student_decoder = StudentDecoder(
            vocab_size=teacher.vocab_size,
            d_model=teacher.d_model,
            n_layers=student_decoder_layers,
            n_heads=student_decoder_heads,
            dropout=dropout,
        )
        
        # Store properties
        self.vocab_size = teacher.vocab_size
        self.d_model = teacher.d_model
        self.config = teacher.config
    
    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_teacher_logits: bool = False,
    ):
        """
        Forward pass
        
        Args:
            input_features: (batch, n_mels, n_frames) audio features
            decoder_input_ids: (batch, seq_len) decoder input ids
            labels: (batch, seq_len) target labels (optional, for loss calculation)
            return_teacher_logits: Whether to return teacher logits (for KD)
            
        Returns:
            If labels is None:
                logits: (batch, seq_len, vocab_size)
            If labels is provided:
                loss: CE loss
                logits: (batch, seq_len, vocab_size)
            If return_teacher_logits:
                Also returns teacher_logits
        """
        # Get encoder outputs from frozen teacher
        with torch.no_grad():
            encoder_outputs = self.teacher.encode(input_features)
        
        # Get encoder hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Student decoder forward
        student_logits = self.student_decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # logits: (batch, seq_len, vocab_size)
            # labels: (batch, seq_len)
            
            # Flatten for CrossEntropyLoss
            shift_logits = student_logits.contiguous().view(-1, self.vocab_size)
            shift_labels = labels.contiguous().view(-1)
            
            # Calculate CE loss (ignore_index=-100 for padding)
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction='mean'
            )
        
        # Get teacher logits if requested (for KD)
        teacher_logits = None
        if return_teacher_logits:
            with torch.no_grad():
                teacher_logits = self.teacher.decode(encoder_outputs, decoder_input_ids)
        
        if return_teacher_logits:
            return loss, student_logits, teacher_logits
        
        if loss is not None:
            return loss, student_logits
        
        return student_logits
    
    def generate(
        self,
        input_features: torch.Tensor,
        max_length: int = 448,
        bos_token_id: int = 50257,  # Whisper's BOS token
        eos_token_id: int = 50257,  # Whisper's EOS token
    ):
        """
        Greedy generation for inference
        
        Args:
            input_features: (batch, n_mels, n_frames) audio features
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            
        Returns:
            generated_ids: (batch, seq_len) generated token ids
        """
        batch_size = input_features.shape[0]
        device = input_features.device
        
        # Get encoder outputs
        with torch.no_grad():
            encoder_outputs = self.teacher.encode(input_features)
            encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Initialize with BOS token
        decoder_input_ids = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        
        # Greedy generation
        for _ in range(max_length - 1):
            # Get logits
            logits = self.student_decoder(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :]  # (batch, vocab_size)
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch, 1)
            
            # Append to sequence
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_tokens == eos_token_id).all():
                break
        
        return decoder_input_ids

