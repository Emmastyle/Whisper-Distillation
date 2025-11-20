"""Student Decoder Implementation"""
import torch
import torch.nn as nn
from transformers import WhisperConfig
from typing import Optional
import math


class StudentDecoder(nn.Module):
    """
    Lightweight student decoder for ASR
    Uses standard Transformer decoder with cross-attention to encoder outputs
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_length: int = 448,  # Whisper's max length
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (should match encoder output)
            n_layers: Number of decoder layers
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout rate
            max_length: Maximum sequence length
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learned, like Whisper)
        self.embed_positions = nn.Embedding(max_length, d_model)
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=False,  # Post-norm like original Transformer
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings with output projection (optional but common)
        self.output_projection.weight = self.embed_tokens.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters"""
        # Use Xavier/Glorot initialization
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.d_model**-0.5)
        nn.init.normal_(self.embed_positions.weight, mean=0, std=self.d_model**-0.5)
    
    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass
        
        Args:
            decoder_input_ids: (batch, tgt_seq_len) decoder input token ids
            encoder_hidden_states: (batch, src_seq_len, d_model) encoder outputs
            encoder_attention_mask: (batch, src_seq_len) attention mask for encoder
            
        Returns:
            logits: (batch, tgt_seq_len, vocab_size)
        """
        batch_size, seq_len = decoder_input_ids.shape
        
        # Token embeddings
        token_embeds = self.embed_tokens(decoder_input_ids)  # (batch, seq_len, d_model)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=decoder_input_ids.device).unsqueeze(0)
        pos_embeds = self.embed_positions(positions)  # (1, seq_len, d_model)
        
        # Combine embeddings
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.dropout(hidden_states)
        
        # Create causal mask for decoder (prevent attending to future tokens)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=decoder_input_ids.device)
        
        # Decode with cross-attention to encoder outputs
        decoded = self.decoder(
            tgt=hidden_states,
            memory=encoder_hidden_states,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=encoder_attention_mask if encoder_attention_mask is not None else None,
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoded)  # (batch, seq_len, vocab_size)
        
        return logits
    
    def _generate_square_subsequent_mask(self, sz: int, device: torch.device):
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

