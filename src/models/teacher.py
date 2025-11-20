"""Teacher Whisper Model Wrapper"""
import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Optional


class TeacherWhisper(nn.Module):
    """
    Teacher Whisper model wrapper
    Provides encoder and decoder inference (no gradients)
    """
    
    def __init__(self, model_name: str = "openai/whisper-tiny"):
        super().__init__()
        print(f"Loading teacher model: {model_name}")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Store model properties
        self.config = self.model.config
        self.vocab_size = self.config.vocab_size
        self.d_model = self.config.d_model
    
    @torch.no_grad()
    def encode(self, input_features: torch.Tensor):
        """
        Encode audio features to encoder hidden states
        
        Args:
            input_features: (batch, n_mels, n_frames) audio features
            
        Returns:
            encoder_outputs: BaseModelOutput with last_hidden_state
        """
        encoder_outputs = self.model.get_encoder()(input_features)
        return encoder_outputs
    
    @torch.no_grad()
    def decode(self, encoder_outputs, decoder_input_ids: torch.Tensor):
        """
        Decode from encoder outputs and decoder input ids
        
        Args:
            encoder_outputs: Output from encoder
            decoder_input_ids: (batch, seq_len) decoder input token ids
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
        )
        return outputs.logits
    
    @torch.no_grad()
    def forward(self, input_features: torch.Tensor, decoder_input_ids: torch.Tensor):
        """
        Full forward pass (encode + decode)
        
        Args:
            input_features: (batch, n_mels, n_frames) audio features
            decoder_input_ids: (batch, seq_len) decoder input token ids
            
        Returns:
            encoder_outputs: Encoder hidden states
            logits: (batch, seq_len, vocab_size) decoder logits
        """
        encoder_outputs = self.encode(input_features)
        logits = self.decode(encoder_outputs, decoder_input_ids)
        return encoder_outputs, logits
    
    def get_encoder(self):
        """Return the encoder"""
        return self.model.get_encoder()
    
    def get_decoder(self):
        """Return the decoder"""
        return self.model.get_decoder()


# Alias for backward compatibility
TeacherWhisperModel = TeacherWhisper

