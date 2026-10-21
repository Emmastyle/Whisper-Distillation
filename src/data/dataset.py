"""Whisper ASR Dataset implementation"""
import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from typing import Dict, List, Optional, Tuple
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not found. Install with: pip install librosa")


class WhisperASRDataset(Dataset):
    """Whisper ASR Dataset for LibriSpeech"""
    
    def __init__(
        self,
        dataset_name: str = "librispeech_asr",
        dataset_config: str = "clean",
        split: str = "train.100",
        feature_extractor: Optional[WhisperFeatureExtractor] = None,
        tokenizer: Optional[WhisperTokenizer] = None,
        max_duration: float = 10.0,
        num_samples: Optional[int] = None,
        sample_rate: int = 16000,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            split: Dataset split to use
            feature_extractor: Whisper feature extractor
            tokenizer: Whisper tokenizer
            max_duration: Maximum audio duration in seconds
            num_samples: Number of samples to use (None = all)
            sample_rate: Target sample rate (Whisper uses 16kHz)
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
        # Load feature extractor and tokenizer
        if feature_extractor is None:
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        else:
            self.feature_extractor = feature_extractor
            
        if tokenizer is None:
            self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
        else:
            self.tokenizer = tokenizer
        
        # Load dataset
        print(f"Loading dataset: {dataset_name} ({dataset_config}) - {split}")
        self.dataset = load_dataset(dataset_name, dataset_config, split=split)
        
        # Filter by duration and limit samples
        print("Filtering dataset by duration...")
        self.dataset = self.dataset.filter(
            lambda x: x['audio']['array'].shape[0] / x['audio']['sampling_rate'] <= max_duration
        )
        
        if num_samples is not None and num_samples < len(self.dataset):
            print(f"Limiting to {num_samples} samples")
            self.dataset = self.dataset.select(range(num_samples))
        
        print(f"Dataset size: {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample"""
        item = self.dataset[idx]
        
        # Load audio
        audio_array = item['audio']['array']
        orig_sr = item['audio']['sampling_rate']
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            if HAS_LIBROSA:
                # Use librosa for resampling (more stable)
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=orig_sr, 
                    target_sr=self.sample_rate
                )
            else:
                # If audio is already numpy array and close to target SR, just use it
                # LibriSpeech is already 16kHz, so this should rarely happen
                if abs(orig_sr - self.sample_rate) > 100:
                    raise RuntimeError(
                        f"Audio sample rate ({orig_sr}) differs from target ({self.sample_rate}). "
                        f"Please install librosa: pip install librosa"
                    )
        
        # Extract features (log-mel spectrogram)
        # feature_extractor returns input_features of shape (n_mels, n_frames)
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        input_features = inputs.input_features.squeeze(0)  # Remove batch dim
        
        # Tokenize text
        text = item['text']
        # Add special tokens: prepend with start token, append with end token
        labels = self.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)
        
        return {
            'input_features': input_features,
            'labels': labels,
            'text': text,
        }


def whisper_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for Whisper ASR batches
    Pads input_features and labels to batch max length
    """
    # Extract items
    input_features = [item['input_features'] for item in batch]
    labels = [item['labels'] for item in batch]
    texts = [item['text'] for item in batch]
    
    # Pad input features (they should all be same shape for Whisper)
    # Shape: (batch, n_mels, n_frames)
    input_features = torch.stack(input_features, dim=0)
    
    # Pad labels to max length in batch
    max_label_len = max(len(label) for label in labels)
    
    # Use -100 as padding (ignored by CrossEntropyLoss)
    padded_labels = []
    for label in labels:
        padding = torch.full((max_label_len - len(label),), -100, dtype=torch.long)
        padded_label = torch.cat([label, padding])
        padded_labels.append(padded_label)
    
    labels = torch.stack(padded_labels, dim=0)
    
    return {
        'input_features': input_features,
        'labels': labels,
        'texts': texts,
    }


def split_dataset(
    dataset: Dataset,
    seed: int,
    validation_size: int = 100,
) -> Tuple[Dataset, Dataset]:
    """Reproduce the original split: at most 100 validation samples, no test set."""
    if len(dataset) < 2:
        raise ValueError("At least two samples are required for a train/validation split.")
    if validation_size < 1:
        raise ValueError("validation_size must be at least 1.")

    val_size = min(validation_size, len(dataset) // 10)
    val_size = max(1, val_size)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)
