"""Configuration utilities"""
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "librispeech_asr"
    dataset_config: str = "clean"
    split: str = "train.100"
    num_samples: Optional[int] = None  # None = use all
    max_duration: float = 10.0  # seconds
    batch_size: int = 8
    num_workers: int = 4
    sample_rate: int = 16000


@dataclass
class ModelConfig:
    """Model configuration"""
    teacher_model: str = "openai/whisper-tiny"
    student_decoder_layers: int = 2
    student_decoder_heads: int = 4
    student_hidden_size: int = 384  # whisper-tiny hidden size
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    log_interval: int = 50
    eval_interval: int = 500
    save_interval: int = 1000
    output_dir: str = "./outputs"
    seed: int = 42
    fp16: bool = True


@dataclass
class KDConfig:
    """Knowledge Distillation configuration"""
    temperature: float = 2.0
    alpha: float = 0.5  # weight for CE loss (1-alpha for KD)


@dataclass
class Config:
    """Main configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    kd: Optional[KDConfig] = None  # Only for KD training


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    kd_config = None
    if 'kd' in config_dict:
        kd_config = KDConfig(**config_dict['kd'])
    
    return Config(
        data=data_config,
        model=model_config,
        training=training_config,
        kd=kd_config
    )

