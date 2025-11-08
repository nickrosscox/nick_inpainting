"""Configuration system for VAE inpainting project."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from copy import deepcopy

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "unet_vae"
    input_channels: int = 3
    latent_dim: int = 512
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 512])
    use_attention: bool = True
    use_skip_connections: bool = True
    dropout: float = 0.1
    pretrained_encoder: Optional[str] = None
    encoder_checkpoint: Optional[str] = None
    freeze_encoder_stages: int = 0


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    batch_size: int = 64
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    epochs: int = 60
    gradient_clip: float = 1.0
    kl_weight: float = 0.001
    perceptual_weight: float = 0.1
    adversarial_weight: float = 0.001
    encoder_lr: Optional[float] = None
    decoder_lr: Optional[float] = None
    warmup_epochs: int = 0
    unfreeze_schedule: Dict[str, int] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset: str = "celeba"
    data_path: str = "./assets/datasets"
    image_size: int = 256
    num_workers: int = 4
    augmentation: bool = True


@dataclass
class MaskConfig:
    """Masking strategy configuration."""
    type: str = "random"  # random, center, irregular
    mask_ratio: float = 0.4
    min_size: int = 32
    max_size: int = 128
    seed: int = 42
    cache_dir: str = "./assets/masks"


@dataclass
class LoggingConfig:
    """Experiment tracking and logging configuration."""
    use_wandb: bool = True
    use_tensorboard: bool = True
    log_interval: int = 100
    save_interval: int = 5
    sample_interval: int = 500
    checkpoint_dir: str = "./weights"
    log_dir: str = "./logs"


@dataclass
class Config:
    """Master configuration class combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def get_default() -> "Config":
        """Get default configuration."""
        return Config(
            model=ModelConfig(),
            training=TrainingConfig(),
            data=DataConfig(),
            mask=MaskConfig(),
            logging=LoggingConfig(),
        )

    @staticmethod
    def get_pretrained() -> "Config":
        """Get configuration for training with pretrained weights."""
        config = Config.get_default()
        
        # Model settings for pretrained
        config.model.pretrained_encoder = "resnet"
        config.model.encoder_checkpoint = None
        config.model.freeze_encoder_stages = 2
        
        # Modified training settings for transfer learning
        config.training.learning_rate = 0.0001
        config.training.encoder_lr = 0.00001
        config.training.decoder_lr = 0.0002
        config.training.warmup_epochs = 5
        config.training.unfreeze_schedule = {
            "epoch_5": 3,
            "epoch_10": 2,
            "epoch_15": 1,
            "epoch_20": 0,
        }
        
        return config

    def to_dict(self) -> Dict:
        """Convert config to dictionary (for logging/serialization)."""
        return asdict(self)

    def apply_overrides(self, **kwargs) -> None:
        """Apply command-line argument overrides to config.
        
        Args:
            **kwargs: Keyword arguments in format 'section.key=value'
                     e.g., training.batch_size=64, model.latent_dim=256
        """
        for key, value in kwargs.items():
            if '.' not in key:
                continue
            
            section, param = key.split('.', 1)
            if not hasattr(self, section):
                continue
            
            section_config = getattr(self, section)
            if hasattr(section_config, param):
                # Try to convert value to appropriate type
                current_value = getattr(section_config, param)
                if current_value is not None:
                    value_type = type(current_value)
                    try:
                        converted_value = value_type(value)
                        setattr(section_config, param, converted_value)
                    except (ValueError, TypeError):
                        setattr(section_config, param, value)
                else:
                    setattr(section_config, param, value)

    def copy(self) -> "Config":
        """Create a deep copy of the config."""
        return deepcopy(self)
