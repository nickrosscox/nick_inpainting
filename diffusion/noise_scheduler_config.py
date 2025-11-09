# diffusion_config.py
from dataclasses import dataclass

@dataclass
class NoiseConfig:
    """Configuration for diffusion model training and sampling."""
    
    # Noise scheduler parameters
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: str = 'linear'  # 'linear' or 'cosine'
    
    # Time embedding dimension
    time_emb_dim: int = 256
    
    # Sampling parameters
    num_inference_steps: int = None  # None = use all timesteps
    
    def __post_init__(self):
        """Validate configuration."""
        if self.schedule_type not in ['linear', 'cosine']:
            raise ValueError(f"schedule_type must be 'linear' or 'cosine', got {self.schedule_type}")
        
        if self.num_inference_steps is None:
            self.num_inference_steps = self.num_timesteps