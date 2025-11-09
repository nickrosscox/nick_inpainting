import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np

# from models.pretrained_encoders import (
#     PretrainedResNetEncoder,
#     PretrainedVAEEncoder,
#     PretrainedStyleGANEncoder
# )

class UNetDiffusion(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: List[int] = None,
        use_attention: bool = True,
        use_skip_connections: bool = True,
        pretrained_encoder: Optional[str] = None,
        encoder_checkpoint: Optional[str] = None,
        freeze_encoder_stages: int = 0,
        input_size: int = 256,
        attention_resolutions: List[int] = None
    ):
        super().__init__()
        
        self.use_skip_connections = use_skip_connections
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]
        print(f"UNetDiffusion initialized with hidden_dims: {hidden_dims}")
        
        # Set default attention resolutions if not provided
        if attention_resolutions is None:
            attention_resolutions = [16]  # Default: only at 16x16
        
        # Calculate which encoder blocks should have attention
        attention_blocks = []
        for i in range(len(hidden_dims)):
            resolution = input_size // (2 ** (i + 1))
            if resolution in attention_resolutions:
                attention_blocks.append(i)
        
        print(f"Applying attention at encoder blocks: {attention_blocks} (resolutions: {[input_size // (2 ** (i + 1)) for i in attention_blocks]})")

        # ========== TIME EMBEDDING (MINIMAL VERSION) ==========
        time_emb_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        # ---------- ENCODER SELECTION ----------
        # if pretrained_encoder == "resnet":
        #     self.encoder = PretrainedResNetEncoder(
        #         model_name="resnet50",
        #         pretrained="imagenet",
        #         frozen_stages=freeze_encoder_stages,
        #         output_channels=hidden_dims,
        #     )
        # elif pretrained_encoder == "vggface":
        #     self.encoder = PretrainedResNetEncoder(
        #         model_name="resnet50",
        #         pretrained="vggface2",
        #         frozen_stages=freeze_encoder_stages,
        #         output_channels=hidden_dims,
        #     )
        # elif pretrained_encoder == "vae" and encoder_checkpoint:
        #     self.encoder = PretrainedVAEEncoder(
        #         checkpoint_path=encoder_checkpoint,
        #         frozen=(freeze_encoder_stages > 0),
        #     )
        # elif pretrained_encoder == "stylegan":
        #     self.encoder = PretrainedStyleGANEncoder(
        #         model_path=encoder_checkpoint,
        #         frozen_layers=freeze_encoder_stages,
        #     )
        # else:
        self.encoder = UNetEncoder(
            input_channels=input_channels,
            hidden_dims=hidden_dims,
            time_emb_dim=time_emb_dim,
            use_attention=use_attention,
            attention_blocks=attention_blocks
        )
        
        # Decoder
        self.decoder = UNetDecoder(
            output_channels=input_channels,
            hidden_dims=list(reversed(hidden_dims)),
            time_emb_dim=time_emb_dim,
            use_attention=use_attention,
            attention_blocks=attention_blocks
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Compute time embedding ONCE
        t_emb = self.time_mlp(t)  # [B, 256]
        
        # apply mask
        masked_input = x * (1 - mask)
        # Concatenate image with mask
        x_input = torch.cat([masked_input, mask], dim=1)
        
        # Pass time embedding to encoder and decoder
        features, skip_connections = self.encoder(x_input, t_emb)
        predicted_noise = self.decoder(features, t_emb, skip_connections)
        
        return predicted_noise


class NoiseScheduler:
    """
    Noise scheduler for diffusion process.
    Defines the beta schedule and computes derived quantities.
    """
    
    def __init__(
        self,
        num_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = 'linear'
    ):
        """
        Args:
            num_timesteps: Total number of diffusion steps (T)
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule_type: 'linear' or 'cosine'
        """
        self.num_timesteps = num_timesteps
        
        # Create beta schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Compute alpha values
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)  # Cumulative product
        
        # For sampling (reverse process)
        self.alpha_bars_prev = np.append(1.0, self.alphas_bars[:-1])
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def to(self, device):
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.alpha_bars_prev = self.alpha_bars_prev.to(device)
        return self
    
    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor, mask: torch.Tensor):
        """
        Add noise to the masked region of the image at timestep t.
        
        Args:
            x_0: Original clean image [B, C, H, W]
            t: Timestep for each sample in batch [B]
            mask: Binary mask [B, 1, H, W], 1 = region to inpaint
        
        Returns:
            x_t: Noisy masked image [B, C, H, W]
            noise: The noise that was added [B, C, H, W]
        """
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Get alpha_bar for the given timesteps
        alpha_bar_t = self.alpha_bars[t]  # [B]
        
        # Reshape for broadcasting: [B] -> [B, 1, 1, 1]
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        
        # Forward diffusion: q(x_t | x_0)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # Apply noise only to masked region
        noisy_region = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        x_t = mask * noisy_region + (1 - mask) * x_0
        
        return x_t, noise


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for timesteps.
    Used to encode timestep information in diffusion models.
    Based on the positional encoding from "Attention is All You Need".
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timesteps, shape [B] - integers from 0 to num_timesteps
        
        Returns:
            embeddings: Sinusoidal embeddings, shape [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        
        # Create frequency spectrum
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Apply to timesteps
        embeddings = time[:, None] * embeddings[None, :]
        
        # Concatenate sin and cos
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings


class UNetEncoder(nn.Module):
    """U-Net style encoder with skip connections."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_dims: List[int],
        time_emb_dim: int,
        use_attention: bool = True,
        attention_blocks: List[int] = None
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        if attention_blocks is None:
            attention_blocks = []
        
        in_channels = input_channels + 1  # +1 for mask channel
        
        for i, out_channels in enumerate(hidden_dims):
            self.blocks.append(
                EncoderBlock(in_channels, out_channels, time_emb_dim)
            )
            
            # Only apply attention at specified block indices
            if use_attention and i in attention_blocks:
                self.attention_blocks.append(
                    SelfAttention(out_channels)
                )
            else:
                self.attention_blocks.append(nn.Identity())
                
            in_channels = out_channels
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        #(f"[DEBUG] Encoder received: {x.shape}")
        skip_connections = []
        
        for block, attention in zip(self.blocks, self.attention_blocks):
            x = block(x, t_emb)  # Pass time embedding
            x = attention(x)
            skip_connections.append(x)
            
        return x, skip_connections[:-1]


class UNetDecoder(nn.Module):
    """U-Net style decoder with skip connections."""
    
    def __init__(
        self,
        output_channels: int,
        hidden_dims: List[int],
        time_emb_dim: int,
        use_attention: bool = True,
        attention_blocks: List[int] = None
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        # Default to no attention if not specified
        if attention_blocks is None:
            attention_blocks = []
        
        # Mirror encoder attention
        decoder_attention_indices = [len(hidden_dims) - 1 - i for i in attention_blocks]
        
        # Create decoder blocks to match encoder blocks
        for i in range(len(hidden_dims)):
            if i == 0:
                in_channels = hidden_dims[0]
                out_channels = hidden_dims[0] if len(hidden_dims) == 1 else hidden_dims[1]
            elif i == len(hidden_dims) - 1:
                in_channels = hidden_dims[i] * 2  # With skip connection
                out_channels = hidden_dims[i]
            else:
                in_channels = hidden_dims[i] * 2
                out_channels = hidden_dims[i + 1]
            
            self.blocks.append(
                DecoderBlock(in_channels, out_channels, time_emb_dim)
            )
            
            # Apply attention at mirrored positions
            if use_attention and i in decoder_attention_indices:
                self.attention_blocks.append(SelfAttention(out_channels))
            else:
                self.attention_blocks.append(nn.Identity())
        
        # Final 1x1 conv to get output channels
        self.final_conv = nn.Conv2d(hidden_dims[-1], output_channels, 1)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, skip_connections: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            skip_connections: List of skip connection tensors
        """
        if skip_connections is not None:
            skip_connections = list(reversed(skip_connections))
        
        for i, (block, attention) in enumerate(zip(self.blocks, self.attention_blocks)):
            if skip_connections is not None and i > 0 and i <= len(skip_connections):
                # Concatenate skip connection
                skip = skip_connections[i - 1]
                # Resize if necessary
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
                
            x = block(x, t_emb)
            x = attention(x)
        
        return self.final_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Time projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # Add time embedding
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Time projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # Add time embedding
        x = x + self.time_proj(t_emb)[:, :, None, None]
        x = self.conv2(x)
        return x


class SelfAttention(nn.Module):
    """Self-attention module for capturing long-range dependencies."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        
        # Generate query, key, value
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)
        
        # Attention
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection with learnable weight
        return x + self.gamma * out