"""Video generation models using modern GAN architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange


class VideoGenerator(nn.Module):
    """Modern video generator with 3D convolutions and attention mechanisms.
    
    Args:
        z_dim: Dimension of the latent noise vector
        num_frames: Number of frames in generated videos
        img_channels: Number of image channels (RGB=3)
        img_size: Size of each frame (assumed square)
        hidden_dim: Hidden dimension for the network
        use_attention: Whether to use self-attention layers
    """
    
    def __init__(
        self,
        z_dim: int = 100,
        num_frames: int = 16,
        img_channels: int = 3,
        img_size: int = 64,
        hidden_dim: int = 512,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.num_frames = num_frames
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        
        # Initial projection from noise to feature volume
        self.init_size = img_size // 4
        self.init_channels = hidden_dim // 4
        
        self.fc = nn.Linear(z_dim, self.init_channels * self.init_size * self.init_size * num_frames)
        
        # 3D convolutional layers
        self.conv_blocks = nn.ModuleList([
            # 8x8x16 -> 16x16x16
            nn.Sequential(
                nn.ConvTranspose3d(self.init_channels, hidden_dim // 2, 4, 2, 1),
                nn.BatchNorm3d(hidden_dim // 2),
                nn.ReLU(inplace=True)
            ),
            # 16x16x16 -> 32x32x16
            nn.Sequential(
                nn.ConvTranspose3d(hidden_dim // 2, hidden_dim // 4, 4, 2, 1),
                nn.BatchNorm3d(hidden_dim // 4),
                nn.ReLU(inplace=True)
            ),
            # 32x32x16 -> 64x64x16
            nn.Sequential(
                nn.ConvTranspose3d(hidden_dim // 4, img_channels, 4, 2, 1),
                nn.Tanh()
            )
        ])
        
        # Optional self-attention layers
        if use_attention:
            self.attention = SelfAttention3D(hidden_dim // 2)
        else:
            self.attention = None
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate video from noise vector.
        
        Args:
            z: Noise tensor of shape (batch_size, z_dim)
            
        Returns:
            Generated video tensor of shape (batch_size, channels, frames, height, width)
        """
        batch_size = z.size(0)
        
        # Project noise to initial feature volume
        x = self.fc(z)
        x = x.view(batch_size, self.init_channels, self.num_frames, self.init_size, self.init_size)
        
        # Apply 3D convolutions
        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            
            # Apply attention after first conv block
            if i == 0 and self.attention is not None:
                x = self.attention(x)
        
        return x


class VideoDiscriminator(nn.Module):
    """Modern video discriminator with spectral normalization and progressive training.
    
    Args:
        img_channels: Number of image channels
        img_size: Size of each frame
        num_frames: Number of frames in videos
        hidden_dim: Hidden dimension for the network
        use_spectral_norm: Whether to use spectral normalization
    """
    
    def __init__(
        self,
        img_channels: int = 3,
        img_size: int = 64,
        num_frames: int = 16,
        hidden_dim: int = 512,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        
        # 3D convolutional layers
        conv_layers = [
            # 64x64x16 -> 32x32x8
            nn.Conv3d(img_channels, hidden_dim // 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32x8 -> 16x16x4
            nn.Conv3d(hidden_dim // 4, hidden_dim // 2, 4, 2, 1),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16x4 -> 8x8x2
            nn.Conv3d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm3d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8x2 -> 4x4x1
            nn.Conv3d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.BatchNorm3d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        
        # Apply spectral normalization if requested
        if use_spectral_norm:
            conv_layers = [apply_spectral_norm(layer) if isinstance(layer, nn.Conv3d) else layer 
                          for layer in conv_layers]
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4 * 4 * 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate between real and fake videos.
        
        Args:
            x: Video tensor of shape (batch_size, channels, frames, height, width)
            
        Returns:
            Probability tensor of shape (batch_size, 1)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SelfAttention3D(nn.Module):
    """3D self-attention layer for video generation."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to video features.
        
        Args:
            x: Feature tensor of shape (batch, channels, frames, height, width)
            
        Returns:
            Attended features of same shape
        """
        batch, channels, frames, height, width = x.shape
        
        # Reshape for attention: (batch, frames*height*width, channels)
        x_flat = x.permute(0, 2, 3, 4, 1).contiguous()
        x_flat = x_flat.view(batch, frames * height * width, channels)
        
        # Apply attention
        attended, _ = self.mha(x_flat, x_flat, x_flat)
        attended = self.norm(attended + x_flat)
        
        # Reshape back
        attended = attended.view(batch, frames, height, width, channels)
        attended = attended.permute(0, 4, 1, 2, 3).contiguous()
        
        return attended


def apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to a module."""
    return nn.utils.spectral_norm(module)


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP training.
    
    Args:
        discriminator: The discriminator model
        real_samples: Real video samples
        fake_samples: Fake video samples
        device: Device to compute on
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, 1).to(device)
    
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    d_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty
