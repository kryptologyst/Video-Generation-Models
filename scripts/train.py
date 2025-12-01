"""Training script for video generation models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path

from src.models.video_gan import VideoGenerator, VideoDiscriminator, EMA, gradient_penalty
from src.data import get_dataset, create_dataloader
from src.utils import set_seed, get_device, setup_logging, count_parameters


class VideoGAN(pl.LightningModule):
    """PyTorch Lightning module for video GAN training."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.z_dim = config['model']['generator']['z_dim']
        self.num_frames = config['model']['generator']['num_frames']
        self.img_size = config['model']['generator']['img_size']
        
        # Initialize models
        self.generator = VideoGenerator(**config['model']['generator'])
        self.discriminator = VideoDiscriminator(**config['model']['discriminator'])
        
        # Initialize EMA
        if config['training']['use_ema']:
            self.ema = EMA(self.generator, config['training']['ema_decay'])
        else:
            self.ema = None
        
        # Loss functions
        self.criterion = nn.BCELoss()
        
        # Training parameters
        self.d_steps_per_g_step = config['training']['d_steps_per_g_step']
        self.gradient_penalty_weight = config['training']['gradient_penalty_weight']
        
        # Validation metrics
        self.val_step_outputs = []
        
        # Log model parameters
        self.log_model_info()
    
    def log_model_info(self):
        """Log model information."""
        g_params = count_parameters(self.generator)
        d_params = count_parameters(self.discriminator)
        
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")
        print(f"Total parameters: {g_params + d_params:,}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate videos from noise."""
        return self.generator(z)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int) -> torch.Tensor:
        """Training step."""
        real_videos = batch
        
        if optimizer_idx == 0:  # Discriminator
            return self._discriminator_step(real_videos)
        elif optimizer_idx == 1:  # Generator
            return self._generator_step(real_videos)
    
    def _discriminator_step(self, real_videos: torch.Tensor) -> torch.Tensor:
        """Discriminator training step."""
        batch_size = real_videos.size(0)
        device = real_videos.device
        
        # Real labels
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        # Train on real videos
        real_outputs = self.discriminator(real_videos)
        d_loss_real = self.criterion(real_outputs, real_labels)
        
        # Train on fake videos
        z = torch.randn(batch_size, self.z_dim, device=device)
        fake_videos = self.generator(z).detach()
        fake_outputs = self.discriminator(fake_videos)
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        
        # Gradient penalty
        if self.gradient_penalty_weight > 0:
            gp = gradient_penalty(self.discriminator, real_videos, fake_videos, device)
            d_loss = d_loss_real + d_loss_fake + self.gradient_penalty_weight * gp
        else:
            d_loss = d_loss_real + d_loss_fake
        
        # Logging
        self.log('train/d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/d_loss_real', d_loss_real, on_step=True, on_epoch=True)
        self.log('train/d_loss_fake', d_loss_fake, on_step=True, on_epoch=True)
        
        return d_loss
    
    def _generator_step(self, real_videos: torch.Tensor) -> torch.Tensor:
        """Generator training step."""
        batch_size = real_videos.size(0)
        device = real_videos.device
        
        # Fake labels (trick discriminator)
        real_labels = torch.ones(batch_size, 1, device=device)
        
        # Generate fake videos
        z = torch.randn(batch_size, self.z_dim, device=device)
        fake_videos = self.generator(z)
        fake_outputs = self.discriminator(fake_videos)
        
        # Generator loss
        g_loss = self.criterion(fake_outputs, real_labels)
        
        # Logging
        self.log('train/g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return g_loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        real_videos = batch
        batch_size = real_videos.size(0)
        device = real_videos.device
        
        # Generate fake videos
        z = torch.randn(batch_size, self.z_dim, device=device)
        fake_videos = self.generator(z)
        
        # Calculate losses
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        
        real_outputs = self.discriminator(real_videos)
        fake_outputs = self.discriminator(fake_videos)
        
        d_loss_real = self.criterion(real_outputs, real_labels)
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        g_loss = self.criterion(fake_outputs, real_labels)
        
        # Store outputs for epoch-end processing
        self.val_step_outputs.append({
            'real_videos': real_videos,
            'fake_videos': fake_videos,
            'd_loss': d_loss,
            'g_loss': g_loss
        })
        
        return {
            'val/d_loss': d_loss,
            'val/g_loss': g_loss
        }
    
    def on_validation_epoch_end(self) -> None:
        """Validation epoch end."""
        if not self.val_step_outputs:
            return
        
        # Calculate average losses
        avg_d_loss = torch.stack([x['d_loss'] for x in self.val_step_outputs]).mean()
        avg_g_loss = torch.stack([x['g_loss'] for x in self.val_step_outputs]).mean()
        
        self.log('val/d_loss_epoch', avg_d_loss, on_epoch=True, prog_bar=True)
        self.log('val/g_loss_epoch', avg_g_loss, on_epoch=True, prog_bar=True)
        
        # Generate sample videos
        if self.current_epoch % 10 == 0:
            self._generate_samples()
        
        self.val_step_outputs.clear()
    
    def _generate_samples(self) -> None:
        """Generate sample videos for visualization."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(4, self.z_dim, device=self.device)
            fake_videos = self.generator(z)
            
            # Save samples
            samples_dir = Path("assets/samples")
            samples_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert to numpy and save
            fake_videos_np = fake_videos.cpu().numpy()
            np.save(samples_dir / f"epoch_{self.current_epoch}.npy", fake_videos_np)
        
        self.train()
    
    def configure_optimizers(self) -> Tuple[optim.Optimizer, optim.Optimizer]:
        """Configure optimizers."""
        lr = self.config['training']['learning_rate']
        beta1 = self.config['training']['beta1']
        beta2 = self.config['training']['beta2']
        weight_decay = self.config['training']['weight_decay']
        
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        
        return d_optimizer, g_optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update EMA after each training batch."""
        if self.ema is not None:
            self.ema.update()


def train_model(config: Dict[str, Any]) -> None:
    """Train the video generation model."""
    # Set random seed
    set_seed(config['seed'])
    
    # Setup logging
    logger = setup_logging()
    
    # Get device
    device = get_device(config['device'])
    logger.info(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = get_dataset(
        dataset_name=config['data']['dataset_name'],
        data_dir=config['data']['data_dir'],
        num_frames=config['model']['generator']['num_frames'],
        img_size=config['model']['generator']['img_size']
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    # Initialize model
    model = VideoGAN(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='video-gan-{epoch:02d}-{val/g_loss_epoch:.4f}',
            monitor='val/g_loss_epoch',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # Setup logger
    if config['logging']['use_wandb']:
        pl_logger = WandbLogger(
            project=config['logging']['wandb_project'],
            name=f"video-gan-{config['seed']}"
        )
    else:
        pl_logger = TensorBoardLogger(
            save_dir=config['logging']['log_dir'],
            name="video-gan"
        )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        devices=1,
        accelerator='auto',
        precision='16-mixed' if config['mixed_precision'] else 32,
        callbacks=callbacks,
        logger=pl_logger,
        deterministic=config['deterministic'],
        log_every_n_steps=config['logging']['log_every_n_steps']
    )
    
    # Train model
    trainer.fit(model, dataloader)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Train model
    train_model(config)
