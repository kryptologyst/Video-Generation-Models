#!/usr/bin/env python3
"""Quick training script to test the setup."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from pathlib import Path

from src.models.video_gan import VideoGenerator, VideoDiscriminator
from src.data import get_dataset, create_dataloader
from src.utils import set_seed, get_device, setup_logging


def quick_test():
    """Quick test of the video generation setup."""
    print("Testing Video Generation Setup")
    print("=" * 40)
    
    # Set random seed
    set_seed(42)
    
    # Setup logging
    logger = setup_logging()
    
    # Get device
    device = get_device('auto')
    logger.info(f"Using device: {device}")
    
    # Load config
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.error("Config file not found")
        return
    
    # Initialize models
    logger.info("Initializing models...")
    generator = VideoGenerator(**config['model']['generator'])
    discriminator = VideoDiscriminator(**config['model']['discriminator'])
    
    generator.to(device)
    discriminator.to(device)
    
    logger.info(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = get_dataset(
        dataset_name=config['data']['dataset_name'],
        num_frames=config['model']['generator']['num_frames'],
        img_size=config['model']['generator']['img_size'],
        num_samples=100  # Small dataset for testing
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=0,  # No multiprocessing for testing
        pin_memory=False
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Test forward pass
    logger.info("Testing forward pass...")
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        # Test generator
        z = torch.randn(2, config['model']['generator']['z_dim'], device=device)
        fake_videos = generator(z)
        logger.info(f"Generated videos shape: {fake_videos.shape}")
        
        # Test discriminator
        real_videos = next(iter(dataloader)).to(device)
        real_outputs = discriminator(real_videos)
        fake_outputs = discriminator(fake_videos)
        
        logger.info(f"Real outputs shape: {real_outputs.shape}")
        logger.info(f"Fake outputs shape: {fake_outputs.shape}")
        
        # Check output ranges
        logger.info(f"Real outputs range: [{real_outputs.min():.3f}, {real_outputs.max():.3f}]")
        logger.info(f"Fake outputs range: [{fake_outputs.min():.3f}, {fake_outputs.max():.3f}]")
    
    # Test training step
    logger.info("Testing training step...")
    generator.train()
    discriminator.train()
    
    # Simple training step
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = torch.nn.BCELoss()
    
    real_videos = next(iter(dataloader)).to(device)
    batch_size = real_videos.size(0)
    
    # Train discriminator
    optimizer_d.zero_grad()
    
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)
    
    real_outputs = discriminator(real_videos)
    d_loss_real = criterion(real_outputs, real_labels)
    
    z = torch.randn(batch_size, config['model']['generator']['z_dim'], device=device)
    fake_videos = generator(z)
    fake_outputs = discriminator(fake_videos.detach())
    d_loss_fake = criterion(fake_outputs, fake_labels)
    
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_d.step()
    
    # Train generator
    optimizer_g.zero_grad()
    
    fake_outputs = discriminator(fake_videos)
    g_loss = criterion(fake_outputs, real_labels)
    
    g_loss.backward()
    optimizer_g.step()
    
    logger.info(f"Discriminator loss: {d_loss.item():.4f}")
    logger.info(f"Generator loss: {g_loss.item():.4f}")
    
    logger.info("All tests passed! Setup is working correctly.")
    print("=" * 40)
    print("SUCCESS: Video generation setup is working!")
    print("You can now run:")
    print("  python scripts/train.py")
    print("  python scripts/sample.py --checkpoint <checkpoint_path>")
    print("  streamlit run demo/app.py")


if __name__ == "__main__":
    quick_test()
