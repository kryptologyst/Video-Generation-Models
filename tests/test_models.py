"""Unit tests for video generation models."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.models.video_gan import VideoGenerator, VideoDiscriminator, EMA, gradient_penalty
from src.data import MovingMNISTDataset, VideoDataset, get_dataset
from src.utils import set_seed, get_device, count_parameters, normalize_video, denormalize_video
from src.utils.evaluation import VideoMetrics, VideoEvaluator


class TestVideoGenerator:
    """Test cases for VideoGenerator."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = VideoGenerator(
            z_dim=100,
            num_frames=16,
            img_channels=3,
            img_size=64,
            hidden_dim=512
        )
        
        assert generator.z_dim == 100
        assert generator.num_frames == 16
        assert generator.img_channels == 3
        assert generator.img_size == 64
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
        
        batch_size = 4
        z = torch.randn(batch_size, 100)
        
        output = generator(z)
        
        assert output.shape == (batch_size, 3, 16, 64, 64)
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_generator_with_attention(self):
        """Test generator with attention mechanism."""
        generator = VideoGenerator(
            z_dim=100,
            num_frames=16,
            img_size=64,
            use_attention=True
        )
        
        batch_size = 2
        z = torch.randn(batch_size, 100)
        
        output = generator(z)
        
        assert output.shape == (batch_size, 3, 16, 64, 64)
        assert generator.attention is not None


class TestVideoDiscriminator:
    """Test cases for VideoDiscriminator."""
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        discriminator = VideoDiscriminator(
            img_channels=3,
            img_size=64,
            num_frames=16,
            hidden_dim=512
        )
        
        assert discriminator.img_channels == 3
        assert discriminator.img_size == 64
        assert discriminator.num_frames == 16
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = VideoDiscriminator(img_channels=3, img_size=64, num_frames=16)
        
        batch_size = 4
        video = torch.randn(batch_size, 3, 16, 64, 64)
        
        output = discriminator(video)
        
        assert output.shape == (batch_size, 1)
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_discriminator_with_spectral_norm(self):
        """Test discriminator with spectral normalization."""
        discriminator = VideoDiscriminator(
            img_channels=3,
            img_size=64,
            num_frames=16,
            use_spectral_norm=True
        )
        
        batch_size = 2
        video = torch.randn(batch_size, 3, 16, 64, 64)
        
        output = discriminator(video)
        
        assert output.shape == (batch_size, 1)


class TestEMA:
    """Test cases for EMA."""
    
    def test_ema_initialization(self):
        """Test EMA initialization."""
        generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
        ema = EMA(generator, decay=0.999)
        
        assert ema.decay == 0.999
        assert len(ema.shadow) > 0
    
    def test_ema_update(self):
        """Test EMA update."""
        generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
        ema = EMA(generator, decay=0.999)
        
        # Store original parameters
        original_params = {name: param.data.clone() for name, param in generator.named_parameters()}
        
        # Update parameters
        for param in generator.parameters():
            param.data += torch.randn_like(param.data) * 0.1
        
        # Update EMA
        ema.update()
        
        # Check that shadow parameters are updated
        for name, shadow_param in ema.shadow.items():
            assert not torch.equal(shadow_param, original_params[name])
    
    def test_ema_apply_restore(self):
        """Test EMA apply and restore."""
        generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
        ema = EMA(generator, decay=0.999)
        
        # Store original parameters
        original_params = {name: param.data.clone() for name, param in generator.named_parameters()}
        
        # Apply shadow parameters
        ema.apply_shadow()
        
        # Check that parameters are changed
        for name, param in generator.named_parameters():
            assert torch.equal(param.data, ema.shadow[name])
        
        # Restore original parameters
        ema.restore()
        
        # Check that parameters are restored
        for name, param in generator.named_parameters():
            assert torch.equal(param.data, original_params[name])


class TestGradientPenalty:
    """Test cases for gradient penalty."""
    
    def test_gradient_penalty(self):
        """Test gradient penalty calculation."""
        discriminator = VideoDiscriminator(img_channels=3, img_size=64, num_frames=16)
        device = torch.device('cpu')
        
        batch_size = 4
        real_samples = torch.randn(batch_size, 3, 16, 64, 64)
        fake_samples = torch.randn(batch_size, 3, 16, 64, 64)
        
        penalty = gradient_penalty(discriminator, real_samples, fake_samples, device)
        
        assert penalty.item() >= 0.0
        assert penalty.requires_grad


class TestDatasets:
    """Test cases for datasets."""
    
    def test_moving_mnist_dataset(self):
        """Test MovingMNISTDataset."""
        dataset = MovingMNISTDataset(
            num_digits=1,
            num_frames=16,
            img_size=64,
            num_samples=100
        )
        
        assert len(dataset) == 100
        
        # Test getting a sample
        sample = dataset[0]
        
        assert sample.shape == (1, 16, 64, 64)
        assert sample.min() >= -1.0
        assert sample.max() <= 1.0
    
    def test_get_dataset(self):
        """Test get_dataset function."""
        dataset = get_dataset(
            dataset_name="moving_mnist",
            num_frames=16,
            img_size=64,
            num_samples=50
        )
        
        assert len(dataset) == 50
        
        sample = dataset[0]
        assert sample.shape == (1, 16, 64, 64)


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        torch.manual_seed(42)
        
        # Generate random numbers
        rand1 = torch.randn(10)
        
        set_seed(42)
        rand2 = torch.randn(10)
        
        assert torch.equal(rand1, rand2)
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device("auto")
        assert isinstance(device, torch.device)
        
        device = get_device("cpu")
        assert device.type == "cpu"
    
    def test_count_parameters(self):
        """Test parameter counting."""
        generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
        num_params = count_parameters(generator)
        
        assert num_params > 0
        assert isinstance(num_params, int)
    
    def test_normalize_denormalize(self):
        """Test video normalization and denormalization."""
        video = torch.rand(2, 3, 16, 64, 64)
        
        normalized = normalize_video(video)
        denormalized = denormalize_video(normalized)
        
        assert torch.allclose(video, denormalized, atol=1e-6)


class TestEvaluation:
    """Test cases for evaluation metrics."""
    
    def test_video_metrics_initialization(self):
        """Test VideoMetrics initialization."""
        device = torch.device('cpu')
        metrics = VideoMetrics(device)
        
        assert metrics.device == device
    
    def test_fvd_calculation(self):
        """Test FVD calculation."""
        device = torch.device('cpu')
        metrics = VideoMetrics(device)
        
        batch_size = 4
        real_videos = torch.randn(batch_size, 3, 16, 64, 64)
        fake_videos = torch.randn(batch_size, 3, 16, 64, 64)
        
        fvd = metrics.calculate_fvd(real_videos, fake_videos)
        
        assert fvd >= 0.0
        assert isinstance(fvd, float)
    
    def test_temporal_consistency(self):
        """Test temporal consistency calculation."""
        device = torch.device('cpu')
        metrics = VideoMetrics(device)
        
        batch_size = 4
        videos = torch.randn(batch_size, 3, 16, 64, 64)
        
        consistency = metrics.calculate_temporal_consistency(videos)
        
        assert 0.0 <= consistency <= 1.0
        assert isinstance(consistency, float)
    
    def test_diversity_score(self):
        """Test diversity score calculation."""
        device = torch.device('cpu')
        metrics = VideoMetrics(device)
        
        batch_size = 4
        videos = torch.randn(batch_size, 3, 16, 64, 64)
        
        diversity = metrics.calculate_diversity_score(videos)
        
        assert diversity >= 0.0
        assert isinstance(diversity, float)


class TestIntegration:
    """Integration tests."""
    
    def test_training_step(self):
        """Test a single training step."""
        generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
        discriminator = VideoDiscriminator(img_channels=3, img_size=64, num_frames=16)
        
        batch_size = 4
        real_videos = torch.randn(batch_size, 3, 16, 64, 64)
        
        # Generator forward
        z = torch.randn(batch_size, 100)
        fake_videos = generator(z)
        
        # Discriminator forward
        real_outputs = discriminator(real_videos)
        fake_outputs = discriminator(fake_videos)
        
        assert real_outputs.shape == (batch_size, 1)
        assert fake_outputs.shape == (batch_size, 1)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = {
            'model': {
                'generator': {
                    'z_dim': 100,
                    'num_frames': 16,
                    'img_channels': 3,
                    'img_size': 64,
                    'hidden_dim': 512
                },
                'discriminator': {
                    'img_channels': 3,
                    'img_size': 64,
                    'num_frames': 16,
                    'hidden_dim': 512
                }
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.0002,
                'use_ema': True,
                'ema_decay': 0.999
            }
        }
        
        # Test that models can be created from config
        generator = VideoGenerator(**config['model']['generator'])
        discriminator = VideoDiscriminator(**config['model']['discriminator'])
        
        assert generator.z_dim == 100
        assert discriminator.img_channels == 3


if __name__ == "__main__":
    pytest.main([__file__])
