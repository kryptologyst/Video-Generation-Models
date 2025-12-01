"""Evaluation metrics for video generation models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2
from pathlib import Path
import json


class VideoMetrics:
    """Collection of video generation evaluation metrics."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = {}
    
    def calculate_all_metrics(
        self,
        real_videos: torch.Tensor,
        fake_videos: torch.Tensor,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Calculate all specified metrics.
        
        Args:
            real_videos: Real video samples
            fake_videos: Fake video samples
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ['fvd', 'lpips', 'clip_score']
        
        results = {}
        
        for metric in metrics:
            if metric == 'fvd':
                results['fvd'] = self.calculate_fvd(real_videos, fake_videos)
            elif metric == 'lpips':
                results['lpips'] = self.calculate_lpips(real_videos, fake_videos)
            elif metric == 'clip_score':
                results['clip_score'] = self.calculate_clip_score(fake_videos)
            elif metric == 'inception_score':
                results['inception_score'] = self.calculate_inception_score(fake_videos)
            else:
                print(f"Unknown metric: {metric}")
        
        return results
    
    def calculate_fvd(
        self,
        real_videos: torch.Tensor,
        fake_videos: torch.Tensor
    ) -> float:
        """Calculate Fréchet Video Distance (FVD).
        
        Note: This is a simplified implementation. For production use,
        consider using a proper FVD implementation with I3D features.
        
        Args:
            real_videos: Real video samples
            fake_videos: Fake video samples
            
        Returns:
            FVD score
        """
        # Reshape videos to (batch, frames*channels, height, width)
        batch_size, channels, frames, height, width = real_videos.shape
        real_flat = real_videos.view(batch_size, channels * frames, height, width)
        fake_flat = fake_videos.view(batch_size, channels * frames, height, width)
        
        # Calculate mean and covariance
        real_mean = real_flat.mean(dim=0)
        fake_mean = fake_flat.mean(dim=0)
        
        # Calculate covariance matrices
        real_centered = real_flat - real_mean
        fake_centered = fake_flat - fake_mean
        
        real_cov = torch.mm(real_centered.T, real_centered) / (batch_size - 1)
        fake_cov = torch.mm(fake_centered.T, fake_centered) / (batch_size - 1)
        
        # Simplified Fréchet distance
        mean_diff = torch.norm(real_mean - fake_mean)
        cov_diff = torch.norm(real_cov - fake_cov)
        
        fvd = mean_diff + cov_diff
        return fvd.item()
    
    def calculate_lpips(
        self,
        real_videos: torch.Tensor,
        fake_videos: torch.Tensor
    ) -> float:
        """Calculate LPIPS (Learned Perceptual Image Patch Similarity).
        
        Args:
            real_videos: Real video samples
            fake_videos: Fake video samples
            
        Returns:
            LPIPS score
        """
        try:
            import lpips
            
            # Initialize LPIPS model
            lpips_model = lpips.LPIPS(net='alex').to(self.device)
            
            # Calculate LPIPS for each frame
            lpips_scores = []
            batch_size, channels, frames, height, width = real_videos.shape
            
            for frame_idx in range(frames):
                real_frame = real_videos[:, :, frame_idx, :, :]
                fake_frame = fake_videos[:, :, frame_idx, :, :]
                
                # Normalize to [-1, 1] for LPIPS
                real_frame = real_frame * 2.0 - 1.0
                fake_frame = fake_frame * 2.0 - 1.0
                
                with torch.no_grad():
                    lpips_score = lpips_model(real_frame, fake_frame)
                    lpips_scores.append(lpips_score.mean().item())
            
            return np.mean(lpips_scores)
        
        except ImportError:
            print("LPIPS not available. Install with: pip install lpips")
            return 0.0
    
    def calculate_clip_score(self, fake_videos: torch.Tensor) -> float:
        """Calculate CLIP score for video quality.
        
        Args:
            fake_videos: Fake video samples
            
        Returns:
            CLIP score
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Load CLIP model
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            model.to(self.device)
            model.eval()
            
            # Calculate CLIP score for each frame
            clip_scores = []
            batch_size, channels, frames, height, width = fake_videos.shape
            
            for frame_idx in range(frames):
                frame = fake_videos[:, :, frame_idx, :, :]
                
                # Convert to PIL images
                frame_np = frame.cpu().numpy()
                frame_np = np.clip(frame_np, 0, 1)
                frame_np = (frame_np * 255).astype(np.uint8)
                
                # Process with CLIP
                inputs = processor(images=frame_np, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    clip_score = outputs.logits_per_image.mean().item()
                    clip_scores.append(clip_score)
            
            return np.mean(clip_scores)
        
        except ImportError:
            print("CLIP not available. Install with: pip install transformers")
            return 0.0
    
    def calculate_inception_score(self, fake_videos: torch.Tensor) -> float:
        """Calculate Inception Score for video quality.
        
        Args:
            fake_videos: Fake video samples
            
        Returns:
            Inception Score
        """
        try:
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            # Load Inception model
            inception = models.inception_v3(pretrained=True)
            inception.eval()
            inception.to(self.device)
            
            # Transform for Inception
            transform = transforms.Compose([
                transforms.Resize(299),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Calculate IS for each frame
            is_scores = []
            batch_size, channels, frames, height, width = fake_videos.shape
            
            for frame_idx in range(frames):
                frame = fake_videos[:, :, frame_idx, :, :]
                
                # Normalize to [0, 1]
                frame = torch.clamp(frame, 0, 1)
                
                # Transform
                frame = transform(frame)
                
                with torch.no_grad():
                    logits = inception(frame)
                    probs = F.softmax(logits, dim=1)
                    
                    # Calculate IS
                    marginal = probs.mean(dim=0)
                    kl_div = (probs * (probs / marginal).log()).sum(dim=1)
                    is_score = kl_div.mean().exp().item()
                    is_scores.append(is_score)
            
            return np.mean(is_scores)
        
        except ImportError:
            print("torchvision not available")
            return 0.0
    
    def calculate_temporal_consistency(self, videos: torch.Tensor) -> float:
        """Calculate temporal consistency score.
        
        Args:
            videos: Video samples
            
        Returns:
            Temporal consistency score
        """
        batch_size, channels, frames, height, width = videos.shape
        
        # Calculate frame differences
        frame_diffs = []
        for i in range(frames - 1):
            frame1 = videos[:, :, i, :, :]
            frame2 = videos[:, :, i + 1, :, :]
            
            diff = torch.abs(frame1 - frame2).mean()
            frame_diffs.append(diff.item())
        
        # Temporal consistency is inverse of average frame difference
        avg_diff = np.mean(frame_diffs)
        consistency = 1.0 / (1.0 + avg_diff)
        
        return consistency
    
    def calculate_diversity_score(self, videos: torch.Tensor) -> float:
        """Calculate diversity score for generated videos.
        
        Args:
            videos: Video samples
            
        Returns:
            Diversity score
        """
        batch_size = videos.size(0)
        
        # Calculate pairwise distances
        distances = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                dist = torch.norm(videos[i] - videos[j]).item()
                distances.append(dist)
        
        return np.mean(distances)


class VideoEvaluator:
    """Video generation model evaluator."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.metrics = VideoMetrics(device)
    
    def evaluate_model(
        self,
        generator: nn.Module,
        real_videos: torch.Tensor,
        num_samples: int = 1000,
        batch_size: int = 32,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """Evaluate a video generation model.
        
        Args:
            generator: Trained generator model
            real_videos: Real video samples for comparison
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of evaluation results
        """
        generator.eval()
        
        # Generate fake videos
        fake_videos = []
        z_dim = next(generator.parameters()).size(0) if hasattr(generator, 'z_dim') else 100
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                z = torch.randn(current_batch_size, z_dim, device=self.device)
                batch_fake = generator(z)
                fake_videos.append(batch_fake)
        
        fake_videos = torch.cat(fake_videos, dim=0)
        
        # Calculate metrics
        results = self.metrics.calculate_all_metrics(
            real_videos[:num_samples],
            fake_videos,
            metrics
        )
        
        # Add additional metrics
        results['temporal_consistency'] = self.metrics.calculate_temporal_consistency(fake_videos)
        results['diversity'] = self.metrics.calculate_diversity_score(fake_videos)
        
        return results
    
    def save_results(self, results: Dict[str, float], filepath: str) -> None:
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results
            filepath: Path to save results
        """
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results in a formatted way.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for metric, value in results.items():
            print(f"{metric.upper():<25}: {value:.4f}")
        
        print("="*50)
