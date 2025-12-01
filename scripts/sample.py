"""Sampling script for video generation models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import Optional, Tuple, List
import argparse
import yaml
from tqdm import tqdm

from src.models.video_gan import VideoGenerator, VideoDiscriminator, EMA
from src.utils import set_seed, get_device, denormalize_video


class VideoSampler:
    """Video sampling utility for trained models."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device or get_device(config['device'])
        
        # Load model
        self.generator = VideoGenerator(**config['model']['generator'])
        self.discriminator = VideoDiscriminator(**config['model']['discriminator'])
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA if available
        if config['training']['use_ema'] and 'ema_state_dict' in checkpoint:
            self.ema = EMA(self.generator, config['training']['ema_decay'])
            self.ema.shadow = checkpoint['ema_state_dict']
            self.ema.apply_shadow()
        
        self.generator.to(self.device)
        self.generator.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Using device: {self.device}")
    
    def generate_videos(
        self,
        num_samples: int = 16,
        seed: Optional[int] = None,
        use_ema: bool = True
    ) -> torch.Tensor:
        """Generate video samples.
        
        Args:
            num_samples: Number of videos to generate
            seed: Random seed for reproducibility
            use_ema: Whether to use EMA parameters
            
        Returns:
            Generated video tensor
        """
        if seed is not None:
            set_seed(seed)
        
        # Use EMA if available
        if use_ema and hasattr(self, 'ema'):
            self.ema.apply_shadow()
        
        with torch.no_grad():
            z = torch.randn(num_samples, self.config['model']['generator']['z_dim'], device=self.device)
            videos = self.generator(z)
        
        # Restore original parameters
        if use_ema and hasattr(self, 'ema'):
            self.ema.restore()
        
        return videos
    
    def interpolate_videos(
        self,
        num_steps: int = 10,
        seed1: Optional[int] = None,
        seed2: Optional[int] = None
    ) -> torch.Tensor:
        """Interpolate between two videos.
        
        Args:
            num_steps: Number of interpolation steps
            seed1: Seed for first video
            seed2: Seed for second video
            
        Returns:
            Interpolated videos tensor
        """
        if seed1 is not None:
            set_seed(seed1)
        z1 = torch.randn(1, self.config['model']['generator']['z_dim'], device=self.device)
        
        if seed2 is not None:
            set_seed(seed2)
        z2 = torch.randn(1, self.config['model']['generator']['z_dim'], device=self.device)
        
        # Interpolate in latent space
        alphas = torch.linspace(0, 1, num_steps).view(-1, 1).to(self.device)
        interpolated_z = alphas * z1 + (1 - alphas) * z2
        
        with torch.no_grad():
            videos = self.generator(interpolated_z)
        
        return videos
    
    def save_videos_as_gif(
        self,
        videos: torch.Tensor,
        output_path: str,
        fps: int = 10,
        duration: Optional[float] = None
    ) -> None:
        """Save videos as GIF files.
        
        Args:
            videos: Video tensor of shape (batch, channels, frames, height, width)
            output_path: Output file path
            fps: Frames per second
            duration: Duration per frame in seconds
        """
        videos_np = videos.cpu().numpy()
        videos_np = denormalize_video(torch.from_numpy(videos_np)).numpy()
        videos_np = np.clip(videos_np, 0, 1)
        
        # Convert to uint8
        videos_np = (videos_np * 255).astype(np.uint8)
        
        # Create GIF for each video
        for i, video in enumerate(videos_np):
            frames = []
            for frame_idx in range(video.shape[1]):
                frame = video[:, frame_idx, :, :].transpose(1, 2, 0)
                frames.append(frame)
            
            # Save as GIF
            gif_path = output_path.replace('.gif', f'_{i}.gif')
            self._save_gif(frames, gif_path, fps, duration)
    
    def _save_gif(self, frames: List[np.ndarray], path: str, fps: int, duration: Optional[float]) -> None:
        """Save frames as GIF."""
        try:
            import imageio
            duration = duration or (1.0 / fps)
            imageio.mimsave(path, frames, duration=duration)
        except ImportError:
            print("imageio not available, saving as individual frames")
            # Fallback: save as individual frames
            frames_dir = Path(path).parent / f"{Path(path).stem}_frames"
            frames_dir.mkdir(exist_ok=True)
            
            for i, frame in enumerate(frames):
                frame_path = frames_dir / f"frame_{i:03d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def save_videos_as_mp4(
        self,
        videos: torch.Tensor,
        output_path: str,
        fps: int = 10
    ) -> None:
        """Save videos as MP4 files.
        
        Args:
            videos: Video tensor
            output_path: Output file path
            fps: Frames per second
        """
        videos_np = videos.cpu().numpy()
        videos_np = denormalize_video(torch.from_numpy(videos_np)).numpy()
        videos_np = np.clip(videos_np, 0, 1)
        
        # Convert to uint8
        videos_np = (videos_np * 255).astype(np.uint8)
        
        # Create MP4 for each video
        for i, video in enumerate(videos_np):
            frames = []
            for frame_idx in range(video.shape[1]):
                frame = video[:, frame_idx, :, :].transpose(1, 2, 0)
                frames.append(frame)
            
            # Save as MP4
            mp4_path = output_path.replace('.mp4', f'_{i}.mp4')
            self._save_mp4(frames, mp4_path, fps)
    
    def _save_mp4(self, frames: List[np.ndarray], path: str, fps: int) -> None:
        """Save frames as MP4."""
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def create_sample_grid(
        self,
        videos: torch.Tensor,
        nrow: int = 4,
        save_path: Optional[str] = None
    ) -> None:
        """Create a grid of video frames for visualization.
        
        Args:
            videos: Video tensor
            nrow: Number of videos per row
            save_path: Path to save the grid image
        """
        videos_np = videos.cpu().numpy()
        videos_np = denormalize_video(torch.from_numpy(videos_np)).numpy()
        
        # Take first frame of each video
        first_frames = videos_np[:, :, 0, :, :]  # (batch, channels, height, width)
        
        # Create grid
        fig, axes = plt.subplots(
            (len(videos) + nrow - 1) // nrow,
            nrow,
            figsize=(nrow * 2, (len(videos) + nrow - 1) // nrow * 2)
        )
        
        if len(videos) == 1:
            axes = [axes]
        elif len(videos) <= nrow:
            axes = axes.reshape(1, -1)
        
        for i, frame in enumerate(first_frames):
            row = i // nrow
            col = i % nrow
            
            if len(videos) > nrow:
                ax = axes[row, col]
            else:
                ax = axes[col]
            
            ax.imshow(frame.transpose(1, 2, 0))
            ax.set_title(f'Video {i+1}')
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(videos), len(axes.flatten())):
            row = i // nrow
            col = i % nrow
            if len(videos) > nrow:
                axes[row, col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grid saved to {save_path}")
        
        plt.show()


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description='Sample videos from trained model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of videos to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='assets/samples', help='Output directory')
    parser.add_argument('--format', type=str, choices=['gif', 'mp4', 'both'], default='gif', help='Output format')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--interpolate', action='store_true', help='Generate interpolation video')
    parser.add_argument('--interpolation_steps', type=int, default=20, help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize sampler
    sampler = VideoSampler(args.checkpoint, config)
    
    if args.interpolate:
        # Generate interpolation
        print("Generating interpolation video...")
        videos = sampler.interpolate_videos(
            num_steps=args.interpolation_steps,
            seed1=args.seed,
            seed2=args.seed + 1
        )
        
        if args.format in ['gif', 'both']:
            sampler.save_videos_as_gif(
                videos,
                str(output_dir / f'interpolation_seed_{args.seed}.gif'),
                fps=args.fps
            )
        
        if args.format in ['mp4', 'both']:
            sampler.save_videos_as_mp4(
                videos,
                str(output_dir / f'interpolation_seed_{args.seed}.mp4'),
                fps=args.fps
            )
    
    else:
        # Generate random videos
        print(f"Generating {args.num_samples} videos...")
        videos = sampler.generate_videos(
            num_samples=args.num_samples,
            seed=args.seed
        )
        
        # Create sample grid
        sampler.create_sample_grid(
            videos,
            nrow=min(4, args.num_samples),
            save_path=str(output_dir / f'sample_grid_seed_{args.seed}.png')
        )
        
        # Save individual videos
        if args.format in ['gif', 'both']:
            sampler.save_videos_as_gif(
                videos,
                str(output_dir / f'samples_seed_{args.seed}.gif'),
                fps=args.fps
            )
        
        if args.format in ['mp4', 'both']:
            sampler.save_videos_as_mp4(
                videos,
                str(output_dir / f'samples_seed_{args.seed}.mp4'),
                fps=args.fps
            )
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
