"""Evaluation script for video generation models."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm

from src.models.video_gan import VideoGenerator, VideoDiscriminator
from src.data import get_dataset, create_dataloader
from src.utils import set_seed, get_device, setup_logging
from src.utils.evaluation import VideoEvaluator


def evaluate_model(
    checkpoint_path: str,
    config: dict,
    num_samples: int = 1000,
    batch_size: int = 32,
    metrics: list = None,
    output_dir: str = "evaluation_results"
) -> None:
    """Evaluate a trained video generation model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        num_samples: Number of samples to generate for evaluation
        batch_size: Batch size for generation
        metrics: List of metrics to calculate
        output_dir: Directory to save results
    """
    # Setup
    set_seed(config['seed'])
    logger = setup_logging()
    device = get_device(config['device'])
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    generator = VideoGenerator(**config['model']['generator'])
    discriminator = VideoDiscriminator(**config['model']['discriminator'])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    
    generator.to(device)
    generator.eval()
    
    # Load dataset for real videos
    logger.info("Loading dataset for evaluation")
    dataset = get_dataset(
        dataset_name=config['data']['dataset_name'],
        data_dir=config['data']['data_dir'],
        num_frames=config['model']['generator']['num_frames'],
        img_size=config['model']['generator']['img_size']
    )
    
    # Get real videos
    real_videos = []
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    logger.info("Collecting real video samples")
    for batch in tqdm(dataloader, desc="Loading real videos"):
        real_videos.append(batch)
        if len(real_videos) * batch_size >= num_samples:
            break
    
    real_videos = torch.cat(real_videos, dim=0)[:num_samples]
    real_videos = real_videos.to(device)
    
    # Initialize evaluator
    evaluator = VideoEvaluator(device)
    
    # Run evaluation
    logger.info(f"Evaluating model with {num_samples} samples")
    results = evaluator.evaluate_model(
        generator=generator,
        real_videos=real_videos,
        num_samples=num_samples,
        batch_size=batch_size,
        metrics=metrics
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    results_file = output_dir / f"evaluation_results_{Path(checkpoint_path).stem}.json"
    evaluator.save_results(results, str(results_file))
    
    # Generate sample videos for visual inspection
    logger.info("Generating sample videos for visual inspection")
    with torch.no_grad():
        z = torch.randn(16, config['model']['generator']['z_dim'], device=device)
        sample_videos = generator(z)
        
        # Save samples
        samples_file = output_dir / f"sample_videos_{Path(checkpoint_path).stem}.npy"
        np.save(samples_file, sample_videos.cpu().numpy())
        logger.info(f"Sample videos saved to {samples_file}")
    
    logger.info("Evaluation completed!")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate video generation model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--metrics', nargs='+', default=['fvd', 'lpips', 'clip_score'], 
                       help='Metrics to calculate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run evaluation
    evaluate_model(
        checkpoint_path=args.checkpoint,
        config=config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        metrics=args.metrics,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
