# Video Generation Models

A production-ready implementation of video generation using Generative Adversarial Networks (GANs). This project provides a complete framework for training, evaluating, and deploying video generation models with state-of-the-art techniques.

## Features

- **Modern GAN Architecture**: 3D convolutional networks with self-attention mechanisms
- **Advanced Training**: Spectral normalization, gradient penalty, exponential moving average (EMA)
- **Comprehensive Evaluation**: FVD, LPIPS, CLIP Score, temporal consistency metrics
- **Interactive Demo**: Streamlit-based web interface for video generation
- **Production Ready**: Type hints, comprehensive testing, CI/CD pipeline
- **Flexible Configuration**: YAML-based configuration with Hydra/OmegaConf support

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Video-Generation-Models.git
cd Video-Generation-Models
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p data logs checkpoints assets/samples
```

### Training

Train a video generation model:

```bash
python scripts/train.py --config configs/config.yaml
```

### Sampling

Generate videos from a trained model:

```bash
python scripts/sample.py --checkpoint checkpoints/best_model.pth --num_samples 16 --seed 42
```

### Evaluation

Evaluate model performance:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --num_samples 1000
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

## Project Structure

```
video-generation-models/
├── src/
│   ├── models/
│   │   └── video_gan.py          # GAN models with modern architectures
│   ├── data/
│   │   └── __init__.py           # Dataset implementations and loaders
│   └── utils/
│       ├── __init__.py           # Utility functions
│       └── evaluation.py         # Evaluation metrics
├── configs/
│   └── config.yaml               # Configuration file
├── scripts/
│   ├── train.py                  # Training script
│   ├── sample.py                 # Sampling script
│   └── evaluate.py               # Evaluation script
├── demo/
│   └── app.py                    # Streamlit demo
├── tests/
│   └── test_models.py            # Unit tests
├── assets/                       # Generated samples and visualizations
├── logs/                         # Training logs
├── checkpoints/                  # Model checkpoints
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore file
├── .pre-commit-config.yaml       # Pre-commit hooks
└── README.md                     # This file
```

## Model Architecture

### Generator

The generator uses a modern architecture with:

- **3D Convolutional Layers**: Proper handling of temporal dimensions
- **Self-Attention**: Captures long-range dependencies in video sequences
- **Batch Normalization**: Stabilizes training
- **Progressive Upsampling**: Generates high-quality videos

### Discriminator

The discriminator features:

- **3D Convolutional Layers**: Processes video sequences effectively
- **Spectral Normalization**: Prevents mode collapse
- **Gradient Penalty**: Enables stable training
- **Leaky ReLU**: Better gradient flow

### Training Techniques

- **Exponential Moving Average (EMA)**: Improves generation quality
- **Mixed Precision Training**: Faster training with lower memory usage
- **Gradient Clipping**: Prevents exploding gradients
- **Deterministic Seeding**: Reproducible results

## Datasets

### Moving MNIST

Synthetic dataset of moving MNIST digits. Automatically generated during training.

### Custom Video Dataset

Support for custom video datasets in common formats (MP4, AVI, MOV, MKV).

## Evaluation Metrics

### Fréchet Video Distance (FVD)
Measures the quality and diversity of generated videos by comparing feature distributions.

### LPIPS (Learned Perceptual Image Patch Similarity)
Perceptual similarity metric for individual frames.

### CLIP Score
Measures semantic quality using CLIP embeddings.

### Temporal Consistency
Evaluates smoothness and coherence across video frames.

### Diversity Score
Measures the variety of generated content.

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# Model configuration
model:
  generator:
    z_dim: 100
    num_frames: 16
    img_channels: 3
    img_size: 64
    hidden_dim: 512

# Training configuration
training:
  batch_size: 32
  num_epochs: 200
  learning_rate: 0.0002
  use_ema: true
  ema_decay: 0.999

# Data configuration
data:
  dataset_name: "moving_mnist"
  data_dir: "./data"
  num_workers: 4
```

## Usage Examples

### Basic Training

```python
from src.models.video_gan import VideoGenerator, VideoDiscriminator
from src.data import get_dataset, create_dataloader
from src.utils import set_seed, get_device

# Set random seed for reproducibility
set_seed(42)

# Load dataset
dataset = get_dataset("moving_mnist", num_frames=16, img_size=64)
dataloader = create_dataloader(dataset, batch_size=32)

# Initialize models
generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
discriminator = VideoDiscriminator(img_channels=3, img_size=64, num_frames=16)

# Training loop (simplified)
for epoch in range(num_epochs):
    for batch in dataloader:
        # Train discriminator and generator
        # ... training code ...
```

### Video Generation

```python
from src.models.video_gan import VideoGenerator
from src.utils import set_seed

# Load trained model
generator = VideoGenerator(z_dim=100, num_frames=16, img_size=64)
generator.load_state_dict(torch.load('checkpoints/best_model.pth'))
generator.eval()

# Generate videos
set_seed(42)
z = torch.randn(4, 100)  # 4 videos
videos = generator(z)

# Save as GIF
import imageio
frames = videos[0].permute(1, 2, 3, 0).numpy()
imageio.mimsave('generated_video.gif', frames, fps=10)
```

### Evaluation

```python
from src.utils.evaluation import VideoEvaluator

# Initialize evaluator
evaluator = VideoEvaluator(device)

# Evaluate model
results = evaluator.evaluate_model(
    generator=generator,
    real_videos=real_videos,
    num_samples=1000,
    metrics=['fvd', 'lpips', 'clip_score']
)

# Print results
evaluator.print_results(results)
```

## Advanced Features

### Mixed Precision Training

Enable mixed precision for faster training:

```yaml
training:
  mixed_precision: true
```

### Multi-GPU Training

Train on multiple GPUs:

```bash
python scripts/train.py --config configs/config.yaml --gpus 2
```

### Custom Datasets

Add your own video dataset:

```python
from src.data import VideoDataset

dataset = VideoDataset(
    data_dir="path/to/videos",
    num_frames=16,
    img_size=64
)
```

## Performance Tips

1. **Use Mixed Precision**: Reduces memory usage and speeds up training
2. **Adjust Batch Size**: Larger batches improve stability but require more memory
3. **Tune Learning Rates**: Different rates for generator and discriminator
4. **Monitor Metrics**: Use FVD and LPIPS to track progress
5. **Save Checkpoints**: Regular checkpointing prevents loss of progress

## Troubleshooting

### Common Issues

**Out of Memory**: Reduce batch size or enable mixed precision training.

**Mode Collapse**: Increase discriminator learning rate or add gradient penalty.

**Poor Quality**: Ensure proper normalization and use EMA for inference.

**Slow Training**: Use mixed precision and optimize data loading.

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_models.py::TestVideoGenerator::test_generator_forward
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{video_generation_models,
  title={Video Generation Models: A Modern GAN Implementation},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Video-Generation-Models}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- PyTorch Lightning for the training infrastructure
- Streamlit for the interactive demo interface
- The open-source community for various components and inspiration

## Model Card

### Intended Use

This model is designed for research and educational purposes in video generation. It can generate short video sequences from random noise vectors.

### Training Data

The model is trained on synthetic Moving MNIST data or custom video datasets. No real-world video data is included by default.

### Limitations

- Generates short video sequences (16 frames)
- Limited to 64x64 resolution
- May not generalize well to complex real-world scenes
- Requires significant computational resources for training

### Bias and Fairness

The model may inherit biases from the training data. Users should be aware of potential biases and evaluate generated content accordingly.

### Safety Considerations

- Generated content should be reviewed before use
- Not suitable for generating misleading or harmful content
- Users are responsible for appropriate use of generated videos
# Video-Generation-Models
