"""Interactive Streamlit demo for video generation."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import yaml
from typing import Optional
import tempfile
import base64

from src.models.video_gan import VideoGenerator, VideoDiscriminator, EMA
from src.utils import set_seed, get_device, denormalize_video


@st.cache_resource
def load_model(checkpoint_path: str, config: dict):
    """Load the trained model."""
    device = get_device(config['device'])
    
    # Initialize models
    generator = VideoGenerator(**config['model']['generator'])
    discriminator = VideoDiscriminator(**config['model']['discriminator'])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['model_state_dict'])
    
    # Load EMA if available
    if config['training']['use_ema'] and 'ema_state_dict' in checkpoint:
        ema = EMA(generator, config['training']['ema_decay'])
        ema.shadow = checkpoint['ema_state_dict']
        ema.apply_shadow()
    
    generator.to(device)
    generator.eval()
    
    return generator, device


def generate_videos(generator, device, num_samples, seed, z_dim):
    """Generate video samples."""
    if seed is not None:
        set_seed(seed)
    
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=device)
        videos = generator(z)
    
    return videos


def videos_to_gif(videos, fps=10):
    """Convert videos to GIF format."""
    videos_np = videos.cpu().numpy()
    videos_np = denormalize_video(torch.from_numpy(videos_np)).numpy()
    videos_np = np.clip(videos_np, 0, 1)
    videos_np = (videos_np * 255).astype(np.uint8)
    
    # Create GIF for first video
    video = videos_np[0]
    frames = []
    for frame_idx in range(video.shape[1]):
        frame = video[:, frame_idx, :, :].transpose(1, 2, 0)
        frames.append(frame)
    
    # Save as temporary GIF
    with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp_file:
        try:
            import imageio
            imageio.mimsave(tmp_file.name, frames, duration=1.0/fps)
            return tmp_file.name
        except ImportError:
            st.error("imageio not available. Please install it: pip install imageio")
            return None


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Video Generation Demo",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Generation Demo")
    st.markdown("Generate videos using trained GAN models")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    checkpoint_path = st.sidebar.file_uploader(
        "Upload model checkpoint",
        type=['pth', 'ckpt'],
        help="Upload a trained model checkpoint file"
    )
    
    if checkpoint_path is None:
        st.warning("Please upload a model checkpoint to continue")
        st.stop()
    
    # Load config
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Config file not found. Please ensure configs/config.yaml exists.")
        st.stop()
    
    # Load model
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            tmp_file.write(checkpoint_path.getvalue())
            tmp_file.flush()
            
            generator, device = load_model(tmp_file.name, config)
            st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    
    num_samples = st.sidebar.slider("Number of videos", 1, 16, 4)
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=2**32-1)
    fps = st.sidebar.slider("FPS", 5, 30, 10)
    
    # Generate videos
    if st.sidebar.button("Generate Videos", type="primary"):
        with st.spinner("Generating videos..."):
            videos = generate_videos(
                generator,
                device,
                num_samples,
                seed,
                config['model']['generator']['z_dim']
            )
            
            st.success(f"Generated {num_samples} videos!")
            
            # Display videos
            st.subheader("Generated Videos")
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Grid View", "Individual Videos", "Download"])
            
            with tab1:
                # Create grid of first frames
                videos_np = videos.cpu().numpy()
                videos_np = denormalize_video(torch.from_numpy(videos_np)).numpy()
                
                fig, axes = plt.subplots(
                    (num_samples + 3) // 4,
                    4,
                    figsize=(12, 3 * ((num_samples + 3) // 4))
                )
                
                if num_samples == 1:
                    axes = [axes]
                elif num_samples <= 4:
                    axes = axes.reshape(1, -1)
                
                for i in range(num_samples):
                    row = i // 4
                    col = i % 4
                    
                    if num_samples > 4:
                        ax = axes[row, col]
                    else:
                        ax = axes[col]
                    
                    # Show first frame
                    frame = videos_np[i, :, 0, :, :].transpose(1, 2, 0)
                    ax.imshow(frame)
                    ax.set_title(f'Video {i+1}')
                    ax.axis('off')
                
                # Hide empty subplots
                for i in range(num_samples, len(axes.flatten())):
                    row = i // 4
                    col = i % 4
                    if num_samples > 4:
                        axes[row, col].axis('off')
                    else:
                        axes[col].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with tab2:
                # Show individual videos
                for i in range(num_samples):
                    st.subheader(f"Video {i+1}")
                    
                    # Create GIF for this video
                    single_video = videos[i:i+1]
                    gif_path = videos_to_gif(single_video, fps)
                    
                    if gif_path:
                        with open(gif_path, "rb") as f:
                            gif_data = f.read()
                        
                        st.image(gif_data, caption=f"Video {i+1}")
            
            with tab3:
                # Download options
                st.subheader("Download Videos")
                
                # Create GIF for first video
                gif_path = videos_to_gif(videos[:1], fps)
                
                if gif_path:
                    with open(gif_path, "rb") as f:
                        gif_data = f.read()
                    
                    st.download_button(
                        label="Download Sample Video (GIF)",
                        data=gif_data,
                        file_name=f"generated_video_seed_{seed}.gif",
                        mime="image/gif"
                    )
                
                # Save as numpy array
                videos_np = videos.cpu().numpy()
                np_bytes = np.save(videos_np)
                
                st.download_button(
                    label="Download All Videos (NPY)",
                    data=np_bytes,
                    file_name=f"generated_videos_seed_{seed}.npy",
                    mime="application/octet-stream"
                )
    
    # Interpolation section
    st.sidebar.subheader("Video Interpolation")
    
    if st.sidebar.button("Generate Interpolation"):
        with st.spinner("Generating interpolation..."):
            # Generate two videos
            seed1 = seed
            seed2 = seed + 1
            
            set_seed(seed1)
            z1 = torch.randn(1, config['model']['generator']['z_dim'], device=device)
            
            set_seed(seed2)
            z2 = torch.randn(1, config['model']['generator']['z_dim'], device=device)
            
            # Interpolate
            num_steps = 20
            alphas = torch.linspace(0, 1, num_steps).view(-1, 1).to(device)
            interpolated_z = alphas * z1 + (1 - alphas) * z2
            
            with torch.no_grad():
                interpolated_videos = generator(interpolated_z)
            
            st.success("Generated interpolation video!")
            
            # Display interpolation
            st.subheader("Video Interpolation")
            st.markdown(f"Interpolating between seed {seed1} and seed {seed2}")
            
            # Create GIF
            gif_path = videos_to_gif(interpolated_videos, fps)
            
            if gif_path:
                with open(gif_path, "rb") as f:
                    gif_data = f.read()
                
                st.image(gif_data, caption="Interpolation Video")
                
                st.download_button(
                    label="Download Interpolation Video",
                    data=gif_data,
                    file_name=f"interpolation_seed_{seed1}_{seed2}.gif",
                    mime="image/gif"
                )
    
    # Model information
    st.sidebar.subheader("Model Information")
    st.sidebar.text(f"Device: {device}")
    st.sidebar.text(f"Z-dim: {config['model']['generator']['z_dim']}")
    st.sidebar.text(f"Frames: {config['model']['generator']['num_frames']}")
    st.sidebar.text(f"Image size: {config['model']['generator']['img_size']}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with PyTorch, Streamlit, and modern GAN architectures")


if __name__ == "__main__":
    main()
