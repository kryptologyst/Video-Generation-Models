#!/usr/bin/env python3
"""Setup script for video generation project."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Setting up Video Generation Project")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    directories = [
        "data",
        "logs", 
        "checkpoints",
        "assets/samples",
        "evaluation_results"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run quick test
    print("\nRunning quick test...")
    if not run_command("python scripts/quick_test.py", "Quick test"):
        print("Quick test failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("SUCCESS: Setup completed!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py")
    print("2. Generate samples: python scripts/sample.py --checkpoint <path>")
    print("3. Launch demo: streamlit run demo/app.py")
    print("4. Run tests: pytest tests/")


if __name__ == "__main__":
    main()
