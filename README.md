# LASER: Learning Adaptive Sparse Representations for Image Compression

The [original development repository](https://anonymous.4open.science/r/dlgan-0CCD) is a little bit messy, so I decided to create a new one with a clean structure.

## Overview

A PyTorch Lightning implementation of two neural compression approaches:
- Vector Quantized VAE (VQ-VAE) with EMA codebook updates
- Dictionary Learning VAE (DL-VAE) with gradient-based dictionary learning

## Features

- 🚀 Two complementary compression approaches:
  - VQ-VAE with EMA codebook updates (referenced from src/models/bottleneck.py, lines 9-68)
  - DL-VAE with adaptive sparse coding (referenced from src/models/bottleneck.py, lines 257-291)
- ⚡ Efficient implementation:
  - Vectorized batch OMP for fast sparse coding
  - Direct gradient updates for dictionary learning
  - GPU-optimized matrix operations
- 📊 Comprehensive evaluation metrics:
  - PSNR & SSIM for reconstruction quality
  - LPIPS for perceptual quality
  - Optional FID score computation
- 🔧 Clean, modular architecture:
  - PyTorch Lightning for organized training
  - Hydra for configuration management
  - Weights & Biases logging

## Usage

Train VQ-VAE:

```bash
python train.py model=vqvae
```

Train DL-VAE:

```bash
python train.py model=dlvae
```

## Configuration

Configuration is managed using Hydra. The configuration files are located in the `configs` directory.

## Model Architecture

### VQ-VAE
- Encoder network with residual blocks
- Vector quantization bottleneck with EMA codebook updates
- Decoder network with skip connections

### DL-VAE
- Similar encoder-decoder architecture
- Dictionary learning bottleneck with:
  - Adaptive sparse coding via batch OMP
  - Direct gradient updates for dictionary learning
  - L1 regularization for sparsity control
- Commitment loss for training stability

## License

MIT


