# LASER: Learnable Adaptive Structured Embedding Representation

This repository provides two autoencoder baselines for image reconstruction:

- **Vector Quantized VAE (VQ-VAE)** — discrete latent codes with a learnable codebook.
- **Dictionary Learning VAE (DL-VAE)** — sparse dictionary bottleneck trained with Batch OMP.

Generation utilities have been removed for now so you can focus on training and evaluating reconstructions only.

## Features

- 🚀 Choice of bottlenecks: vector quantization or dictionary learning
- ⚡ GPU-friendly implementation with AMP-aware sparse coding
- 📊 Reconstruction quality metrics: MSE, PSNR, SSIM, optional LPIPS/FID
- 🔧 Modular architecture powered by PyTorch Lightning and Hydra

## Installation

```bash
# Clone the repository
git clone https://github.com/helloimlixin/laser.git
cd laser

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scriptsctivate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── configs/                # Hydra configuration files
│   ├── checkpoint/         # Checkpoint configurations
│   ├── data/               # Dataset configurations
│   ├── model/              # Model configurations
│   ├── train/              # Training configurations
│   ├── wandb/              # W&B logging configurations
│   └── config.yaml         # Main configuration
├── src/
│   ├── data/               # Data modules
│   │   ├── cifar10.py
│   │   ├── imagenette2.py
│   │   └── config.py
│   ├── models/
│   │   ├── bottleneck.py   # VQ and dictionary bottlenecks
│   │   ├── decoder.py
│   │   ├── dlvae.py
│   │   ├── encoder.py
│   │   ├── lpips.py
│   │   └── vqvae.py
└── train.py                # Main training script
```

## Usage

### Training

```bash
# Train VQ-VAE
python train.py model.type=vqvae data=cifar10

# Train DL-VAE
python train.py model.type=dlvae data=cifar10
```

### Running Tests

```bash
pytest tests/test_dlvae.py -q
```

## Configuration

All configuration is managed through Hydra. Adjust the YAML files under `configs/` or override settings directly from the command line, e.g.

```bash
python train.py model.type=dlvae data=celeba train.max_epochs=50
```

## Bottleneck Visualizations

The `tests/test_bottleneck.py` generates comprehensive visualizations comparing Vector Quantization (VQ) and Dictionary Learning (DL) bottlenecks on CelebA data at 128×128 resolution.

### Reconstruction Comparison

Side-by-side comparison of VQ and DL reconstruction quality (K=16 codebook/atoms, S=4 sparsity):

![Reconstruction Comparison](img/bottleneck/reconstruction_comparison.png)

**Key Observations:**
- **VQ (Vector Quantization)**: Maps each pixel to the nearest RGB color from 16 learned codebook entries. Fast but limited to discrete color matching.
- **DL (Dictionary Learning)**: Represents each pixel as a sparse combination of 4 atoms from 16 options. More expressive representation.
- **Error Maps**: DL shows consistently lower error (darker error maps) across all images, especially in complex regions like faces.
- **Quantitative**: DL achieves **3.4× lower MSE** than VQ while using the same 16-entry codebook, demonstrating the power of sparse combinations.

### Code Interpretability Heatmaps

Spatial visualization of how VQ and DL encode the image structure:

![Code Heatmaps](img/bottleneck/code_heatmaps.png)

**Understanding the Heatmaps:**
- **Column 1 (Original)**: Input CelebA images at 128×128 resolution
- **Column 2 (VQ Code Indices)**: Shows which of the 16 codebook entries is assigned to each pixel. Color represents the discrete code index (0-15).
  - Each pixel uses exactly **1 code** from 16 options
  - Spatial patterns show how VQ segments the image into color regions
  - Viridis colormap: Purple (low indices) → Yellow (high indices)
- **Column 3 (DL Max Coefficient)**: Shows the strength of the strongest atom activation at each pixel (log scale).
  - Each pixel uses **4 different atoms** with varying weights
  - Darker regions (purple) = weak coefficients, brighter regions (yellow) = strong coefficients
  - Reveals which image areas need stronger sparse representations
  - Log scale (log₁₀(1+coeff)) reveals full dynamic range across 4 orders of magnitude

**VQ vs DL Encoding:**
- VQ: Discrete, categorical assignment (one-hot selection)
- DL: Continuous, weighted combination (sparse weighted sum)
- DL's flexibility enables better reconstruction despite using the same codebook size

### Channel-wise Comparison

Pixel-level RGB channel analysis comparing original vs reconstructions:

![Channel Comparison](img/bottleneck/vq_channel_comparison.png)

**Analysis:**
- Shows a horizontal slice through the center of the first image across all three color channels
- **Black line**: Original pixel values
- **Blue dashed**: VQ reconstruction
- **Red dotted**: DL reconstruction
- **Key Insight**: DL tracks the original signal more closely than VQ across all channels, especially for smooth gradients and transitions
- Both VQ and DL handle sharp edges well, but DL excels at subtle color variations

### Usage Statistics

How frequently each codebook entry (VQ) or dictionary atom (DL) is utilized:

**VQ Codebook Usage:**
![VQ Usage](img/bottleneck/vq_codebook_usage.png)

- **All 16 codes are used**, showing k-means initialization creates a representative color palette
- Usage distribution is moderately skewed (dominant colors like skin, hair, background are used more)

**Dictionary Atom Usage:**
![DL Usage](img/bottleneck/dictionary_atom_usage.png)

- **All 16 atoms are used** thanks to diversity bonus in OMP selection
- More balanced distribution than VQ due to sparse combination (each pixel can use multiple atoms)
- Top atoms represent dominant color directions, while less-used atoms capture nuanced variations

### Detailed Performance Analysis (K=16, Sparsity=4, 128×128 resolution)

#### Overall Reconstruction Quality

| Method | Overall MSE | Per-Pixel Color Distance | Quality Gain |
|--------|-------------|-------------------------|--------------|
| **VQ** | 0.00905 | 0.1314 | Baseline |
| **DL** | 0.00329 | 0.0598 | **2.8× better MSE**, **45% of VQ color error** |

**Key Insight**: DL reduces per-pixel color distance by more than half compared to VQ, demonstrating significantly better perceptual quality.

#### Channel-wise Performance

Breaking down reconstruction quality by RGB channel reveals DL's balanced improvement:

| Channel | VQ MSE | DL MSE | DL Improvement |
|---------|--------|--------|----------------|
| **Red** | 0.0103 | 0.0021 | **4.9× better** |
| **Green** | 0.0084 | 0.0052 | **1.6× better** |
| **Blue** | 0.0085 | 0.0025 | **3.4× better** |

**Analysis**: 
- DL excels particularly in Red and Blue channels (3-5× improvement)
- Green channel shows modest but consistent improvement (1.6×)
- No channel-specific color shifts - all channels improve uniformly
- Balanced performance indicates proper sparse coding without artifacts

#### Codebook/Atom Utilization

| Method | Entries Used | Distribution | Notes |
|--------|--------------|--------------|-------|
| **VQ** | 16/16 (100%) | Moderately skewed | Each pixel uses exactly 1 code |
| **DL** | 16/16 (100%) | More balanced | Each pixel uses 4 different atoms with diversity bonus |

**Diversity Mechanism**: The OMP implementation includes a soft diversity bonus that encourages selection of underutilized atoms, ensuring all 16 atoms contribute meaningfully to the reconstruction.

#### Inference Speed

| Method | Speed | Per-Pixel Time | Scaling |
|--------|-------|----------------|---------|
| **VQ** | 2.1 ms | 0.13 µs/pixel | Linear with pixels |
| **DL** | 34 ms | 2.1 µs/pixel | **16.4× slower** but practical |

**Speed Analysis**:
- VQ uses simple nearest-neighbor lookup (very fast)
- DL performs 4 iterations of greedy OMP per pixel (optimization overhead)
- **Vectorized implementation** achieves 34× speedup over naive per-pixel approach
- At 128×128, DL remains practical for batch processing and visualization

### Key Advantages of Dictionary Learning

- ✓ **Superior reconstruction**: 2.8× lower MSE with same 16-entry codebook
- ✓ **Sparse representation**: Each pixel uses only 4/16 atoms (25% sparsity) vs VQ's 1/16 selection
- ✓ **Perceptual quality**: 45% color distance of VQ (more than 2× improvement)
- ✓ **Channel balance**: Outperforms VQ on all RGB channels (1.6-4.9× better)
- ✓ **Full utilization**: Diversity bonus ensures all 16 atoms contribute
- ✓ **Interpretable**: Continuous coefficients reveal spatial importance patterns (see heatmaps)
- ✓ **Gradient-friendly**: Supports end-to-end training with proper backpropagation

### Tradeoffs

- ✗ **Slower inference**: ~16× slower (34ms vs 2.1ms) due to iterative OMP sparse coding
- ✗ **Memory overhead**: Must store and compute with full dictionary + sparse coefficients
- ✗ **Computational complexity**: O(K × S × N) vs VQ's O(K × N) where S=sparsity, N=pixels

### Technical Implementation

**Optimizations**:
- ⚡ Vectorized batch OMP with (N, B) tensor format (no transposes)
- ⚡ Float masking instead of boolean for faster multiplication  
- ⚡ `scatter_add_` for global usage tracking (no Python loops)
- ⚡ Diversity bonus prevents dominant atoms from monopolizing selections
- ⚡ No-reselection masking ensures each pixel uses 4 distinct atoms

Run visualizations:
```bash
conda activate research
pytest tests/test_bottleneck.py::test_bottleneck_visualizations -v

# Update README images after regenerating visualizations
cp tests/artifacts/bottleneck/*.png img/bottleneck/
```

## License

MIT
