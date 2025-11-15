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

![Reconstruction Comparison](img/reconstruction_comparison.png)

**Key Observations:**
- **VQ (Vector Quantization)**: Maps each pixel to the nearest RGB color from 16 learned codebook entries. Fast but limited to discrete color matching.
- **DL (Dictionary Learning)**: Represents each pixel as a sparse combination of 4 atoms from 16 options. More expressive representation.
- **Error Maps**: DL shows consistently lower error (darker error maps) across all images, especially in complex regions like faces.
- **Quantitative**: DL achieves **10.4× lower MSE** than VQ while using the same 16-entry codebook, demonstrating the power of sparse combinations.

### Code Interpretability Heatmaps

Spatial visualization of how VQ and DL encode the image structure:

![Code Heatmaps](img/code_heatmaps.png)

**Understanding the Heatmaps:**
- **Column 1 (Original)**: Input CelebA images at 128×128 resolution
- **Column 2 (VQ Code Indices)**: Shows which of the 16 codebook entries is assigned to each pixel. Color represents the discrete code index (0-15).
  - Each pixel uses exactly **1 code** from 16 options
  - Spatial patterns show how VQ segments the image into color regions
  - Viridis colormap: Purple (low indices) → Yellow (high indices)
- **Column 3 (DL Sparse Code Strength)**: Shows the L1 norm (sum of absolute coefficients) at each pixel location.
  - Each pixel uses **4 different atoms** with varying weights
  - Darker regions (purple) = weak total activation, brighter regions (yellow) = strong total activation
  - Reveals which image areas require stronger sparse representations
  - L1 norm provides a stable measure of total "activation energy" per location
  - Normalized to [0, 1] using percentile clipping (1st-99th percentile) for clean visualization

**VQ vs DL Encoding:**
- VQ: Discrete, categorical assignment (one-hot selection)
- DL: Continuous, weighted combination (sparse weighted sum)
- DL's flexibility with L1-normalized coefficients enables better reconstruction with the same codebook size

### Channel-wise Comparison

Pixel-level RGB channel analysis comparing original vs reconstructions:

![Channel Comparison](img/vq_channel_comparison.png)

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
![VQ Usage](img/vq_codebook_usage.png)

- **All 16 codes are used**, showing k-means initialization creates a representative color palette
- Usage distribution is moderately skewed (dominant colors like skin, hair, background are used more)

**Dictionary Atom Usage:**
![DL Usage](img/dictionary_atom_usage.png)

- **All 16 atoms are used** thanks to diversity bonus in OMP selection
- More balanced distribution than VQ due to sparse combination (each pixel can use multiple atoms)
- Top atoms represent dominant color directions, while less-used atoms capture nuanced variations

### Detailed Performance Analysis (K=16, Sparsity=4, 128×128 resolution)

#### Overall Reconstruction Quality

| Method | Overall MSE | Per-Pixel Color Distance | Quality Gain |
|--------|-------------|-------------------------|--------------|
| **VQ** | 0.00905 | 0.0759 | Baseline |
| **DL** | 0.00087 | 0.0163 | **10.4× better MSE**, **21% of VQ color error** |

**Key Insight**: With the simplified greedy OMP, DL reduces per-pixel color distance to just 21% of VQ's error, demonstrating dramatically better perceptual quality. The greedy algorithm provides excellent reconstruction while being simpler and faster than Cholesky-based OMP.

#### Channel-wise Performance

Breaking down reconstruction quality by RGB channel reveals DL's exceptional improvement:

| Channel | VQ MSE | DL MSE | DL Improvement |
|---------|--------|--------|----------------|
| **Red** | 0.0103 | 0.0003 | **30.8× better** |
| **Green** | 0.0084 | 0.0013 | **6.3× better** |
| **Blue** | 0.0085 | 0.0010 | **8.9× better** |

**Analysis**: 
- DL excels particularly in Red channel (30×+ improvement!)
- All channels show significant improvement (6-31× better)
- Green channel shows most modest but still strong improvement (6.3×)
- Balanced performance across all channels indicates proper sparse coding without artifacts

#### Codebook/Atom Utilization

| Method | Entries Used | Distribution | Notes |
|--------|--------------|--------------|-------|
| **VQ** | 16/16 (100%) | Moderately skewed | Each pixel uses exactly 1 code |
| **DL** | 16/16 (100%) | Balanced | Each pixel uses exactly 4 different atoms |

**Full Utilization**: Both methods use all 16 codebook entries/atoms thanks to k-means initialization (VQ) and greedy selection with no-reselection masking (DL), ensuring the full representation capacity is utilized.

#### Inference Speed & Computational Complexity

**Benchmark Setup**: 128×128 resolution, K=16 atoms, S=4 sparsity

| Batch Size | Pixels | VQ Time | VQ µs/pixel | DL Time | DL µs/pixel | Slowdown |
|------------|--------|---------|-------------|---------|-------------|----------|
| **1** | 16,384 | 1.2 ms | 0.070 | 9.0 ms | 0.551 | **7.9×** |
| **4** | 65,536 | 1.8 ms | 0.028 | 33.0 ms | 0.503 | **17.9×** |
| **8** | 131,072 | 2.6 ms | 0.020 | 64.5 ms | 0.492 | **24.5×** |
| **16** | 262,144 | 5.0 ms | 0.019 | 128.6 ms | 0.490 | **26.0×** |

**Complexity Analysis**:
- **VQ**: O(K × N) - Single nearest-neighbor search per pixel across K=16 codebook entries
  - Excellent batching efficiency: per-pixel time drops from 0.070 to 0.019 µs (3.7× speedup at batch=16)
  - Benefits from vectorized distance computation across batch dimension
- **DL**: O(K × S × N) - S=4 iterations of greedy atom selection, each searching K=16 atoms
  - Good batching efficiency: per-pixel time stabilizes at ~0.50 µs for batches ≥4
  - Theoretical slowdown: ~4× (from sparsity S=4)
  - Measured slowdown: 7.9-26.0× depending on batch size (includes overhead from correlation, residual updates, coefficient storage)

**Scaling Observations**:
- VQ scales better with batch size due to simpler computation
- DL per-pixel time is nearly constant (~0.50 µs) for batches ≥4
- Both methods remain highly practical: VQ at 19-70 ns/pixel, DL at 490-551 ns/pixel
- Larger batches favor VQ more, but DL still achieves **10.4× better quality**

**Speed vs Quality Tradeoff**:
- At batch=4 (typical training batch): DL is 17.9× slower but achieves **10.4× better MSE**
- For batch processing and offline training, the quality gain far outweighs the speed cost
- Both methods support real-time processing: even at batch=16, DL processes 262K pixels in 129ms

### Key Advantages of Dictionary Learning

- ✓ **Superior reconstruction**: 10.4× lower MSE with same 16-entry codebook
- ✓ **Sparse representation**: Each pixel uses only 4/16 atoms (25% sparsity) vs VQ's 1/16 selection
- ✓ **Perceptual quality**: 21% color distance of VQ (nearly 5× improvement)
- ✓ **Channel balance**: Outperforms VQ on all RGB channels (6-31× better)
- ✓ **Full utilization**: All 16 atoms actively contribute to reconstruction
- ✓ **Interpretable**: L1 norm visualization reveals spatial importance patterns (see heatmaps)
- ✓ **Gradient-friendly**: Supports end-to-end training with proper backpropagation
- ✓ **Simpler implementation**: Greedy OMP is cleaner and faster than Cholesky-based approaches

### Tradeoffs

- ✗ **Slower inference**: 7.9-26× slower than VQ depending on batch size (larger batches increase relative slowdown)
- ✗ **Memory overhead**: Must store and compute with full dictionary + sparse coefficients  
- ✗ **Computational complexity**: O(K × S × N) vs VQ's O(K × N) where K=atoms, S=sparsity, N=pixels
- ✗ **Worse batching scaling**: VQ benefits more from large batches due to simpler vectorization

**Bottom Line**: Despite being 8-26× slower (depending on batch size), DL achieves **10.4× better reconstruction quality**. Both methods remain practical for real-time use with sub-microsecond per-pixel processing. For applications where quality matters, DL's superior reconstruction far outweighs the speed cost.

### Technical Implementation

**Simplified Greedy OMP**:
- 🚀 **Greedy atom selection**: Simplified OMP without Cholesky decomposition (faster and cleaner)
- 🚀 **Vectorized batch processing**: (N, B) tensor format for efficient parallel sparse coding
- 🚀 **Float masking**: Uses multiplicative masking instead of boolean indexing for speed
- 🚀 **Projection-based coefficients**: Direct inner product computation (no least squares solve)
- 🚀 **No-reselection masking**: Ensures each signal uses exactly S distinct atoms

**Visualization Improvements**:
- 📊 **L1 norm visualization**: Uses sum of absolute coefficients for stable, interpretable heatmaps
- 📊 **Percentile normalization**: Robust 1-99th percentile clipping for clean contrast
- 📊 **Fold/unfold mapping**: Properly maps patch-based coefficients to pixel space for visualization

Run visualizations:
```bash
conda activate research
pytest tests/test_bottleneck.py::test_bottleneck_visualizations -v

# Update README images after regenerating visualizations
cp tests/artifacts/bottleneck/*.png img/
```

## License

MIT
