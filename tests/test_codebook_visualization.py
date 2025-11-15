"""
Test for visualizing VQ codebooks and DL dictionary atoms in RGB space.

This test creates 2D embeddings of the learned representations using:
- PCA (Principal Component Analysis): Linear projection
- t-SNE (t-Distributed Stochastic Neighbor Embedding): Non-linear, preserves local structure
- UMAP (Uniform Manifold Approximation and Projection): Non-linear, faster than t-SNE
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Ensure project sources are importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.bottleneck import DictionaryLearning, VectorQuantizer
from src.data.celeba import CelebADataModule
from src.data.config import DataConfig

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "codebook_embeddings"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _load_celeba_batch(batch_size=4, image_size=128):
    """Load a batch of CelebA images."""
    celeba_dir = Path.home() / "data" / "celeba"
    if not celeba_dir.exists():
        return None
    
    data_config = DataConfig(
        name="celeba",
        data_dir=str(celeba_dir),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
    )
    dm = CelebADataModule(data_config)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    
    # Normalize to [-1, 1] range (typical for neural networks)
    x = (x - 0.5) * 2.0
    return x


def _kmeans_codebook(data: torch.Tensor, num_embeddings: int, iters: int = 10) -> torch.Tensor:
    """Initialize codebook using k-means clustering."""
    B, C, H, W = data.shape
    flat = data.permute(0, 2, 3, 1).reshape(-1, C)
    
    # Random initialization
    indices = torch.randperm(flat.size(0))[:num_embeddings]
    codebook = flat[indices].clone()
    
    # K-means iterations
    for _ in range(iters):
        # Assign to nearest centroid
        distances = torch.cdist(flat, codebook)
        assignments = distances.argmin(dim=1)
        
        # Update centroids
        for k in range(num_embeddings):
            mask = assignments == k
            if mask.sum() > 0:
                codebook[k] = flat[mask].mean(dim=0)
    
    return codebook  # [K, C] for VQ embedding


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn and umap-learn not installed")
def test_visualize_codebook_embeddings():
    """
    Visualize VQ codebook and DL dictionary atoms in 2D RGB space.
    
    This test trains both VQ and DL on CelebA data, then projects their
    learned representations (codebook vectors / dictionary atoms) into 2D
    using PCA, t-SNE, and UMAP for interpretability.
    """
    print("\n" + "="*70)
    print("CODEBOOK VISUALIZATION IN RGB SPACE")
    print("="*70)
    
    # Load data
    celeba_dir = Path.home() / "data" / "celeba"
    if celeba_dir.exists():
        z = _load_celeba_batch(batch_size=16, image_size=128)
        print("\nUsing CelebA images for training")
    else:
        z = torch.randn(16, 3, 128, 128)
        print("\nWarning: Using random noise (CelebA not available)")
    
    B, C, H, W = z.shape
    K = 64  # Use more atoms for better visualization
    
    # Initialize VQ with k-means
    print(f"\nInitializing {K} codebook vectors with k-means...")
    codebook = _kmeans_codebook(z, K)  # [K, C]
    vq = VectorQuantizer(num_embeddings=K, embedding_dim=C, commitment_cost=0.25, decay=0.99)
    vq.embedding.weight.data.copy_(codebook)
    
    # Initialize DL with k-means (pixel-level)
    dl = DictionaryLearning(
        num_embeddings=K, 
        embedding_dim=C, 
        sparsity_level=4,
        normalize_atoms=False,
        patch_size=1
    )
    dl.dictionary.data.copy_(codebook.t().contiguous())  # DL uses [C, K]
    
    # Train for a few iterations to let representations evolve
    print(f"Training both models for 50 iterations...")
    optimizer_vq = torch.optim.Adam(vq.parameters(), lr=1e-3)
    optimizer_dl = torch.optim.Adam(dl.parameters(), lr=1e-3)
    
    vq.train()
    dl.train()
    
    for i in range(50):
        # Train VQ
        optimizer_vq.zero_grad()
        z_q_vq, loss_vq, perplexity, _ = vq(z)
        loss_vq.backward()
        optimizer_vq.step()
        
        # Train DL
        optimizer_dl.zero_grad()
        z_q_dl, loss_dl, _ = dl(z)
        loss_dl.backward()
        optimizer_dl.step()
        
        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/50: VQ loss={loss_vq.item():.4f}, DL loss={loss_dl.item():.4f}")
    
    # Extract learned representations
    with torch.no_grad():
        vq_codebook = vq.embedding.weight.data.cpu().numpy()  # [K, C]
        dl_atoms = dl.dictionary.data.t().cpu().numpy()  # [K, C]
    
    print(f"\nVQ Codebook shape: {vq_codebook.shape}")
    print(f"DL Dictionary shape: {dl_atoms.shape}")
    
    # Compute statistics
    print(f"\nVQ Codebook stats:")
    print(f"  Mean: {vq_codebook.mean(axis=0)}")
    print(f"  Std: {vq_codebook.std(axis=0)}")
    print(f"  Range: [{vq_codebook.min():.3f}, {vq_codebook.max():.3f}]")
    
    print(f"\nDL Dictionary stats:")
    print(f"  Mean: {dl_atoms.mean(axis=0)}")
    print(f"  Std: {dl_atoms.std(axis=0)}")
    print(f"  Range: [{dl_atoms.min():.3f}, {dl_atoms.max():.3f}]")
    
    # Normalize to [0, 1] for visualization (assuming data was in [-1, 1] range)
    vq_codebook_normalized = (vq_codebook + 1) / 2
    vq_codebook_normalized = np.clip(vq_codebook_normalized, 0, 1)
    
    dl_atoms_normalized = (dl_atoms + 1) / 2
    dl_atoms_normalized = np.clip(dl_atoms_normalized, 0, 1)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Codebook Visualization in RGB Space', fontsize=16, fontweight='bold')
    
    methods = [
        ('PCA', PCA(n_components=2)),
        ('t-SNE', TSNE(n_components=2, perplexity=min(30, K-1), random_state=42)),
        ('UMAP', umap.UMAP(n_components=2, random_state=42))
    ]
    
    for col_idx, (method_name, method) in enumerate(methods):
        print(f"\nComputing {method_name} embeddings...")
        
        # VQ codebook embedding
        vq_2d = method.fit_transform(vq_codebook)
        
        # DL dictionary embedding (fit new instance to avoid data leakage)
        if method_name == 'PCA':
            dl_method = PCA(n_components=2)
        elif method_name == 't-SNE':
            dl_method = TSNE(n_components=2, perplexity=min(30, K-1), random_state=42)
        else:
            dl_method = umap.UMAP(n_components=2, random_state=42)
        dl_2d = dl_method.fit_transform(dl_atoms)
        
        # Plot VQ (top row)
        ax_vq = axes[0, col_idx]
        scatter = ax_vq.scatter(
            vq_2d[:, 0], vq_2d[:, 1],
            c=vq_codebook_normalized,  # RGB colors normalized to [0,1]
            s=100,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        ax_vq.set_title(f'VQ Codebook - {method_name}', fontsize=12, fontweight='bold')
        ax_vq.set_xlabel(f'{method_name} Component 1')
        ax_vq.set_ylabel(f'{method_name} Component 2')
        ax_vq.grid(True, alpha=0.3)
        
        # Annotate a few points
        for i in range(min(10, K)):
            ax_vq.annotate(
                f'{i}',
                (vq_2d[i, 0], vq_2d[i, 1]),
                fontsize=8,
                alpha=0.6,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Plot DL (bottom row)
        ax_dl = axes[1, col_idx]
        scatter = ax_dl.scatter(
            dl_2d[:, 0], dl_2d[:, 1],
            c=dl_atoms_normalized,  # RGB colors normalized to [0,1]
            s=100,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        ax_dl.set_title(f'DL Dictionary - {method_name}', fontsize=12, fontweight='bold')
        ax_dl.set_xlabel(f'{method_name} Component 1')
        ax_dl.set_ylabel(f'{method_name} Component 2')
        ax_dl.grid(True, alpha=0.3)
        
        # Annotate a few points
        for i in range(min(10, K)):
            ax_dl.annotate(
                f'{i}',
                (dl_2d[i, 0], dl_2d[i, 1]),
                fontsize=8,
                alpha=0.6,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    plt.tight_layout()
    
    # Save visualization
    output_path = ARTIFACT_DIR / "codebook_embeddings.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close()
    
    # Additional analysis: Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform
    
    vq_dists = pdist(vq_codebook, metric='euclidean')
    dl_dists = pdist(dl_atoms, metric='euclidean')
    
    print(f"\nPairwise distance statistics:")
    print(f"  VQ codebook: mean={vq_dists.mean():.4f}, std={vq_dists.std():.4f}")
    print(f"  DL dictionary: mean={dl_dists.mean():.4f}, std={dl_dists.std():.4f}")
    
    # Create distance distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(vq_dists, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('VQ Codebook Pairwise Distances', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Euclidean Distance')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(vq_dists.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={vq_dists.mean():.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(dl_dists, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('DL Dictionary Pairwise Distances', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Euclidean Distance')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(dl_dists.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={dl_dists.mean():.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path_dist = ARTIFACT_DIR / "codebook_distances.png"
    plt.savefig(output_path_dist, dpi=150, bbox_inches='tight')
    print(f"✓ Saved distance distribution to: {output_path_dist}")
    plt.close()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    
    # The test passes if we successfully created the visualizations
    assert output_path.exists()
    assert output_path_dist.exists()


if __name__ == "__main__":
    test_visualize_codebook_embeddings()
