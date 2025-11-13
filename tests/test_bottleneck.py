import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

# Ensure project sources are importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.bottleneck import DictionaryLearning, VectorQuantizer  # noqa: E402
from src.data.celeba import CelebADataModule  # noqa: E402
from src.data.config import DataConfig  # noqa: E402


ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "bottleneck"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# --------- Vector-quantizer tests ---------


def test_vq_basic_shapes():
    """Test VQ output shapes and finite values."""
    torch.manual_seed(0)
    vq = VectorQuantizer(num_embeddings=32, embedding_dim=16, commitment_cost=0.25, decay=0.99)
    z = torch.randn(2, 16, 4, 4, requires_grad=True)
    z_q, loss, perplexity, encodings = vq(z)
    
    assert z_q.shape == z.shape
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert perplexity.ndim == 0 and torch.isfinite(perplexity)
    assert encodings.shape == (2 * 4 * 4, 32)


def test_vq_backward_and_ema():
    """Test VQ gradient flow and EMA updates."""
    torch.manual_seed(1)
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=8, commitment_cost=0.25, decay=0.99)
    z = torch.randn(1, 8, 2, 2, requires_grad=True)
    
    z_q, loss, _, _ = vq(z)
    (z_q.mean() + loss).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    
    # Test EMA update
    before = vq._ema_cluster_size.clone()
    vq.train()
    with torch.no_grad():
        vq(z.detach())
    assert not torch.equal(before, vq._ema_cluster_size)


def test_vq_codebook_selection():
    """Test VQ selects correct codebook entries."""
    torch.manual_seed(2)
    vq = VectorQuantizer(num_embeddings=3, embedding_dim=4, commitment_cost=0.25, decay=0.0)
    vq.eval()
    
    # Set codebook to identity matrix
    weight = torch.eye(4)[:3]
    vq.embedding.weight.data.copy_(weight)
    
    # Create input that exactly matches codebook entries
    idx_map = torch.tensor([[0, 1, 2], [2, 1, 0]])
    z = torch.zeros(1, 4, 2, 3)
    for i in range(2):
        for j in range(3):
            z[0, :, i, j] = weight[idx_map[i, j]]
    
    z_q, _, _, encodings = vq(z)
    assert torch.allclose(z_q, z, atol=1e-6)
    assert torch.equal(torch.argmax(encodings, dim=1).view(2, 3), idx_map)


# --------- CelebA helpers ---------


def _get_celeba_dir():
    """Find CelebA directory from common locations."""
    candidates = [
        os.environ.get("CELEBA_DIR", ""),
        Path.cwd().parents[0] / "Data" / "celeba" / "img_align_celeba",
        Path.home() / "Data" / "celeba" / "img_align_celeba",
    ]
    for c in candidates:
        c = Path(c).expanduser()
        if c.is_dir() and any(c.rglob("*.jpg")):
            return c
    return None


def _load_celeba_batch(batch_size: int = 4, image_size: int = 64):
    """Load a batch of CelebA images."""
    data_dir = _get_celeba_dir()
    if data_dir is None:
        raise RuntimeError("CelebA directory not found.")

    cfg = DataConfig(
        dataset="celeba",
        data_dir=str(data_dir),
        batch_size=batch_size,
        num_workers=0,
        image_size=image_size,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        augment=False,
    )
    dm = CelebADataModule(cfg)
    dm.prepare_data()
    dm.setup("fit")
    loader = dm.val_dataloader() or dm.train_dataloader()
    batch = next(iter(loader))
    return batch[0][:batch_size].detach()


def _kmeans_codebook(data: torch.Tensor, num_embeddings: int, iters: int = 6) -> torch.Tensor:
    """Simple k-means on image pixels to obtain representative colors."""
    flat = data.permute(0, 2, 3, 1).reshape(-1, data.shape[1]).contiguous()
    perm = torch.randperm(flat.size(0))
    centers = flat[perm[:num_embeddings]].clone()
    for _ in range(iters):
        distances = torch.cdist(flat, centers, p=2)
        assign = distances.argmin(dim=1)
        for k in range(num_embeddings):
            mask = assign == k
            if mask.any():
                centers[k] = flat[mask].mean(dim=0)
    return centers


def _denorm(img: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    """Denormalize image tensor."""
    mean_t = torch.tensor(mean, device=img.device, dtype=img.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, device=img.device, dtype=img.dtype).view(-1, 1, 1)
    return (img * std_t + mean_t).clamp(0.0, 1.0)


def _to_rgb(img: torch.Tensor, denormalize: bool = False) -> torch.Tensor:
    """Convert tensor to RGB numpy array for visualization."""
    if denormalize:
        img = _denorm(img)
    return img.permute(1, 2, 0).cpu().numpy().clip(0, 1)


# --------- Dictionary-learning tests ---------


def _build_dictionary_learning(num_embeddings=32, embedding_dim=16, sparsity_level=3, normalize_atoms=True):
    """Helper to create DictionaryLearning with standard params."""
    return DictionaryLearning(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        sparsity_level=sparsity_level,
        commitment_cost=0.25,
        decay=0.99,
        tolerance=1e-7,
        omp_debug=False,
        normalize_atoms=normalize_atoms,
    )


def test_dictionary_learning_shapes():
    """Test DL output shapes and finite loss."""
    torch.manual_seed(3)
    model = _build_dictionary_learning(num_embeddings=32, embedding_dim=16, sparsity_level=3)
    z = torch.randn(2, 16, 4, 4, requires_grad=True)
    z_dl, loss, coeffs = model(z)
    
    assert z_dl.shape == z.shape
    assert coeffs.shape == (32, 2 * 4 * 4)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_dictionary_learning_backward():
    """Test DL gradient flow."""
    torch.manual_seed(4)
    model = _build_dictionary_learning(num_embeddings=32, embedding_dim=16, sparsity_level=3)
    z = torch.randn(1, 16, 4, 4, requires_grad=True)
    z_dl, loss, _ = model(z)
    
    (torch.nn.functional.mse_loss(z_dl, z) + loss).backward()
    assert model.dictionary.grad is not None and torch.isfinite(model.dictionary.grad).all()
    assert z.grad is not None and torch.isfinite(z.grad).all()


# --------- Visualization helpers ---------


def _plot_usage(usage, sorted_idx, title, filename):
    """Plot usage histogram."""
    plt.figure(figsize=(8, 3))
    plt.bar(range(len(usage)), usage[sorted_idx].cpu().numpy())
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / filename)
    plt.close()


def _plot_atoms(dictionary, sorted_idx, is_celeba, num_atoms=9):
    """Plot dictionary atoms as color patches."""
    C, K = dictionary.shape
    atoms_to_show = sorted_idx[:num_atoms].tolist()
    
    plt.figure(figsize=(12, 8))
    for i, atom_idx in enumerate(atoms_to_show):
        plt.subplot(3, 3, i + 1)
        atom_vec = dictionary[:, atom_idx].detach().cpu().numpy()
        
        if C == 3:  # RGB color data
            # Denormalize if CelebA data (from [-1,1] to [0,1])
            if is_celeba:
                atom_rgb = (atom_vec * 0.5 + 0.5).clip(0, 1)
            else:
                atom_rgb = atom_vec.clip(0, 1)
            
            # Create larger color patch for better visibility
            color_patch = np.tile(atom_rgb.reshape(1, 1, 3), (50, 50, 1))
            plt.imshow(color_patch)
            plt.title(f"Atom {atom_idx}\nRGB: [{atom_vec[0]:.3f}, {atom_vec[1]:.3f}, {atom_vec[2]:.3f}]", fontsize=9)
        else:
            plt.plot(atom_vec)
            plt.title(f"Atom {atom_idx}")
        plt.axis('off')
    
    plt.suptitle("Dictionary Learning Atoms (Most Used)", fontsize=14)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "dictionary_atoms.png", dpi=100)
    plt.close()


def _plot_codebook(codebook, sorted_idx, is_celeba, num_codes=9):
    """Plot VQ codebook vectors as color patches."""
    K, C = codebook.shape
    codes_to_show = sorted_idx[:num_codes].tolist()
    
    plt.figure(figsize=(12, 8))
    for i, code_idx in enumerate(codes_to_show):
        plt.subplot(3, 3, i + 1)
        code_vec = codebook[code_idx].detach().cpu().numpy()
        
        if C == 3:  # RGB color data
            # Denormalize if CelebA data (from [-1,1] to [0,1])
            if is_celeba:
                code_rgb = (code_vec * 0.5 + 0.5).clip(0, 1)
            else:
                code_rgb = code_vec.clip(0, 1)
            
            # Create larger color patch for better visibility
            color_patch = np.tile(code_rgb.reshape(1, 1, 3), (50, 50, 1))
            plt.imshow(color_patch)
            plt.title(f"Code {code_idx}\nRGB: [{code_vec[0]:.3f}, {code_vec[1]:.3f}, {code_vec[2]:.3f}]", fontsize=9)
        else:
            plt.plot(code_vec)
            plt.title(f"Code {code_idx}")
        plt.axis('off')
    
    plt.suptitle("VQ Codebook Vectors (Most Used)", fontsize=14)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "vq_codebook_vectors.png", dpi=100)
    plt.close()


def _plot_reconstructions(z, z_q_vq, z_q_dl, is_celeba):
    """Plot reconstruction comparison."""
    B, C, H, W = z.shape
    diff_vq = torch.mean((z - z_q_vq) ** 2, dim=1)
    diff_dl = torch.mean((z - z_q_dl) ** 2, dim=1)
    
    rows = min(4, B)
    plt.figure(figsize=(21, 4 * rows))
    
    for i in range(rows):
        row_offset = i * 5
        
        # Original
        plt.subplot(rows, 5, row_offset + 1)
        plt.imshow(_to_rgb(z[i], denormalize=is_celeba))
        plt.title(f"Original #{i+1}", fontsize=10)
        plt.axis("off")
        
        # VQ reconstruction
        plt.subplot(rows, 5, row_offset + 2)
        plt.imshow(_to_rgb(z_q_vq[i], denormalize=is_celeba))
        vq_mse = torch.mean((z[i] - z_q_vq[i]) ** 2).item()
        plt.title(f"VQ\nMSE={vq_mse:.4f}", fontsize=10)
        plt.axis("off")
        
        # VQ error
        plt.subplot(rows, 5, row_offset + 3)
        im = plt.imshow(diff_vq[i].cpu().numpy(), cmap="magma")
        plt.title("VQ Error", fontsize=10)
        plt.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # DL reconstruction
        plt.subplot(rows, 5, row_offset + 4)
        plt.imshow(_to_rgb(z_q_dl[i], denormalize=is_celeba))
        dl_mse = torch.mean((z[i] - z_q_dl[i]) ** 2).item()
        plt.title(f"DL\nMSE={dl_mse:.4f}", fontsize=10)
        plt.axis("off")
        
        # DL error
        plt.subplot(rows, 5, row_offset + 5)
        im = plt.imshow(diff_dl[i].cpu().numpy(), cmap="magma")
        plt.title("DL Error", fontsize=10)
        plt.axis("off")
        plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.suptitle("VQ vs Dictionary Learning", fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "reconstruction_comparison.png", dpi=100)
    plt.close()


def _plot_channel_comparison(z, z_q_vq, z_q_dl):
    """Plot channel-wise comparison for RGB images."""
    B, C, H, W = z.shape
    if C != 3:
        return
    
    channel_names = ['Red', 'Green', 'Blue']
    center_row = H // 2
    
    plt.figure(figsize=(15, 5))
    for ch in range(3):
        plt.subplot(1, 3, ch + 1)
        
        x = range(W)
        plt.plot(x, z[0, ch, center_row].cpu().numpy(), 'k-', label='Original', linewidth=2, alpha=0.7)
        plt.plot(x, z_q_vq[0, ch, center_row].cpu().numpy(), 'b--', label='VQ', linewidth=1.5)
        plt.plot(x, z_q_dl[0, ch, center_row].cpu().numpy(), 'r:', label='DL', linewidth=1.5)
        
        plt.title(f'{channel_names[ch]} Channel (row {center_row})', fontsize=12)
        plt.xlabel('Column')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Channel-wise Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "vq_channel_comparison.png", dpi=100)
    plt.close()


def _plot_code_heatmaps(z, enc_vq, coeffs, is_celeba):
    """Plot VQ indices and DL sparse codes as heatmaps for interpretability."""
    B, C, H, W = z.shape
    
    # Get VQ indices: enc_vq is (B*H*W, K), find argmax for each position
    vq_indices = torch.argmax(enc_vq, dim=1).reshape(B, H, W)
    
    # Get DL sparsity pattern: coeffs is (K, B*H*W)
    # For each position, count how many atoms are active
    dl_sparsity = (coeffs.abs() > 1e-6).sum(dim=0).reshape(B, H, W).float()
    
    # Get DL coefficient magnitudes
    dl_magnitude_raw = coeffs.abs().sum(dim=0).reshape(B, H, W)
    # Use max coefficient per pixel for clearer interpretation
    dl_max_coeff = coeffs.abs().max(dim=0)[0].reshape(B, H, W)
    
    num_images = min(2, B)  # Show first 2 images
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Original image
        axes[i, 0].imshow(_to_rgb(z[i], denormalize=is_celeba))
        axes[i, 0].set_title(f'Original #{i+1}', fontsize=11)
        axes[i, 0].axis('off')
        
        # VQ indices heatmap
        im1 = axes[i, 1].imshow(vq_indices[i].cpu().numpy(), cmap='viridis', interpolation='nearest')
        axes[i, 1].set_title(f'VQ Code Indices\n({vq_indices[i].unique().numel()} unique)', fontsize=11)
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # DL max coefficient magnitude
        mag_data = dl_max_coeff[i].cpu().numpy()
        # Use log1p for better handling of small values
        mag_log = np.log1p(mag_data)
        
        # Use percentile-based normalization for better contrast
        vmin, vmax = np.percentile(mag_log, [2, 98])
        im2 = axes[i, 2].imshow(mag_log, cmap='viridis', interpolation='nearest', 
                                 vmin=vmin, vmax=vmax)
        axes[i, 2].set_title(f'DL Max Coefficient\n(log scale)', fontsize=11)
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.suptitle('Code Interpretability: VQ Indices vs DL Sparse Codes', fontsize=14)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "code_heatmaps.png", dpi=100)
    plt.close()


def test_bottleneck_visualizations():
    """Test bottleneck visualizations on CelebA or synthetic data."""
    torch.manual_seed(6)
    
    # Load data
    celeba_dir = _get_celeba_dir()
    if celeba_dir:
        z = _load_celeba_batch(batch_size=4, image_size=128)
    else:
        z = torch.randn(4, 3, 128, 128)
    
    B, C, H, W = z.shape
    K = 16  # Use 16 atoms/codebook vectors for maximum interpretability
    
    # Initialize models with k-means codebook
    codebook = _kmeans_codebook(z, K)
    vq = VectorQuantizer(num_embeddings=K, embedding_dim=C, commitment_cost=0.25, decay=0.0)
    vq.embedding.weight.data.copy_(codebook)
    
    # DL at pixel level: disable normalization and use 4 atoms per pixel
    # Sparsity=4 balances quality and interpretability
    dl = _build_dictionary_learning(num_embeddings=K, embedding_dim=C, sparsity_level=4, normalize_atoms=False)
    dl.dictionary.data.copy_(codebook.t().contiguous())
    
    # Forward pass
    with torch.no_grad():
        vq.eval()
        dl.eval()
        z_q_vq, _, _, enc_vq = vq(z)
        z_q_dl, _, coeffs = dl(z)
    
    is_celeba = celeba_dir is not None
    
    # Plot usage
    usage_vq = enc_vq.sum(dim=0)
    _, sorted_idx_vq = torch.sort(usage_vq, descending=True)
    _plot_usage(usage_vq, sorted_idx_vq, "VQ Codebook Usage", "vq_codebook_usage.png")
    
    usage_dl = (coeffs.abs() > 1e-6).sum(dim=1).float()
    _, sorted_idx_dl = torch.sort(usage_dl, descending=True)
    _plot_usage(usage_dl, sorted_idx_dl, "Dictionary Atom Usage", "dictionary_atom_usage.png")
    
    torch.save(sorted_idx_dl.cpu(), ARTIFACT_DIR / "dictionary_atom_usage_order.pt")
    
    # Plot reconstructions
    _plot_reconstructions(z, z_q_vq, z_q_dl, is_celeba)
    
    # Plot channel comparison if RGB
    if C == 3:
        _plot_channel_comparison(z, z_q_vq, z_q_dl)
    
    # Plot code heatmaps for interpretability
    _plot_code_heatmaps(z, enc_vq, coeffs, is_celeba)
    
    # Verify all files created
    expected_files = [
        "vq_codebook_usage.png",
        "dictionary_atom_usage.png",
        "dictionary_atom_usage_order.pt",
        "reconstruction_comparison.png",
        "code_heatmaps.png",
    ]
    if C == 3:
        expected_files.append("vq_channel_comparison.png")
    
    for name in expected_files:
        assert (ARTIFACT_DIR / name).exists()

