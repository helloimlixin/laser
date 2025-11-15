import os
import sys
from pathlib import Path

import math
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torchvision

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


def _load_celeba_batch(batch_size: int = 4, image_size: int = 128):
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


def test_dictionary_learning_patch_tokens():
    """Patch-wise sparse codes reduce number of tokens."""
    torch.manual_seed(5)
    model = DictionaryLearning(
        num_embeddings=32,
        embedding_dim=8,
        sparsity_level=3,
        patch_size=2,
    )
    z = torch.randn(2, 8, 8, 8, requires_grad=True)
    z_dl, loss, coeffs = model(z)

    assert z_dl.shape == z.shape
    patches_per_sample = (8 // 2) * (8 // 2)
    assert coeffs.shape == (32, 2 * patches_per_sample)
    assert torch.isfinite(loss)


def test_patch_dictionary_reconstructs_tiled_patterns():
    """Patch-based encoding should perfectly reconstruct tiled atoms."""
    torch.manual_seed(7)
    patch_size = 8
    model = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=1,
        sparsity_level=1,
        patch_size=patch_size,
        normalize_atoms=False,
    )
    atom_dim = model.atom_dim
    with torch.no_grad():
        ones_atom = torch.ones(atom_dim)
        zeros_atom = torch.zeros(atom_dim)
        model.dictionary.copy_(torch.stack([ones_atom, zeros_atom], dim=1))

    # Create latent grid with quadrants of ones/zeros aligned with patch tiling
    latent_size = patch_size * 2
    z = torch.zeros(1, 1, latent_size, latent_size)
    z[:, :, :patch_size, :patch_size] = 1.0
    z[:, :, patch_size:, patch_size:] = 1.0

    z_dl, loss, coeffs = model(z)
    assert torch.allclose(z_dl, z, atol=1e-6)
    assert torch.isfinite(loss)
    patches_per_sample = (z.shape[2] // patch_size) * (z.shape[3] // patch_size)
    assert coeffs.shape == (2, patches_per_sample)
    # Each patch should select at most sparsity_level atoms
    active_counts = (coeffs.abs() > 1e-6).sum(dim=0)
    assert torch.all(active_counts <= model.sparsity_level)


def test_patch_dictionary_visualization_artifact():
    """Visualize CelebA latent RGB projection patches and corresponding pixel regions."""
    celeba_dir = _get_celeba_dir()
    if celeba_dir is None:
        pytest.skip("CelebA images not available for patch visualization.")

    torch.manual_seed(0)
    patch_size = 4  # latent cells
    pixel_image_size = 64
    latent_stride = 4  # each latent cell corresponds to 4x4 pixels (avg pool)

    batch = _load_celeba_batch(batch_size=1, image_size=pixel_image_size)
    # Denormalize to [0,1]
    pixel_rgb_tensor = _denorm(batch[0]).unsqueeze(0)

    # Simple latent representation: avg pool to emulate encoder downsampling (H/4, W/4)
    latent_map = torch.nn.functional.avg_pool2d(
        pixel_rgb_tensor, kernel_size=latent_stride, stride=latent_stride
    )
    latent_map = latent_map.clamp(0.0, 1.0)

    latent_h, latent_w = latent_map.shape[2], latent_map.shape[3]
    patch_rows = latent_h // patch_size
    patch_cols = latent_w // patch_size
    num_patches = patch_rows * patch_cols

    patches_flat = torch.nn.functional.unfold(
        latent_map, kernel_size=patch_size, stride=patch_size
    )
    patch_vectors = patches_flat.squeeze(0)
    coeff_map = torch.arange(num_patches, dtype=torch.long).reshape(patch_rows, patch_cols)

    def _to_disp(arr):
        return np.clip(arr, 0.0, 1.0) ** (1 / 2.2)

    latent_rgb = latent_map[0].permute(1, 2, 0).cpu().numpy()
    latent_rgb_disp = _to_disp(latent_rgb)
    latent_tokens = patch_vectors.t().contiguous().reshape(num_patches, 3, patch_size, patch_size)
    latent_token_grid = torchvision.utils.make_grid(
        latent_tokens, nrow=patch_cols, padding=2
    ).permute(1, 2, 0).cpu().numpy()
    latent_token_grid_disp = _to_disp(latent_token_grid)

    pixel_rgb = pixel_rgb_tensor[0].permute(1, 2, 0).cpu().numpy()
    pixel_rgb_disp = _to_disp(pixel_rgb)
    pixel_patch_px = patch_size * latent_stride
    pixel_patches_flat = torch.nn.functional.unfold(
        pixel_rgb_tensor, kernel_size=pixel_patch_px, stride=pixel_patch_px
    )
    pixel_patches = pixel_patches_flat.permute(0, 2, 1).reshape(
        num_patches, 3, pixel_patch_px, pixel_patch_px
    )
    pixel_token_grid = torchvision.utils.make_grid(
        pixel_patches, nrow=patch_cols, padding=4
    ).permute(1, 2, 0).cpu().numpy()
    pixel_token_grid_disp = _to_disp(pixel_token_grid)

    def _plot_with_grid(ax, image, patch_px, title, labels=None):
        ax.imshow(image, interpolation="nearest")
        h, w = image.shape[:2]
        ax.set_xticks(np.arange(0, w + 1, patch_px))
        ax.set_yticks(np.arange(0, h + 1, patch_px))
        ax.grid(color="white", linewidth=1.0, alpha=0.7)
        if labels is not None:
            for r in range(labels.shape[0]):
                for c in range(labels.shape[1]):
                    ax.text(
                        c * patch_px + patch_px / 2,
                        r * patch_px + patch_px / 2,
                        f"{int(labels[r, c])}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=10,
                        weight="bold",
                    )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(title)

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    _plot_with_grid(
        axes[0, 0],
        latent_rgb_disp,
        patch_size,
        "Latent RGB projection\n(avg pooled) with patch indices",
        labels=coeff_map,
    )
    axes[0, 1].imshow(latent_token_grid_disp, interpolation="nearest")
    axes[0, 1].set_title("Latent patch tokens (normalized for visibility)")
    axes[0, 1].axis("off")

    _plot_with_grid(
        axes[1, 0],
        pixel_rgb_disp,
        pixel_patch_px,
        "Original CelebA crop with corresponding patches",
        labels=coeff_map,
    )
    axes[1, 1].imshow(pixel_token_grid_disp, interpolation="nearest")
    axes[1, 1].set_title("Pixel-space patches for each latent token")
    axes[1, 1].axis("off")

    artifact_path = ARTIFACT_DIR / "patch_latent_reconstruction.png"
    fig.tight_layout()
    fig.savefig(artifact_path, dpi=160)
    plt.close(fig)
    assert artifact_path.exists()


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
    num_atoms = coeffs.shape[0]  # K = number of dictionary atoms
    # Columns are ordered as (patch_idx, batch_idx) because batching happens inside the unfolding.
    # Reshape to [num_atoms, num_patches, batch] first, then permute so batch dimension comes before tokens.
    num_patches_total = coeffs.shape[1] // B
    coeffs_view = coeffs.view(coeffs.shape[0], num_patches_total, B).permute(0, 2, 1).contiguous()
    num_tokens = max(1, coeffs_view.shape[2])

    def _infer_patch_dims(height, width, tokens):
        """Infer patch height/width from the number of sparse-code tokens."""
        total_pixels = height * width
        if total_pixels % tokens != 0:
            return 1, 1  # fallback to per-pixel visualization
        patch_area = total_pixels // tokens
        best = (1, patch_area)
        target = math.sqrt(patch_area)
        for candidate in range(1, patch_area + 1):
            if patch_area % candidate != 0:
                continue
            patch_h = candidate
            patch_w = patch_area // candidate
            if height % patch_h == 0 and width % patch_w == 0:
                best = min(best, (patch_h, patch_w), key=lambda hw: abs(hw[0] - target))
                if patch_h == patch_w:
                    return patch_h, patch_w
        patch_h, patch_w = best
        if height % patch_h != 0 or width % patch_w != 0:
            return 1, 1
        return patch_h, patch_w

    patch_h, patch_w = _infer_patch_dims(H, W, num_tokens)
    patch_rows = max(1, H // patch_h)
    patch_cols = max(1, W // patch_w)

    def _tokens_to_grid(sample_tokens):
        """Reshape [B, num_tokens] tensor into [B, patch_rows, patch_cols]."""
        if sample_tokens.numel() != B * patch_rows * patch_cols:
            return sample_tokens.reshape(B, H, W)
        return sample_tokens.view(B, patch_rows, patch_cols)

    def _expand_patches(patch_tensor):
        """Nearest-neighbour upsample patch tokens back to pixel resolution for visualization."""
        upsampled = patch_tensor.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)
        return upsampled[:, :H, :W]
    
    # Get VQ indices: enc_vq is (B*H*W, K), find argmax for each position
    vq_indices = torch.argmax(enc_vq, dim=1).reshape(B, H, W)
    
    # Create pixel-level coefficient map for DL
    # Use L1 norm (sum of absolute coefficients) as it's more stable and interpretable
    # This represents the total "activation strength" at each patch
    
    # coeffs_view shape: [num_atoms, batch, num_patches]
    # Compute L1 norm per patch (sum of absolute coefficients across all atoms)
    l1_norm_per_patch = coeffs_view.abs().sum(dim=0)  # [batch, num_patches]
    
    if patch_h == 1 and patch_w == 1:
        # Pixel-level DL: direct mapping
        dl_activation = l1_norm_per_patch.reshape(B, H, W)
    else:
        # Patch-level DL: use fold operation to map to pixels
        coeff_for_fold = l1_norm_per_patch.unsqueeze(1)  # [batch, 1, num_patches]
        
        dl_activation = F.fold(
            coeff_for_fold,
            output_size=(H, W),
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w)
        ).squeeze(1)  # [batch, H, W]
    
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
        
        # DL: Show L1 norm of sparse codes (simpler and cleaner)
        activation_data = dl_activation[i].cpu().numpy()
        
        # Simple percentile normalization for clean visualization
        p_low, p_high = np.percentile(activation_data, [1, 99])
        activation_norm = np.clip(activation_data, p_low, p_high)
        activation_norm = (activation_norm - p_low) / (p_high - p_low + 1e-10)
        
        im2 = axes[i, 2].imshow(activation_norm, cmap='viridis', interpolation='nearest')
        axes[i, 2].set_title(f'DL Sparse Code Strength\n(L1 norm)', fontsize=11)
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
    
    # Benchmark inference speed (warmup + timed runs)
    with torch.no_grad():
        vq.eval()
        dl.eval()
        
        # Warmup
        for _ in range(3):
            _ = vq(z)
            _ = dl(z)
        
        # Benchmark VQ
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        vq_times = []
        for _ in range(10):
            start = time.perf_counter()
            z_q_vq, _, _, enc_vq = vq(z)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            vq_times.append(time.perf_counter() - start)
        vq_time_ms = np.mean(vq_times) * 1000
        
        # Benchmark DL
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dl_times = []
        for _ in range(10):
            start = time.perf_counter()
            z_q_dl, _, coeffs = dl(z)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dl_times.append(time.perf_counter() - start)
        dl_time_ms = np.mean(dl_times) * 1000
    
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
    
    # Compute and print comprehensive metrics
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE METRICS")
    print("="*60)
    
    # Overall reconstruction quality
    mse_vq = F.mse_loss(z_q_vq, z).item()
    mse_dl = F.mse_loss(z_q_dl, z).item()
    color_dist_vq = torch.sqrt(F.mse_loss(z_q_vq, z, reduction='none').mean(dim=1)).mean().item()
    color_dist_dl = torch.sqrt(F.mse_loss(z_q_dl, z, reduction='none').mean(dim=1)).mean().item()
    
    print(f"\n📊 Overall Reconstruction Quality:")
    print(f"   VQ  - MSE: {mse_vq:.5f}, Per-Pixel Color Distance: {color_dist_vq:.4f}")
    print(f"   DL  - MSE: {mse_dl:.5f}, Per-Pixel Color Distance: {color_dist_dl:.4f}")
    print(f"   DL Improvement: {mse_vq/mse_dl:.1f}× better MSE, {color_dist_dl/color_dist_vq:.1%} of VQ color error")
    
    # Channel-wise metrics
    if C == 3:
        print(f"\n🎨 Channel-wise Performance:")
        channel_names = ['Red', 'Green', 'Blue']
        for ch in range(3):
            ch_mse_vq = F.mse_loss(z_q_vq[:, ch], z[:, ch]).item()
            ch_mse_dl = F.mse_loss(z_q_dl[:, ch], z[:, ch]).item()
            print(f"   {channel_names[ch]:5} - VQ: {ch_mse_vq:.5f}, DL: {ch_mse_dl:.5f} ({ch_mse_vq/ch_mse_dl:.1f}× improvement)")
    
    # Codebook/atom utilization
    print(f"\n📚 Codebook/Atom Utilization:")
    vq_used = (usage_vq > 0).sum().item()
    dl_used = (usage_dl > 0).sum().item()
    print(f"   VQ Codes Used: {vq_used}/{K} ({100*vq_used/K:.0f}%)")
    print(f"   DL Atoms Used: {dl_used}/{K} ({100*dl_used/K:.0f}%)")
    print(f"   Average atoms per pixel: {(coeffs.abs() > 1e-6).sum(dim=0).float().mean().item():.2f}")
    
    # Inference speed
    num_pixels = B * H * W
    print(f"\n⚡ Inference Speed (batch={B}, resolution={H}×{W}, {num_pixels:,} pixels):")
    print(f"   VQ  - {vq_time_ms:.2f} ms total, {vq_time_ms*1000/num_pixels:.3f} µs/pixel")
    print(f"   DL  - {dl_time_ms:.2f} ms total, {dl_time_ms*1000/num_pixels:.3f} µs/pixel")
    print(f"   DL Slowdown: {dl_time_ms/vq_time_ms:.1f}× (greedy OMP overhead)")
    
    # Complexity analysis
    print(f"\n🔢 Computational Complexity:")
    print(f"   VQ  - O(K × N) = O({K} × {num_pixels:,}) distance computations")
    print(f"   DL  - O(K × S × N) = O({K} × 4 × {num_pixels:,}) for {4} OMP iterations per pixel")
    print(f"   Theoretical Slowdown: ~{4}× (measured: {dl_time_ms/vq_time_ms:.1f}×)")
    
    print("\n" + "="*60)
    
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
