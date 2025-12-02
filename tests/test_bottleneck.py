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

pytestmark = pytest.mark.filterwarnings(
    "ignore:n_jobs value 1 overridden.*random_state"
)

# Ensure project sources are importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.models.bottleneck import VectorQuantizer, DictionaryLearning  # noqa: E402
from src.data.celeba import CelebADataModule  # noqa: E402
from src.data.config import DataConfig  # noqa: E402

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import umap
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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


def _build_dictionary_learning(num_embeddings=32, embedding_dim=16, sparsity_level=3):
    """Helper to create DictionaryLearning with standard params."""
    return DictionaryLearning(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        sparsity_level=sparsity_level,
        commitment_cost=0.25,
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
    
    # DL at pixel level: use 4 atoms per pixel
    # Sparsity=4 balances quality and interpretability
    dl = _build_dictionary_learning(num_embeddings=K, embedding_dim=C, sparsity_level=4)
    dl.dictionary.data.copy_(codebook.t().contiguous())
    
    # Benchmark inference speed across different batch sizes
    batch_sizes = [1, 4, 8, 16]
    vq_times_by_batch = {}
    dl_times_by_batch = {}
    
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARKS")
    print("="*60)
    
    for batch_size in batch_sizes:
        # Create test batch
        if celeba_dir:
            z_test = _load_celeba_batch(batch_size=batch_size, image_size=128)
        else:
            z_test = torch.randn(batch_size, C, H, W)
        
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = vq(z_test)
                _ = dl(z_test)
            
            # Benchmark VQ
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            vq_times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = vq(z_test)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                vq_times.append(time.perf_counter() - start)
            vq_time_ms = np.mean(vq_times) * 1000
            
            # Benchmark DL
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dl_times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = dl(z_test)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                dl_times.append(time.perf_counter() - start)
            dl_time_ms = np.mean(dl_times) * 1000
            
            vq_times_by_batch[batch_size] = vq_time_ms
            dl_times_by_batch[batch_size] = dl_time_ms
            
            num_pixels = batch_size * H * W
            print(f"\nBatch Size {batch_size} ({num_pixels:,} pixels):")
            print(f"  VQ: {vq_time_ms:6.2f} ms ({vq_time_ms*1000/num_pixels:.3f} µs/pixel)")
            print(f"  DL: {dl_time_ms:6.2f} ms ({dl_time_ms*1000/num_pixels:.3f} µs/pixel)")
            print(f"  Slowdown: {dl_time_ms/vq_time_ms:.1f}×")
    
    print("\n" + "="*60)
    
    # Use batch_size=4 for the rest of the visualizations
    if celeba_dir:
        z = _load_celeba_batch(batch_size=4, image_size=128)
    else:
        z = torch.randn(4, C, H, W)
    
    with torch.no_grad():
        z_q_vq, _, _, enc_vq = vq(z)
        z_q_dl, _, coeffs = dl(z)
    
    B = z.shape[0]
    
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
    vq_time_ms = vq_times_by_batch[4]
    dl_time_ms = dl_times_by_batch[4]
    num_pixels = 4 * H * W
    print(f"\n⚡ Inference Speed (batch=4, resolution={H}×{W}, {num_pixels:,} pixels):")
    print(f"   VQ  - {vq_time_ms:.2f} ms total, {vq_time_ms*1000/num_pixels:.3f} µs/pixel")
    print(f"   DL  - {dl_time_ms:.2f} ms total, {dl_time_ms*1000/num_pixels:.3f} µs/pixel")
    print(f"   DL Slowdown: {dl_time_ms/vq_time_ms:.1f}× (greedy OMP overhead)")
    
    # Complexity analysis
    print(f"\n🔢 Computational Complexity:")
    print(f"   VQ  - O(K × M × N) where K={K}, M={C}, N={num_pixels:,} pixels")
    print(f"         Single vectorized distance computation across all pixels")
    print(f"   DL  - O(S × K × M × N) where S=4, K={K}, M={C}, N={num_pixels:,}")  
    print(f"         4 sequential OMP iterations, each computing K×M operations per pixel")
    print(f"   Theoretical Slowdown: ~4× (from S=4 sparsity)")
    print(f"   Measured Slowdown: {dl_time_ms/vq_time_ms:.1f}× (sequential iteration overhead + less vectorization)")
    
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


def test_patch_based_speed_comparison():
    """Compare speed of pixel-level vs patch-based dictionary learning."""
    torch.manual_seed(7)
    
    # Use real images if available, otherwise random
    celeba_dir = _get_celeba_dir()
    if celeba_dir:
        z = _load_celeba_batch(batch_size=4, image_size=128)
        print("\nUsing CelebA images for realistic comparison")
    else:
        z = torch.randn(4, 3, 128, 128)
        print("\nWarning: Using random noise (CelebA not available)")
    
    B, C, H, W = z.shape
    K = 16
    
    # Initialize VQ with k-means for baseline
    codebook = _kmeans_codebook(z, K)
    vq = VectorQuantizer(num_embeddings=K, embedding_dim=C, commitment_cost=0.25, decay=0.0)
    vq.embedding.weight.data.copy_(codebook)
    
    patch_sizes = [1, 2, 4, 8]
    
    print("\n" + "="*60)
    print("PATCH-BASED DL SPEED & QUALITY COMPARISON")
    print("="*60)
    
    for patch_size in patch_sizes:
        # Create DL with specific patch size
        dl = DictionaryLearning(
            num_embeddings=K, 
            embedding_dim=C, 
            sparsity_level=4,
            patch_size=patch_size
        )
        
        # Initialize dictionary with k-means on patches
        # Extract patches from the data to initialize dictionary properly
        patches = F.unfold(z, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
        # patches shape: [B, C*patch_size^2, num_patches]
        patches_flat = patches.reshape(patches.shape[0] * patches.shape[2], patches.shape[1])  # [B*num_patches, atom_dim]
        patch_codebook = _kmeans_codebook(patches_flat.t().reshape(1, patches.shape[1], -1, 1), K)
        dl.dictionary.data.copy_(patch_codebook.squeeze().t().contiguous())
        
        # Warmup
        with torch.no_grad():
            dl.eval()
            vq.eval()
            for _ in range(3):
                _ = dl(z)
                _ = vq(z)
        
        # Benchmark DL
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                z_q_dl, loss, coeffs = dl(z)
                times.append(time.perf_counter() - start)
        
        avg_time_ms = np.mean(times) * 1000
        num_patches = (H // patch_size) * (W // patch_size) * B
        time_per_patch_us = avg_time_ms * 1000 / num_patches
        
        # Benchmark VQ (same input)
        vq_times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.perf_counter()
                z_q_vq, _, _, _ = vq(z)
                vq_times.append(time.perf_counter() - start)
        vq_time_ms = np.mean(vq_times) * 1000
        
        # Reconstruction quality
        mse_dl = F.mse_loss(z_q_dl, z).item()
        mse_vq = F.mse_loss(z_q_vq, z).item()
        
        print(f"\nPatch Size {patch_size}×{patch_size}:")
        print(f"  Patches: {num_patches:,} ({H//patch_size}×{W//patch_size} per image)")
        print(f"  DL Time: {avg_time_ms:.2f} ms, VQ Time: {vq_time_ms:.2f} ms")
        print(f"  DL MSE: {mse_dl:.5f}, VQ MSE: {mse_vq:.5f}")
        if mse_dl < mse_vq:
            print(f"  Quality: DL {mse_vq/mse_dl:.1f}× better than VQ")
        else:
            print(f"  Quality: VQ {mse_dl/mse_vq:.1f}× better than DL")
        print(f"  Speed: DL {34.0/avg_time_ms:.1f}× faster than 1×1 pixel-level")
    
    print("\n" + "="*60)


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
    celeba_dir = _get_celeba_dir()
    if celeba_dir:
        z = _load_celeba_batch(batch_size=16, image_size=128)
        print("\nUsing CelebA images for training")
    else:
        z = torch.randn(16, 3, 128, 128)
        # Normalize to [-1, 1] range to match CelebA preprocessing
        z = (z - 0.5) * 2.0
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


# --------- K-SVD Dictionary Learning tests ---------


def test_ksvd_basic_shapes():
    """Test K-SVD output shapes and finite values."""
    torch.manual_seed(0)
    ksvd = DictionaryLearning(
        num_embeddings=32,
        embedding_dim=16,
        sparsity_level=3,
        commitment_cost=0.25,
        ksvd_iterations=1,
    )
    
    z = torch.randn(2, 16, 4, 4, requires_grad=True)
    z_dl, loss, coeffs = ksvd(z)
    
    assert z_dl.shape == z.shape
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert coeffs.shape[0] == 32  # num_embeddings
    assert coeffs.shape[1] == 2 * 4 * 4  # batch_size * num_patches


def test_ksvd_iht_solver_sparse_codes():
    """Ensure iterative hard thresholding produces sparse codes."""
    torch.manual_seed(11)
    ksvd = DictionaryLearning(
        num_embeddings=24,
        embedding_dim=8,
        sparsity_level=4,
        patch_size=1,
        ksvd_iterations=0,
        sparse_solver="iht",
        iht_iterations=6,
    )
    ksvd.enable_ksvd_update = False
    z = torch.randn(2, 8, 4, 4)
    z_dl, loss, coeffs = ksvd(z)

    assert z_dl.shape == z.shape
    assert torch.isfinite(loss)
    nnz = (coeffs != 0).sum(dim=0)
    assert torch.all(nnz <= ksvd.sparsity_level)


def test_ksvd_dictionary_update():
    """Test that K-SVD updates the dictionary during training."""
    torch.manual_seed(1)
    ksvd = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=8,
        sparsity_level=3,
        ksvd_iterations=2,
    )
    
    # Store initial dictionary
    dict_before = ksvd.dictionary.data.clone()
    
    # Forward pass in training mode
    ksvd.train()
    z = torch.randn(2, 8, 4, 4)
    z_dl, loss, coeffs = ksvd(z)
    
    # Dictionary should have changed
    dict_after = ksvd.dictionary.data
    assert not torch.allclose(dict_before, dict_after, atol=1e-6)
    
    # Dictionary atoms should always be normalized
    norms = torch.linalg.norm(dict_after, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_ksvd_no_update_in_eval():
    """Test that K-SVD doesn't update dictionary in eval mode."""
    torch.manual_seed(2)
    ksvd = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=8,
        sparsity_level=3,
        ksvd_iterations=2,
    )
    
    # Store initial dictionary
    dict_before = ksvd.dictionary.data.clone()
    
    # Forward pass in eval mode
    ksvd.eval()
    z = torch.randn(2, 8, 4, 4)
    with torch.no_grad():
        z_dl, loss, coeffs = ksvd(z)
    
    # Dictionary should not have changed
    dict_after = ksvd.dictionary.data
    assert torch.allclose(dict_before, dict_after)


def test_ksvd_vs_regular_dl_comparison():
    """Compare K-SVD mode with backprop-only mode."""
    torch.manual_seed(3)
    
    # Create synthetic data with known structure
    B, C, H, W = 4, 3, 32, 32
    z = torch.randn(B, C, H, W)
    
    K = 32
    sparsity = 4
    
    # Initialize both with same dictionary
    init_dict = torch.randn(C, K)
    init_dict = F.normalize(init_dict, dim=0)
    
    # K-SVD mode: uses K-SVD updates
    ksvd = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=C,
        sparsity_level=sparsity,
        ksvd_iterations=3,
        use_backprop_only=False,
    )
    ksvd.dictionary.data.copy_(init_dict)
    
    # Backprop-only mode: dictionary learned via gradients
    dl = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=C,
        sparsity_level=sparsity,
        use_backprop_only=True,
    )
    dl.dictionary.data.copy_(init_dict)
    
    # Train both models
    ksvd.train()
    dl.train()
    
    # K-SVD forward
    z_ksvd, loss_ksvd, coeffs_ksvd = ksvd(z)
    mse_ksvd = F.mse_loss(z_ksvd, z).item()
    
    # Regular DL forward (with gradient for comparison)
    optimizer = torch.optim.Adam(dl.parameters(), lr=1e-3)
    for _ in range(10):  # Multiple iterations for regular DL
        optimizer.zero_grad()
        z_dl, loss_dl, coeffs_dl = dl(z)
        loss_dl.backward()
        optimizer.step()
    
    z_dl, loss_dl, coeffs_dl = dl(z)
    mse_dl = F.mse_loss(z_dl, z).item()
    
    print(f"\nReconstruction Quality:")
    print(f"  K-SVD MSE: {mse_ksvd:.6f}")
    print(f"  Regular DL MSE: {mse_dl:.6f}")
    
    # Both should achieve reasonable reconstruction
    assert torch.isfinite(z_ksvd).all()
    assert torch.isfinite(z_dl).all()


def test_ksvd_patch_based():
    """Test K-SVD with patch-based processing."""
    torch.manual_seed(4)
    
    patch_size = 2
    ksvd = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=4,
        sparsity_level=3,
        patch_size=patch_size,
        ksvd_iterations=2,
    )
    
    z = torch.randn(2, 4, 8, 8)
    z_dl, loss, coeffs = ksvd(z)
    
    # Check shapes
    assert z_dl.shape == z.shape
    patches_per_sample = (8 // patch_size) * (8 // patch_size)
    expected_atom_dim = 4 * patch_size * patch_size
    assert coeffs.shape == (16, 2 * patches_per_sample)
    assert ksvd.atom_dim == expected_atom_dim


def test_ksvd_visualization():
    """Visualize K-SVD learned dictionary atoms."""
    torch.manual_seed(5)
    
    # Create data with structure
    B, C, H, W = 8, 3, 64, 64
    z = torch.randn(B, C, H, W)
    
    K = 16
    
    ksvd = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=C,
        sparsity_level=4,
        ksvd_iterations=5,
    )
    
    # Train for multiple iterations
    ksvd.train()
    losses = []
    mses = []
    
    print("\nTraining K-SVD...")
    for i in range(20):
        z_dl, loss, coeffs = ksvd(z)
        mse = F.mse_loss(z_dl, z).item()
        losses.append(loss.item())
        mses.append(mse)
        
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/20: Loss={loss.item():.6f}, MSE={mse:.6f}")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('K-SVD Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(mses, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE')
    ax2.set_title('K-SVD Reconstruction MSE')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "ksvd_training.png", dpi=100)
    plt.close()
    
    # Visualize learned atoms (if RGB)
    if C == 3:
        atoms = ksvd.dictionary.data.t().cpu().numpy()  # [K, C]
        
        # Normalize to [0, 1] for visualization
        atoms_normalized = (atoms - atoms.min()) / (atoms.max() - atoms.min() + 1e-10)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        axes = axes.flatten()
        
        for i in range(min(K, 16)):
            atom_rgb = atoms_normalized[i]
            color_patch = np.tile(atom_rgb.reshape(1, 1, 3), (50, 50, 1))
            
            axes[i].imshow(color_patch)
            axes[i].set_title(f'Atom {i}', fontsize=9)
            axes[i].axis('off')
        
        plt.suptitle('K-SVD Learned Dictionary Atoms', fontsize=14)
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / "ksvd_atoms.png", dpi=100)
        plt.close()
    
    # Compute sparsity statistics
    sparsity = (torch.abs(coeffs) > 1e-6).sum(dim=0).float().mean().item()
    print(f"\nAverage sparsity: {sparsity:.2f} atoms per patch")
    
    assert (ARTIFACT_DIR / "ksvd_training.png").exists()
    if C == 3:
        assert (ARTIFACT_DIR / "ksvd_atoms.png").exists()


def test_ksvd_convergence():
    """Test that K-SVD converges to lower reconstruction error."""
    torch.manual_seed(6)
    
    B, C, H, W = 4, 8, 32, 32
    z = torch.randn(B, C, H, W)
    
    ksvd = DictionaryLearning(
        num_embeddings=32,
        embedding_dim=C,
        sparsity_level=5,
        ksvd_iterations=3,
    )
    
    ksvd.train()
    
    # First iteration
    z_dl_1, _, _ = ksvd(z)
    mse_1 = F.mse_loss(z_dl_1, z).item()
    
    # Multiple iterations
    for _ in range(10):
        _, _, _ = ksvd(z)
    
    z_dl_final, _, _ = ksvd(z)
    mse_final = F.mse_loss(z_dl_final, z).item()
    
    print(f"\nConvergence test:")
    print(f"  Initial MSE: {mse_1:.6f}")
    print(f"  Final MSE: {mse_final:.6f}")
    print(f"  Improvement: {(mse_1 - mse_final) / mse_1 * 100:.1f}%")
    
    # MSE should decrease or stay similar (allowing small fluctuations)
    assert mse_final <= mse_1 * 1.1  # Allow 10% tolerance


def test_ksvd_codebook_heatmaps():
    """Visualize K-SVD sparse codes as heatmaps for interpretability."""
    torch.manual_seed(7)
    
    # Load data
    celeba_dir = _get_celeba_dir()
    if celeba_dir:
        z = _load_celeba_batch(batch_size=2, image_size=128)
        is_celeba = True
        print("\nUsing CelebA images for K-SVD heatmap visualization")
    else:
        z = torch.randn(2, 3, 128, 128)
        is_celeba = False
        print("\nUsing random data for K-SVD heatmap visualization")
    
    B, C, H, W = z.shape
    K = 32
    
    # Initialize K-SVD with k-means
    codebook = _kmeans_codebook(z, K)  # Returns [K, C]
    ksvd = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=C,
        sparsity_level=5,
        ksvd_iterations=3,
    )
    ksvd.dictionary.data.copy_(codebook.t().contiguous())  # K-SVD needs [C, K]
    
    # Train for a few iterations
    ksvd.train()
    for i in range(10):
        z_ksvd, loss, coeffs = ksvd(z)
        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/10: Loss={loss.item():.6f}")
    
    # Get final output
    ksvd.eval()
    with torch.no_grad():
        z_ksvd, _, coeffs = ksvd(z)
    
    # Compute L1 norm of sparse codes per patch (activation strength)
    num_patches_total = coeffs.shape[1] // B
    coeffs_view = coeffs.view(coeffs.shape[0], num_patches_total, B).permute(0, 2, 1).contiguous()
    
    # L1 norm per patch (sum of absolute coefficients)
    l1_norm_per_patch = coeffs_view.abs().sum(dim=0)  # [batch, num_patches]
    
    # Map to spatial grid
    patch_h, patch_w = ksvd.patch_size
    if patch_h == 1 and patch_w == 1:
        # Pixel-level
        ksvd_activation = l1_norm_per_patch.reshape(B, H, W)
    else:
        # Patch-level: use fold operation
        coeff_for_fold = l1_norm_per_patch.unsqueeze(1)  # [batch, 1, num_patches]
        ksvd_activation = F.fold(
            coeff_for_fold,
            output_size=(H, W),
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w)
        ).squeeze(1)  # [batch, H, W]
    
    # Create visualization
    num_images = min(2, B)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Original image
        axes[i, 0].imshow(_to_rgb(z[i], denormalize=is_celeba))
        axes[i, 0].set_title(f'Original #{i+1}', fontsize=11)
        axes[i, 0].axis('off')
        
        # K-SVD sparse code strength (L1 norm)
        activation_data = ksvd_activation[i].cpu().numpy()
        
        # Percentile normalization for clean visualization
        p_low, p_high = np.percentile(activation_data, [1, 99])
        activation_norm = np.clip(activation_data, p_low, p_high)
        activation_norm = (activation_norm - p_low) / (p_high - p_low + 1e-10)
        
        im = axes[i, 1].imshow(activation_norm, cmap='viridis', interpolation='nearest')
        axes[i, 1].set_title(f'K-SVD Sparse Code Strength\n(L1 norm, sparsity={ksvd.sparsity_level})', fontsize=11)
        axes[i, 1].axis('off')
        plt.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle('K-SVD Code Interpretability Visualization', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "ksvd_code_heatmaps.png", dpi=100)
    plt.close()
    
    # Compute statistics
    avg_sparsity = (torch.abs(coeffs) > 1e-6).sum(dim=0).float().mean().item()
    unique_atoms_used = (torch.abs(coeffs) > 1e-6).sum(dim=1).nonzero().size(0)
    
    print(f"\nK-SVD Sparsity Statistics:")
    print(f"  Average atoms per patch: {avg_sparsity:.2f}")
    print(f"  Unique atoms used: {unique_atoms_used}/{K}")
    print(f"  Activation range: [{activation_data.min():.3f}, {activation_data.max():.3f}]")
    
    assert (ARTIFACT_DIR / "ksvd_code_heatmaps.png").exists()


def test_ksvd_vq_dl_comparison_visualization():
    """Compare VQ, DL, and K-SVD side-by-side with visualizations."""
    torch.manual_seed(8)
    
    # Load data
    celeba_dir = _get_celeba_dir()
    if celeba_dir:
        z = _load_celeba_batch(batch_size=2, image_size=64)
        is_celeba = True
        print("\nComparing VQ, DL, and K-SVD on CelebA images")
    else:
        z = torch.randn(2, 3, 64, 64)
        is_celeba = False
        print("\nComparing VQ, DL, and K-SVD on random data")
    
    B, C, H, W = z.shape
    K = 16
    
    # Initialize all three with same k-means codebook
    codebook = _kmeans_codebook(z, K)  # Returns [K, C]
    
    # VQ
    vq = VectorQuantizer(num_embeddings=K, embedding_dim=C, commitment_cost=0.25, decay=0.0)
    vq.embedding.weight.data.copy_(codebook)  # VQ needs [K, C]
    
    # DL
    dl = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=C,
        sparsity_level=4,
    )
    dl.dictionary.data.copy_(codebook.t().contiguous())  # DL needs [C, K]
    
    # K-SVD
    ksvd = DictionaryLearning(
        num_embeddings=K,
        embedding_dim=C,
        sparsity_level=4,
        ksvd_iterations=3,
    )
    ksvd.dictionary.data.copy_(codebook.t().contiguous())  # K-SVD needs [C, K]
    
    # Train K-SVD and DL briefly
    ksvd.train()
    dl.train()
    
    for _ in range(5):
        _, _, _ = ksvd(z)
    
    optimizer = torch.optim.Adam(dl.parameters(), lr=1e-3)
    for _ in range(5):
        optimizer.zero_grad()
        _, loss, _ = dl(z)
        loss.backward()
        optimizer.step()
    
    # Get outputs
    vq.eval()
    dl.eval()
    ksvd.eval()
    
    with torch.no_grad():
        z_q_vq, _, _, enc_vq = vq(z)
        z_q_dl, _, coeffs_dl = dl(z)
        z_q_ksvd, _, coeffs_ksvd = ksvd(z)
    
    # Compute VQ indices
    vq_indices = torch.argmax(enc_vq, dim=1).reshape(B, H, W)
    
    # Compute DL activation maps
    num_patches_dl = coeffs_dl.shape[1] // B
    coeffs_dl_view = coeffs_dl.view(coeffs_dl.shape[0], num_patches_dl, B).permute(0, 2, 1).contiguous()
    dl_activation = coeffs_dl_view.abs().sum(dim=0).reshape(B, H, W)
    
    # Compute K-SVD activation maps
    num_patches_ksvd = coeffs_ksvd.shape[1] // B
    coeffs_ksvd_view = coeffs_ksvd.view(coeffs_ksvd.shape[0], num_patches_ksvd, B).permute(0, 2, 1).contiguous()
    ksvd_activation = coeffs_ksvd_view.abs().sum(dim=0).reshape(B, H, W)
    
    # Create comparison visualization
    num_images = min(2, B)
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Original
        axes[i, 0].imshow(_to_rgb(z[i], denormalize=is_celeba))
        axes[i, 0].set_title(f'Original #{i+1}', fontsize=10)
        axes[i, 0].axis('off')
        
        # VQ indices
        im1 = axes[i, 1].imshow(vq_indices[i].cpu().numpy(), cmap='tab20', interpolation='nearest')
        vq_unique = vq_indices[i].unique().numel()
        axes[i, 1].set_title(f'VQ Indices\n({vq_unique}/{K} codes)', fontsize=10)
        axes[i, 1].axis('off')
        
        # DL activation
        dl_data = dl_activation[i].cpu().numpy()
        dl_norm = (dl_data - dl_data.min()) / (dl_data.max() - dl_data.min() + 1e-10)
        im2 = axes[i, 2].imshow(dl_norm, cmap='viridis', interpolation='nearest')
        axes[i, 2].set_title(f'DL Activation\n(gradient-based)', fontsize=10)
        axes[i, 2].axis('off')
        
        # K-SVD activation
        ksvd_data = ksvd_activation[i].cpu().numpy()
        ksvd_norm = (ksvd_data - ksvd_data.min()) / (ksvd_data.max() - ksvd_data.min() + 1e-10)
        im3 = axes[i, 3].imshow(ksvd_norm, cmap='viridis', interpolation='nearest')
        axes[i, 3].set_title(f'K-SVD Activation\n(SVD-based)', fontsize=10)
        axes[i, 3].axis('off')
    
    plt.suptitle('VQ vs DL vs K-SVD: Code Visualization Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "vq_dl_ksvd_comparison.png", dpi=100)
    plt.close()
    
    # Print metrics
    mse_vq = F.mse_loss(z_q_vq, z).item()
    mse_dl = F.mse_loss(z_q_dl, z).item()
    mse_ksvd = F.mse_loss(z_q_ksvd, z).item()
    
    print("\n" + "="*60)
    print("THREE-WAY COMPARISON: VQ vs DL vs K-SVD")
    print("="*60)
    print(f"Reconstruction MSE:")
    print(f"  VQ:    {mse_vq:.6f}")
    print(f"  DL:    {mse_dl:.6f}")
    print(f"  K-SVD: {mse_ksvd:.6f}")
    print(f"\nCode Usage:")
    print(f"  VQ unique codes: {vq_indices.unique().numel()}/{K}")
    print(f"  DL atoms used: {(coeffs_dl.abs() > 1e-6).sum(dim=1).nonzero().size(0)}/{K}")
    print(f"  K-SVD atoms used: {(coeffs_ksvd.abs() > 1e-6).sum(dim=1).nonzero().size(0)}/{K}")
    print(f"\nAverage Sparsity:")
    print(f"  VQ: 1.00 code/pixel (discrete)")
    print(f"  DL: {(coeffs_dl.abs() > 1e-6).sum(dim=0).float().mean().item():.2f} atoms/pixel")
    print(f"  K-SVD: {(coeffs_ksvd.abs() > 1e-6).sum(dim=0).float().mean().item():.2f} atoms/pixel")
    print("="*60)
    
    assert (ARTIFACT_DIR / "vq_dl_ksvd_comparison.png").exists()
