import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

import matplotlib

# Force non-interactive backend so CI environments without displays can render figures.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Compute repository root to load source modules directly.
ROOT = Path(__file__).resolve().parents[1]
# Build path to the actual Python sources under src/.
SRC = ROOT / "src"
# Ensure both root and src live on sys.path so relative imports succeed under pytest.
for path_str in (str(ROOT), str(SRC)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Import the freshly-implemented VectorQuantizer (non-EMA) and VectorQuantizerEMA for focused testing.
from src.models.bottleneck import VectorQuantizer, VectorQuantizerEMA, DictionaryLearning

# Directory to store visualization artifacts emitted by the tests below.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts" / "vq"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _plot_code_usage_histogram(code_counts, output_path):
    """Persist a simple bar chart describing how often each code is selected."""

    plt.figure(figsize=(6, 3))
    plt.bar(np.arange(len(code_counts)), code_counts, color="tab:blue")
    plt.xlabel("Code index")
    plt.ylabel("Selections")
    plt.title("VectorQuantizer code usage")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def _plot_latent_vs_quantized_heatmaps(original, quantized, output_path):
    """Visualize channel 0 of the latent map before/after quantization plus error."""

    # Convert tensors to numpy for matplotlib and detach from autograd graph.
    original_np = original[0].detach().cpu().numpy()
    quantized_np = quantized[0].detach().cpu().numpy()
    error_np = np.abs(original_np - quantized_np)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    images = [original_np, quantized_np, error_np]
    titles = ["Original c0", "Quantized c0", "Absolute error"]
    for ax, data, title in zip(axes, images, titles):
        im = ax.imshow(data, cmap="viridis", interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("VectorQuantizer channel-wise effect", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _render_latent_comparison_frame(original, quantized, step_label):
    """Return an RGB array visualizing original/quantized/error slices for GIF creation."""

    original_np = original[0].detach().cpu().numpy()
    quantized_np = quantized[0].detach().cpu().numpy()
    error_np = np.abs(original_np - quantized_np)

    # Use shared value ranges so colors remain comparable across panels.
    vmin = min(original_np.min(), quantized_np.min())
    vmax = max(original_np.max(), quantized_np.max())
    err_max = error_np.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    images = [original_np, quantized_np, error_np]
    titles = ["Original c0", "Quantized c0", "Abs error"]
    for ax, data, title in zip(axes, images, titles):
        if title == "Abs error":
            im = ax.imshow(data, cmap="magma", interpolation="nearest", vmin=0.0, vmax=err_max)
        else:
            im = ax.imshow(data, cmap="viridis", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{step_label}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.canvas.draw()
    # Matplotlib 3.9 removed tostring_rgb, so use buffer interface.
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def _write_gif(frames, output_path, duration=0.5):
    """Persist a looping GIF using Pillow so we avoid optional imageio dependency."""

    if not frames:
        raise ValueError("Cannot create GIF without frames")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - extremely rare in practice
        raise RuntimeError("Pillow is required to export GIF artifacts") from exc

    pil_frames = [Image.fromarray(frame) for frame in frames]
    first, rest = pil_frames[0], pil_frames[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=int(duration * 1000),
        loop=0,
    )


@pytest.fixture()
def sample_latents():
    """Small spatial map with deterministic randomness for reproducible assertions."""

    # Fix RNG so failing snapshots are stable across runs.
    torch.manual_seed(0)
    # Return a 2×16×4×4 latent tensor with gradients enabled to test backward paths.
    return torch.randn(2, 16, 4, 4, requires_grad=True)


def test_vq_basic_shapes(sample_latents):
    """Quantized outputs keep spatial shape and produce finite stats."""

    # Instantiate VQ with 32 codes and matching embedding dimension.
    vq = VectorQuantizer(num_embeddings=32, embedding_dim=16, commitment_cost=0.25)
    # Run the forward pass to obtain quantized values, auxiliary loss, perplexity, and encodings.
    z_q, loss, perplexity, encodings = vq(sample_latents)

    # Quantized tensor should match the latent shape exactly (straight-through estimator makes it so).
    assert z_q.shape == sample_latents.shape
    # One-hot encodings must be N×K where N is the flattened spatial size.
    assert encodings.shape == (
        sample_latents.shape[0] * sample_latents.shape[2] * sample_latents.shape[3],
        32,
    )
    # Loss value must be scalar and finite to be meaningful for optimization.
    assert loss.ndim == 0 and torch.isfinite(loss)
    # Perplexity serves as a utilization metric; it should also be finite scalar.
    assert perplexity.ndim == 0 and torch.isfinite(perplexity)


def test_vq_backward_updates_codebook(sample_latents):
    """Backward pass must hit both encoder latents and the learnable codebook."""

    # Use a smaller codebook to simplify gradient reasoning.
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=16, commitment_cost=0.25)
    # Capture outputs for the current mini batch.
    z_q, loss, _, _ = vq(sample_latents)

    # Compose a simple scalar objective to force autograd traversal.
    (z_q.mean() + loss).backward()

    # Encoder latents require gradients so training the encoder stays viable.
    assert sample_latents.grad is not None and torch.isfinite(sample_latents.grad).all()
    # The embedding table itself must also accumulate gradients when EMA is disabled.
    assert vq.embedding.weight.grad is not None
    assert torch.isfinite(vq.embedding.weight.grad).all()


def test_vq_codebook_selection():
    """Exact matches in the codebook should round-trip through the quantizer."""

    # Deterministic RNG state for constructing the toy example below.
    torch.manual_seed(2)
    # Build a VQ module whose codebook dimension aligns with our tiny latent volume.
    vq = VectorQuantizer(num_embeddings=3, embedding_dim=4, commitment_cost=0.25)
    vq.eval()

    # Use identity rows as the codebook so index expectations are obvious.
    weight = torch.eye(4)[:3]
    vq.embedding.weight.data.copy_(weight)

    # Predefine which code each spatial site should map to for test readability.
    idx_map = torch.tensor([[0, 1, 2], [2, 1, 0]])
    # Allocate the latent tensor we will populate with matching code vectors.
    z = torch.zeros(1, 4, 2, 3)
    for i in range(2):
        for j in range(3):
            # Fill each site with the corresponding code vector so quantization becomes identity.
            z[0, :, i, j] = weight[idx_map[i, j]]

    # Feed through the VQ module and inspect the quantized output plus one-hot encodings.
    z_q, _, _, encodings = vq(z)
    # Quantized result should match the crafted input exactly, within numerical tolerance.
    assert torch.allclose(z_q, z, atol=1e-6)
    # One-hot argmax indices should reproduce the idx_map layout precisely.
    assert torch.equal(torch.argmax(encodings, dim=1).view(2, 3), idx_map)


def test_vq_channel_mismatch_error():
    """Forward should fail fast when encoder channels disagree with embedding_dim."""

    # Instantiate a VQ module expecting 8-channel latents.
    vq = VectorQuantizer(num_embeddings=4, embedding_dim=8, commitment_cost=0.25)
    # Craft a latent tensor with only 4 channels to trigger the validation path.
    bad_latent = torch.randn(1, 4, 2, 2)

    # A ValueError with the provided message fragment should be raised.
    with pytest.raises(ValueError, match="Expected channel dim"):
        vq(bad_latent)


def test_vq_visualizations():
    """Smoke-test helper visualizations that summarize VQ behavior on random latents."""

    # Use RGB-like latents to make heatmaps visually intuitive.
    torch.manual_seed(5)
    latents = torch.randn(4, 3, 16, 16)

    # Build a modest codebook so usage plots remain readable.
    vq = VectorQuantizer(num_embeddings=16, embedding_dim=3, commitment_cost=0.25)

    with torch.no_grad():
        z_q, _, _, encodings = vq(latents)

    # Derive code selection counts from the one-hot encodings tensor.
    indices = torch.argmax(encodings, dim=1)
    usage_counts = torch.bincount(indices, minlength=vq.num_embeddings).cpu().numpy()

    # Emit histogram summarizing code utilization.
    usage_path = ARTIFACT_DIR / "vq_code_usage.png"
    _plot_code_usage_histogram(usage_counts, usage_path)

    # Visualize the first sample's channel-0 activation before/after quantization.
    heatmap_path = ARTIFACT_DIR / "vq_latent_vs_quantized.png"
    _plot_latent_vs_quantized_heatmaps(latents[0], z_q[0], heatmap_path)

    # Ensure the artifacts exist so CI surfaces failures if plotting breaks.
    assert usage_path.exists()
    assert heatmap_path.exists()


def test_vq_training_loop_generates_gif():
    """Run a tiny training loop to ensure recon error drops and GIF frames converge."""

    torch.manual_seed(11)
    latents = torch.randn(4, 3, 16, 16)
    vq = VectorQuantizer(num_embeddings=32, embedding_dim=3, commitment_cost=0.25)
    optimizer = torch.optim.Adam(vq.parameters(), lr=1e-2)

    recon_history = []
    frames = []
    total_steps = 1000

    for step in range(total_steps):
        optimizer.zero_grad()
        z_q, loss_vq, _, _ = vq(latents)
        recon_loss = torch.nn.functional.mse_loss(z_q, latents)
        total_loss = recon_loss + loss_vq
        total_loss.backward()
        optimizer.step()

        recon_history.append(recon_loss.detach().item())

        # Capture a frame at regular intervals to visualize convergence over time.
        if step % 10 == 0 or step == total_steps - 1:
            frames.append(
                _render_latent_comparison_frame(latents[0], z_q[0], step_label=f"step {step}")
            )

    # Reconstruction error should decrease as the codebook adapts.
    assert recon_history[-1] < recon_history[0]

    gif_path = ARTIFACT_DIR / "vq_training.gif"
    _write_gif(frames, gif_path, duration=0.5)
    assert gif_path.exists()


def test_vq_ema_basic_shapes(sample_latents):
    """EMA variant: Quantized outputs keep spatial shape and produce finite stats."""

    # Instantiate EMA VQ with 32 codes and matching embedding dimension.
    vq_ema = VectorQuantizerEMA(num_embeddings=32, embedding_dim=16, commitment_cost=0.25, ema_decay=0.99)
    # Run the forward pass to obtain quantized values, auxiliary loss, perplexity, and encodings.
    z_q, loss, perplexity, encodings = vq_ema(sample_latents)

    # Quantized tensor should match the latent shape exactly (straight-through estimator makes it so).
    assert z_q.shape == sample_latents.shape
    # One-hot encodings must be N×K where N is the flattened spatial size.
    assert encodings.shape == (
        sample_latents.shape[0] * sample_latents.shape[2] * sample_latents.shape[3],
        32,
    )
    # Loss value must be scalar and finite to be meaningful for optimization.
    assert loss.ndim == 0 and torch.isfinite(loss)
    # Perplexity serves as a utilization metric; it should also be finite scalar.
    assert perplexity.ndim == 0 and torch.isfinite(perplexity)


def test_vq_ema_backward_updates_encoder_only(sample_latents):
    """EMA variant: Backward pass only hits encoder latents, codebook updates via EMA."""

    # Use a smaller codebook to simplify gradient reasoning.
    vq_ema = VectorQuantizerEMA(num_embeddings=8, embedding_dim=16, commitment_cost=0.25, ema_decay=0.99)
    # Capture outputs for the current mini batch.
    z_q, loss, _, _ = vq_ema(sample_latents)

    # Compose a simple scalar objective to force autograd traversal.
    (z_q.mean() + loss).backward()

    # Encoder latents require gradients so training the encoder stays viable.
    assert sample_latents.grad is not None and torch.isfinite(sample_latents.grad).all()
    # The embedding table itself does NOT accumulate gradients when EMA is enabled.
    assert vq_ema.embedding.weight.grad is None


def test_vq_ema_codebook_selection():
    """EMA variant: Exact matches in the codebook should round-trip through the quantizer."""

    # Deterministic RNG state for constructing the toy example below.
    torch.manual_seed(2)
    # Build an EMA VQ module whose codebook dimension aligns with our tiny latent volume.
    vq_ema = VectorQuantizerEMA(num_embeddings=3, embedding_dim=4, commitment_cost=0.25, ema_decay=0.99)
    vq_ema.eval()

    # Use identity rows as the codebook so index expectations are obvious.
    weight = torch.eye(4)[:3]
    vq_ema.embedding.weight.data.copy_(weight)

    # Predefine which code each spatial site should map to for test readability.
    idx_map = torch.tensor([[0, 1, 2], [2, 1, 0]])
    # Allocate the latent tensor we will populate with matching code vectors.
    z = torch.zeros(1, 4, 2, 3)
    for i in range(2):
        for j in range(3):
            # Fill each site with the corresponding code vector so quantization becomes identity.
            z[0, :, i, j] = weight[idx_map[i, j]]

    # Feed through the EMA VQ module and inspect the quantized output plus one-hot encodings.
    z_q, _, _, encodings = vq_ema(z)
    # Quantized result should match the crafted input exactly, within numerical tolerance.
    assert torch.allclose(z_q, z, atol=1e-6)
    # One-hot argmax indices should reproduce the idx_map layout precisely.
    assert torch.equal(torch.argmax(encodings, dim=1).view(2, 3), idx_map)


def test_vq_ema_channel_mismatch_error():
    """EMA variant: Forward should fail fast when encoder channels disagree with embedding_dim."""

    # Instantiate an EMA VQ module expecting 8-channel latents.
    vq_ema = VectorQuantizerEMA(num_embeddings=4, embedding_dim=8, commitment_cost=0.25, ema_decay=0.99)
    # Craft a latent tensor with only 4 channels to trigger the validation path.
    bad_latent = torch.randn(1, 4, 2, 2)

    # A ValueError with the provided message fragment should be raised.
    with pytest.raises(ValueError, match="Expected channel dim"):
        vq_ema(bad_latent)


def test_vq_ema_updates_codebook_during_training(sample_latents):
    """EMA variant: Codebook should be updated via EMA during training mode."""

    # Create EMA VQ and capture initial codebook state.
    vq_ema = VectorQuantizerEMA(num_embeddings=8, embedding_dim=16, commitment_cost=0.25, ema_decay=0.5)
    initial_weights = vq_ema.embedding.weight.data.clone()

    # Ensure we're in training mode.
    vq_ema.train()
    # Run forward pass to trigger EMA updates.
    vq_ema(sample_latents)

    # Codebook should have been updated via EMA.
    assert not torch.equal(vq_ema.embedding.weight.data, initial_weights)


def test_vq_ema_no_updates_during_eval(sample_latents):
    """EMA variant: Codebook should NOT be updated during evaluation mode."""

    # Create EMA VQ and capture initial codebook state.
    vq_ema = VectorQuantizerEMA(num_embeddings=8, embedding_dim=16, commitment_cost=0.25, ema_decay=0.5)
    initial_weights = vq_ema.embedding.weight.data.clone()

    # Ensure we're in evaluation mode.
    vq_ema.eval()
    # Run forward pass - should not trigger EMA updates.
    vq_ema(sample_latents)

    # Codebook should remain unchanged.
    assert torch.equal(vq_ema.embedding.weight.data, initial_weights)


def test_vq_ema_cluster_size_initialization():
    """EMA variant: Cluster sizes should be properly initialized to zeros."""

    vq_ema = VectorQuantizerEMA(num_embeddings=16, embedding_dim=8, commitment_cost=0.25, ema_decay=0.99)

    # Cluster sizes should start at zero.
    assert torch.all(vq_ema._buffers['ema_cluster_size'] == 0.0)
    # EMA weights should be initialized.
    assert hasattr(vq_ema, '_ema_w')
    assert vq_ema._ema_w.shape == (16, 8)


def test_vq_ema_decay_parameter_effect():
    """EMA variant: Different decay values should produce different update behaviors."""

    torch.manual_seed(42)
    latents = torch.randn(2, 8, 4, 4)

    # Create two EMA VQs with different decay rates.
    vq_slow = VectorQuantizerEMA(num_embeddings=4, embedding_dim=8, commitment_cost=0.25, ema_decay=0.99)
    vq_fast = VectorQuantizerEMA(num_embeddings=4, embedding_dim=8, commitment_cost=0.25, ema_decay=0.5)

    # Both should start with same random initialization.
    vq_fast.embedding.weight.data.copy_(vq_slow.embedding.weight.data)
    vq_fast._ema_w.data.copy_(vq_slow._ema_w.data)

    # Run training steps.
    vq_slow.train()
    vq_fast.train()
    vq_slow(latents)
    vq_fast(latents)

    # Different decay rates should lead to different final codebooks.
    assert not torch.allclose(vq_slow.embedding.weight.data, vq_fast.embedding.weight.data, atol=1e-4)


def test_vq_ema_visualizations():
    """EMA variant: Smoke-test helper visualizations that summarize EMA VQ behavior on random latents."""

    # Use RGB-like latents to make heatmaps visually intuitive.
    torch.manual_seed(5)
    latents = torch.randn(4, 3, 16, 16)

    # Build a modest codebook so usage plots remain readable.
    vq_ema = VectorQuantizerEMA(num_embeddings=16, embedding_dim=3, commitment_cost=0.25, ema_decay=0.99)

    with torch.no_grad():
        z_q, _, _, encodings = vq_ema(latents)

    # Derive code selection counts from the one-hot encodings tensor.
    indices = torch.argmax(encodings, dim=1)
    usage_counts = torch.bincount(indices, minlength=vq_ema.num_embeddings).cpu().numpy()

    # Emit histogram summarizing code utilization.
    usage_path = ARTIFACT_DIR / "vq_ema_code_usage.png"
    _plot_code_usage_histogram(usage_counts, usage_path)

    # Visualize the first sample's channel-0 activation before/after quantization.
    heatmap_path = ARTIFACT_DIR / "vq_ema_latent_vs_quantized.png"
    _plot_latent_vs_quantized_heatmaps(latents[0], z_q[0], heatmap_path)

    # Ensure the artifacts exist so CI surfaces failures if plotting breaks.
    assert usage_path.exists()
    assert heatmap_path.exists()


def test_vq_ema_training_loop_generates_gif():
    """EMA variant: Run a tiny training loop to ensure encoder adapts and codebook updates via EMA."""

    torch.manual_seed(11)
    latents = torch.randn(4, 3, 16, 16)
    vq_ema = VectorQuantizerEMA(num_embeddings=32, embedding_dim=3, commitment_cost=0.25, ema_decay=0.99)

    # For EMA VQ, we only optimize the encoder-like parameters (simulated here with a simple linear layer)
    encoder_sim = nn.Linear(3, 3)  # Simulate encoder that maps to latent space
    optimizer = torch.optim.Adam([encoder_sim.weight, encoder_sim.bias], lr=1e-2)

    recon_history = []
    frames = []
    total_steps = 1000

    for step in range(total_steps):
        optimizer.zero_grad()
        # Simulate encoder forward pass
        z_e = encoder_sim(latents.view(-1, 3)).view(latents.shape)
        z_q, loss_vq, _, _ = vq_ema(z_e)
        recon_loss = torch.nn.functional.mse_loss(z_q, latents)
        total_loss = recon_loss + loss_vq
        total_loss.backward()
        optimizer.step()

        recon_history.append(recon_loss.detach().item())

        # Capture a frame at regular intervals to visualize convergence over time.
        if step % 10 == 0 or step == total_steps - 1:
            frames.append(
                _render_latent_comparison_frame(latents[0], z_q[0], step_label=f"step {step}")
            )

    # Reconstruction error should decrease as both encoder and codebook adapt.
    assert recon_history[-1] < recon_history[0]

    gif_path = ARTIFACT_DIR / "vq_ema_training.gif"
    _write_gif(frames, gif_path, duration=0.5)
    assert gif_path.exists()

def test_dictionary_learning_forward_shapes():
    """DictionaryLearning: Output shapes and loss."""
    torch.manual_seed(42)
    # [B, C, H, W]
    x = torch.randn(2, 4, 8, 8)
    
    # Init DL: patch_size=2 -> atom_dim = 4 * 2 * 2 = 16
    dl = DictionaryLearning(
        num_embeddings=32, 
        embedding_dim=4, 
        patch_size=2,
        sparsity_level=2,
        sparse_solver='omp'
    )
    
    z_dl, loss, coeffs = dl(x)
    
    # Shapes
    assert z_dl.shape == x.shape
    assert torch.isfinite(loss)
    # coeffs: [atom_dim, B*L] -> [16, 2 * (4*4)] = [16, 32] ?? 
    # Wait, batch_omp returns [num_embeddings, B*L].
    # Let's check the code:
    # signals = patches_flat.t() [atom_dim, N]
    # coeffs = self.batch_omp(signals, self.dictionary) -> [num_embeddings, N]
    # So coeffs should be [32, 32] in this case where L = (8/2)*(8/2) = 16 per image. Total N=32.
    
    # 8x8 image, patch=2, stride=2 -> 4x4 patches = 16 patches per image.
    # Batch=2 -> 32 patches total.
    # num_embeddings = 32.
    assert coeffs.shape == (32, 2 * 16)
    
def test_dictionary_learning_sparsity():
    """DictionaryLearning: Check if coefficients are actually sparse."""
    torch.manual_seed(42)
    x = torch.randn(1, 4, 8, 8)
    dl = DictionaryLearning(
        num_embeddings=32, 
        embedding_dim=4, 
        patch_size=2,
        sparsity_level=3,
        sparse_solver='omp'
    )
    
    _, _, coeffs = dl(x)
    
    # Check max non-zeros per column
    # coeffs is [num_embeddings, N]
    non_zeros = (coeffs.abs() > 1e-6).sum(dim=0)
    assert non_zeros.max() <= 3
