import torch
import pytest
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.bottleneck import VectorQuantizer


ARTIFACTS_DIR = (Path(__file__).resolve().parent / 'artifacts' / 'vq')
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_shapes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    torch.manual_seed(0)
    dev = torch.device(device)
    B, D, H, W = (4, 16, 8, 8) if device == "cpu" else (2, 16, 8, 8)
    K = 32 if device == "cpu" else 64

    vq = VectorQuantizer(num_embeddings=K, embedding_dim=D, commitment_cost=0.25, decay=0.99).to(dev)
    vq.eval()  # disable EMA updates

    z = torch.randn(B, D, H, W, device=dev, requires_grad=True)
    z_q, loss, perplexity, encodings = vq(z)

    assert z_q.shape == z.shape
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Perplexity range: [1, K]
    assert perplexity.ndim == 0
    assert 1.0 <= float(perplexity) <= float(K) + 1e-3

    # One-hot encodings with correct shape
    assert encodings.shape == (B * H * W, K)
    rowsums = encodings.sum(dim=1)
    assert torch.allclose(rowsums, torch.ones_like(rowsums))
    assert torch.all((encodings == 0) | (encodings == 1))


def test_straight_through_grad():
    torch.manual_seed(0)
    B, D, H, W = 2, 8, 4, 4
    K = 16

    vq = VectorQuantizer(num_embeddings=K, embedding_dim=D, commitment_cost=0.25, decay=0.99)
    vq.eval()

    z = torch.randn(B, D, H, W, requires_grad=True)
    z_q, loss, _, _ = vq(z)
    total = z_q.mean() + loss
    total.backward()

    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_exact_selection_no_ema():
    # Ensure exact nearest-neighbor selection when embeddings are set and EMA is off
    torch.manual_seed(0)
    B, D, H, W = 1, 4, 2, 3
    K = 3

    vq = VectorQuantizer(num_embeddings=K, embedding_dim=D, commitment_cost=0.25, decay=0.0)
    vq.eval()

    # Set codebook to known vectors
    # e0 = [1,0,0,0], e1 = [0,1,0,0], e2 = [0,0,1,0]
    weight = torch.zeros(K, D)
    weight[0, 0] = 1.0
    weight[1, 1] = 1.0
    weight[2, 2] = 1.0
    with torch.no_grad():
        vq.embedding.weight.copy_(weight)

    # Build z with a pattern of exact embeddings at positions
    idx_map = torch.tensor([
        [0, 1, 2],
        [2, 1, 0],
    ])  # H x W
    z = torch.zeros(B, D, H, W)
    for i in range(H):
        for j in range(W):
            z[0, :, i, j] = weight[idx_map[i, j]]

    z_q, loss, perplexity, encodings = vq(z)

    # Quantized output should match input exactly at all positions
    assert torch.allclose(z_q, z, atol=1e-6)

    # Encodings argmax should equal desired indices
    enc = encodings.view(H * W, K)
    chosen = torch.argmax(enc, dim=1).view(H, W)
    assert torch.equal(chosen, idx_map)

    # Perplexity should be between 1 and K, and here > 1 since we used multiple codes
    assert 1.0 <= float(perplexity) <= float(K) + 1e-3


def test_ema_eval_no_update():
    torch.manual_seed(0)
    B, D, H, W = 2, 8, 4, 4
    K = 16
    vq = VectorQuantizer(num_embeddings=K, embedding_dim=D, commitment_cost=0.25, decay=0.99)
    vq.eval()

    before = vq.embedding.weight.detach().clone()
    z = torch.randn(B, D, H, W)
    _ = vq(z)
    after = vq.embedding.weight.detach().clone()

    # In eval mode, EMA path should not run; weights unchanged
    assert torch.allclose(before, after)


def test_ema_updates_train():
    torch.manual_seed(0)
    B, D, H, W = 2, 8, 4, 4
    K = 16
    vq = VectorQuantizer(num_embeddings=K, embedding_dim=D, commitment_cost=0.25, decay=0.99)
    vq.train()

    # Start with zeros counter
    with torch.no_grad():
        vq._ema_cluster_size.zero_()

    z = torch.randn(B, D, H, W)
    _ = vq(z)

    # EMA cluster size should have received counts
    assert (vq._ema_cluster_size > 0).any()


def test_save_visuals():
    torch.manual_seed(0)
    B, D, H, W = 2, 16, 8, 8
    K = 64

    vq = VectorQuantizer(num_embeddings=K, embedding_dim=D, commitment_cost=0.25, decay=0.99)
    vq.eval()

    z = torch.randn(B, D, H, W)
    z_q, loss, perplexity, encodings = vq(z)

    # 1) Codebook usage histogram
    usage = encodings.sum(dim=0).cpu().numpy()
    plt.figure(figsize=(8, 3))
    plt.bar(range(K), usage)
    plt.title('VQ Codebook Usage')
    plt.xlabel('Code index')
    plt.ylabel('Count')
    usage_path = ARTIFACTS_DIR / 'codebook_usage.png'
    plt.tight_layout()
    plt.savefig(usage_path.as_posix())
    plt.close()

    # 2) Before/after single-channel heatmaps
    ch = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(z[0, ch].cpu().numpy(), cmap='viridis')
    plt.title('z (ch=0)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(z_q[0, ch].detach().cpu().numpy(), cmap='viridis')
    plt.title('z_q (ch=0)')
    plt.axis('off')
    comp_path = ARTIFACTS_DIR / 'channel_comparison.png'
    plt.tight_layout()
    plt.savefig(comp_path.as_posix())
    plt.close()

    assert usage_path.exists() and usage_path.stat().st_size > 0
    assert comp_path.exists() and comp_path.stat().st_size > 0


