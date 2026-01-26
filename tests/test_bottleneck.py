import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path_str in (str(ROOT), str(SRC)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.models.bottleneck import (
    DictionaryLearning,
    VectorQuantizer,
    VectorQuantizerEMA,
)


def test_bottleneck_smoke():
    torch.manual_seed(0)

    # VectorQuantizer: basic shape + gradient check.
    vq_input = torch.randn(2, 4, 8, 8, requires_grad=True)
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    vq_out, vq_loss, _, _ = vq(vq_input)
    assert vq_out.shape == vq_input.shape
    assert torch.isfinite(vq_loss)
    (vq_out.mean() + vq_loss).backward()
    assert vq.embedding.weight.grad is not None

    # VectorQuantizerEMA: basic shape + no codebook gradients.
    vq_ema_input = torch.randn(2, 4, 8, 8, requires_grad=True)
    vq_ema = VectorQuantizerEMA(
        num_embeddings=8,
        embedding_dim=4,
        commitment_cost=0.25,
        ema_decay=0.99,
    )
    vq_ema_out, vq_ema_loss, _, _ = vq_ema(vq_ema_input)
    assert vq_ema_out.shape == vq_ema_input.shape
    assert torch.isfinite(vq_ema_loss)
    (vq_ema_out.mean() + vq_ema_loss).backward()
    assert vq_ema.embedding.weight.grad is None

    # DictionaryLearning: per-pixel sparse coding smoke test.
    dl_input = torch.randn(2, 4, 8, 8, requires_grad=True)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
        sparse_solver="omp",
    )
    dl_out, dl_loss, coeffs = dl(dl_input)
    assert dl_out.shape == dl_input.shape
    assert torch.isfinite(dl_loss)
    expected_n = dl_input.shape[0] * dl_input.shape[2] * dl_input.shape[3]
    assert coeffs.shape == (dl.num_embeddings, expected_n)
    (dl_out.mean() + dl_loss).backward()
    assert dl.dictionary.grad is not None
