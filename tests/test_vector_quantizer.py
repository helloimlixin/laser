import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.quantizers import VectorQuantizer, VectorQuantizerEMA


def test_vector_quantizers_smoke():
    torch.manual_seed(0)

    vq_input = torch.randn(2, 4, 8, 8, requires_grad=True)
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    vq_out, vq_loss, _, _ = vq(vq_input)
    assert vq_out.shape == vq_input.shape
    assert torch.isfinite(vq_loss)
    (vq_out.mean() + vq_loss).backward()
    assert vq.embedding.weight.grad is not None

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
