"""Core bottleneck behavior: sparse codes train separately from token quantization."""

import torch

from src.models.bottleneck import DictionaryLearning, SparseCodes, VectorQuantizerEMA
from src.sparse_token_codec import sparse_codes_to_tokens, tokens_to_sparse_codes


def test_dictionary_learning_forward_and_token_roundtrip():
    torch.manual_seed(0)
    dl = DictionaryLearning(num_embeddings=8, embedding_dim=4, sparsity_level=2)
    z = torch.randn(2, 4, 3, 3, requires_grad=True)

    z_q, loss, codes = dl(z)

    assert z_q.shape == z.shape
    assert torch.isfinite(loss)
    assert isinstance(codes, SparseCodes)
    assert codes.support.shape == (2, 3, 3, 2)
    assert codes.values.shape == codes.support.shape

    # Token/cache extraction owns coefficient quantization; the bottleneck keeps
    # only real-valued sparse coefficients.
    tokens, coeff_q = sparse_codes_to_tokens(
        codes,
        num_embeddings=dl.num_embeddings,
        sparsity_level=dl.sparsity_level,
        coeff_vocab_size=5,
        coeff_max=2.0,
    )
    support, values = tokens_to_sparse_codes(
        tokens,
        num_embeddings=dl.num_embeddings,
        sparsity_level=dl.sparsity_level,
        coeff_vocab_size=5,
        coeff_max=2.0,
    )
    decoded = dl._reconstruct_sparse(support, values, height=z.shape[-2], width=z.shape[-1])

    assert tokens.shape == (2, 3, 3, 4)
    assert coeff_q.shape == codes.values.shape
    assert decoded.shape == z.shape

    assert torch.allclose(loss.detach(), dl._last_bottleneck_loss)
    assert torch.allclose(
        dl._last_bottleneck_objective,
        dl._last_dictionary_loss + dl._last_bottleneck_loss,
    )
    (z_q.mean() + dl._last_bottleneck_objective_for_backward).backward()
    assert z.grad is not None
    assert dl.dictionary.grad is not None


def test_ema_quantizer_runs_without_codebook_gradients():
    torch.manual_seed(0)
    vq = VectorQuantizerEMA(
        num_embeddings=8,
        embedding_dim=4,
        commitment_cost=0.25,
        ema_decay=0.99,
        codebook_init=True,
    )
    vq.train()
    z = torch.randn(2, 4, 3, 3, requires_grad=True)

    z_q, loss, perplexity, encodings = vq(z)

    assert z_q.shape == z.shape
    assert torch.isfinite(loss)
    assert torch.isfinite(perplexity)
    assert encodings.shape == (18, 8)
    assert bool(vq._codebook_initialized.item())
    assert vq.embedding.weight.requires_grad is False
