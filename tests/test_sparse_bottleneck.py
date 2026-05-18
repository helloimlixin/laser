"""Core bottleneck behavior: sparse codes can be trained, tokenized, and decoded."""

import torch

from src.models.bottleneck import DictionaryLearning, SparseCodes, VectorQuantizerEMA


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

    # Stage 2 consumes this interleaved atom/bin token stream, so the roundtrip
    # is the contract worth keeping under test.
    tokens, coeff_q = dl.sparse_codes_to_tokens(
        codes,
        coeff_vocab_size=5,
        coeff_max=2.0,
    )
    decoded = dl.tokens_to_latent(
        tokens,
        coeff_vocab_size=5,
        coeff_max=2.0,
    )

    assert tokens.shape == (2, 3, 3, 4)
    assert coeff_q.shape == codes.values.shape
    assert decoded.shape == z.shape

    (z_q.mean() + loss).backward()
    assert z.grad is not None
    assert dl.dictionary.grad is not None


def test_dictionary_learning_large_dictionary_updates_usage_ema():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=128,
        embedding_dim=4,
        sparsity_level=2,
        patch_based=True,
        patch_size=2,
        patch_stride=2,
        dictionary_usage_grad_scale=0.5,
    )
    z = torch.randn(2, 4, 4, 4, requires_grad=True)

    z_q, loss, codes = dl(z)

    assert z_q.shape == z.shape
    assert codes.support.shape == (2, 2, 2, 2)
    assert int(dl.dictionary_usage_steps.item()) == 1
    assert float(dl.dictionary_usage_ema.max()) > 0.0

    (z_q.mean() + loss).backward()
    assert dl.dictionary.grad is not None
    dl.project_dictionary_gradient_()
    after = dl.dictionary.grad.detach()

    assert torch.isfinite(after).all()


def test_dictionary_learning_coherence_stats_are_sampled_for_large_dictionaries():
    torch.manual_seed(0)
    dl = DictionaryLearning(num_embeddings=128, embedding_dim=16, sparsity_level=2)

    exact = dl.coherence_stats(max_exact_atoms=128)
    sampled = dl.coherence_stats(max_exact_atoms=16, sample_atoms=8)

    assert len(exact) == 3
    assert len(sampled) == 3
    assert all(torch.isfinite(value) for value in exact)
    assert all(torch.isfinite(value) for value in sampled)


def test_dictionary_learning_online_ksvd_updates_dictionary_without_gradients():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=4,
        sparsity_level=2,
        dictionary_update_mode="online_ksvd",
        dictionary_ksvd_lr=0.5,
        dictionary_ksvd_max_atoms_per_step=8,
        dictionary_usage_ema_decay=0.0,
    )
    dl.train()
    z = torch.randn(2, 4, 3, 3, requires_grad=True)

    z_q, loss, _ = dl(z)
    before = dl.dictionary.detach().clone()
    (z_q.mean() + loss).backward()

    assert z.grad is not None
    assert dl.dictionary.grad is None
    dl.online_ksvd_update_()

    after = dl.dictionary.detach()
    assert torch.isfinite(after).all()
    assert torch.allclose(after.norm(dim=0), torch.ones_like(after.norm(dim=0)), atol=1e-4)
    assert not torch.allclose(before, after)


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
