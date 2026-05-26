"""Core stage-2 behavior: cached sparse tokens build, train, and generate."""

from types import SimpleNamespace

import torch

from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_cache,
    infer_sparse_vocab_sizes,
)
from src.stage2_preview import _build_sparse_visuals, _sparse_generation_stats


def _quantized_cache():
    tokens = torch.tensor(
        [
            [0, 3, 1, 4, 1, 5, 2, 6],
            [1, 4, 2, 5, 0, 3, 2, 6],
            [2, 6, 0, 3, 1, 4, 0, 5],
            [0, 5, 2, 4, 1, 6, 2, 3],
        ],
        dtype=torch.int32,
    )
    return {
        "tokens_flat": tokens,
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_vocab_size": 4,
            "coef_max": 2.0,
            "coeff_bin_values": torch.linspace(-2.0, 2.0, steps=4),
        },
    }


def _real_valued_cache():
    return {
        "tokens_flat": torch.tensor(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 0],
            ],
            dtype=torch.int32,
        ),
        "coeffs_flat": torch.tensor(
            [
                [1.0, -2.0, 3.0, -4.0],
                [-1.5, 2.5, -3.5, 4.5],
            ],
            dtype=torch.float32,
        ),
        "shape": (1, 2, 2),
        "meta": {
            "num_atoms": 8,
            "coef_max": 5.0,
            "quantize_sparse_coeffs": False,
            "variational_coeffs": False,
        },
    }


def test_stage2_preview_writes_sparse_visual_maps(tmp_path):
    batch = SimpleNamespace(
        imgs=torch.randn(2, 3, 8, 8),
        atoms=torch.arange(2 * 3 * 4 * 2).view(2, 3, 4, 2) % 8,
        coeffs=torch.randn(2, 3, 4, 2),
        toks=None,
    )
    cache = {"meta": {"num_atoms": 8}}

    paths = _build_sparse_visuals(batch, cache, tmp_path, stem="preview")
    stats = _sparse_generation_stats(batch, cache)

    assert {"atom_id_maps", "coeff_value_maps", "coeff_abs_maps"} <= set(paths)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert int(stats["generation/unique_atoms"]) == 8
    assert "generation/coeff_abs_mean" in stats


def test_stage2_preview_writes_quantized_coeff_bin_maps(tmp_path):
    toks = torch.tensor(
        [
            [
                [[0, 8, 1, 9], [2, 10, 3, 11]],
                [[4, 12, 5, 13], [6, 14, 7, 15]],
            ]
        ],
        dtype=torch.long,
    )
    batch = SimpleNamespace(imgs=torch.randn(1, 3, 8, 8), toks=toks, atoms=None, coeffs=None)
    cache = {"meta": {"num_atoms": 8}}

    paths = _build_sparse_visuals(batch, cache, tmp_path, stem="quantized")
    stats = _sparse_generation_stats(batch, cache)

    assert {"atom_id_maps", "coeff_bin_maps"} <= set(paths)
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())
    assert stats["generation/coeff_bin_min"] == 0.0
    assert stats["generation/coeff_bin_max"] == 7.0


def test_sparse_prior_builds_trains_and_generates_quantized_tokens():
    torch.manual_seed(0)
    cache = _quantized_cache()

    vocab, atom_vocab, coeff_vocab = infer_sparse_vocab_sizes(
        cache,
        total_vocab_size=None,
        atom_vocab_size=None,
        coeff_vocab_size=None,
    )
    prior = build_sparse_prior_from_cache(
        cache,
        architecture="spatial_depth",
        total_vocab_size=vocab,
        atom_vocab_size=atom_vocab,
        coeff_vocab_size=coeff_vocab,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    module = SparseTokenPriorModule(
        prior=prior,
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
    )
    module.log = lambda *args, **kwargs: None

    loss = module.training_step(cache["tokens_flat"].to(torch.long), batch_idx=0)
    generated = module.generate_tokens(batch_size=2, top_k=2)

    assert (vocab, atom_vocab, coeff_vocab) == (7, 3, 4)
    assert torch.isfinite(loss)
    assert generated.shape == (2, 2, 4)
    # Atom positions and coefficient positions live in different vocab ranges.
    assert generated[..., 0::2].max() < atom_vocab
    assert generated[..., 1::2].min() >= atom_vocab
    assert generated[..., 1::2].max() < vocab


def test_atom_id_ordered_cache_generates_sorted_supports():
    torch.manual_seed(1)
    cache = {
        "tokens_flat": torch.tensor(
            [
                [0, 5, 2, 6],
                [1, 7, 3, 5],
                [0, 6, 4, 7],
            ],
            dtype=torch.int32,
        ),
        "shape": (1, 1, 4),
        "meta": {
            "num_atoms": 5,
            "coeff_vocab_size": 3,
            "coeff_bin_values": torch.linspace(-1.0, 1.0, steps=3),
            "support_order": "atom_id",
        },
    }
    vocab, atom_vocab, coeff_vocab = infer_sparse_vocab_sizes(
        cache,
        total_vocab_size=None,
        atom_vocab_size=None,
        coeff_vocab_size=None,
    )
    prior = build_sparse_prior_from_cache(
        cache,
        architecture="spatial_depth",
        total_vocab_size=vocab,
        atom_vocab_size=atom_vocab,
        coeff_vocab_size=coeff_vocab,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    module = SparseTokenPriorModule(prior=prior, learning_rate=1e-3, weight_decay=0.0, warmup_steps=0)

    generated = module.generate_tokens(batch_size=8)
    atom_tokens = generated[..., 0::2]

    assert generated.shape == (8, 1, 4)
    assert torch.all(atom_tokens[..., 1:] > atom_tokens[..., :-1])


def test_real_valued_non_gaussian_coeff_sampling_falls_back_to_mean():
    cache = _real_valued_cache()
    prior = build_sparse_prior_from_cache(
        cache,
        architecture="spatial_depth",
        total_vocab_size=None,
        atom_vocab_size=None,
        coeff_vocab_size=None,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )

    import pytest

    with pytest.warns(RuntimeWarning, match="requires a variational/Gaussian coefficient head"):
        module = SparseTokenPriorModule(prior=prior, sample_coeff_mode="gaussian")

    assert module.sample_coeff_mode == "mean"
