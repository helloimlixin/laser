import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


scratch_laser = _load_module("scratch_laser_tokenization_test_module", "scratch/laser.py")

DictionaryLearning = scratch_laser.DictionaryLearning
Prior = scratch_laser.Prior
PriorConfig = scratch_laser.PriorConfig


def test_quantized_tokenization_uses_separate_atom_and_coeff_ranges():
    bottleneck = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=2,
        sparsity_level=2,
        patch_size=1,
        patch_stride=1,
        n_bins=5,
        coef_max=2.0,
        quantize_sparse_coeffs=True,
        epsilon=1e-6,
    )

    support = torch.tensor([[[[0, 3]]]], dtype=torch.long)
    bin_idx = torch.tensor([[[[1, 4]]]], dtype=torch.long)

    tokens = bottleneck._pack_quantized_tokens(support, bin_idx)
    atom_ids, coeffs = bottleneck._unpack_quantized_tokens(tokens)

    assert tokens.shape == (1, 1, 1, 4)
    assert torch.equal(tokens.view(-1), torch.tensor([0, 5, 3, 8], dtype=torch.long))
    assert torch.equal(atom_ids, support)
    assert torch.allclose(coeffs, bottleneck._dequantize_coeff(bin_idx), atol=1e-6)
    assert bottleneck.pad_token_id == 9
    assert bottleneck.bos_token_id == 10
    assert bottleneck.vocab_size == 11


def test_quantized_tokens_to_latent_round_trips_alternating_token_pairs():
    bottleneck = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=2,
        patch_size=1,
        patch_stride=1,
        n_bins=5,
        coef_max=1.0,
        quantize_sparse_coeffs=True,
        epsilon=1e-6,
    )
    with torch.no_grad():
        bottleneck.dictionary.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )

    support = torch.tensor([[[[0, 1]]]], dtype=torch.long)
    bin_idx = torch.tensor([[[[4, 0]]]], dtype=torch.long)
    tokens = bottleneck._pack_quantized_tokens(support, bin_idx)

    z_q = bottleneck.tokens_to_latent(tokens)
    expected = bottleneck._reconstruct_sparse(support, bottleneck._dequantize_coeff(bin_idx))

    assert z_q.shape == (1, 2, 1, 1)
    assert torch.allclose(z_q, expected, atol=1e-6)


def test_non_overlapping_patch_round_trip_preserves_layout():
    bottleneck = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=1,
        sparsity_level=1,
        patch_size=2,
        patch_stride=2,
        quantize_sparse_coeffs=False,
        epsilon=1e-6,
    )
    z = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)

    patches, patch_h, patch_w = bottleneck._extract_patches(z)
    z_q = bottleneck._fold_patches(patches, patch_h, patch_w)

    assert patches.shape == (1, 2, 2, 4)
    assert z_q.shape == z.shape
    assert torch.allclose(z_q, z, atol=1e-6)


def test_overlapping_patch_round_trip_preserves_layout():
    bottleneck = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=1,
        sparsity_level=1,
        patch_size=2,
        patch_stride=1,
        quantize_sparse_coeffs=False,
        epsilon=1e-6,
    )
    z = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)

    patches, patch_h, patch_w = bottleneck._extract_patches(z)
    z_q = bottleneck._fold_patches(patches, patch_h, patch_w)

    assert patches.shape == (1, 3, 3, 4)
    assert z_q.shape == z.shape
    assert torch.allclose(z_q, z, atol=1e-6)


def test_sparse_coefficients_are_clipped_to_signal_l2_norm():
    bottleneck = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=1,
        sparsity_level=2,
        patch_size=1,
        patch_stride=1,
        quantize_sparse_coeffs=False,
        epsilon=1e-6,
    )
    signals = torch.tensor(
        [
            [3.0, 4.0],
            [4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    coeffs = torch.tensor(
        [
            [8.0, -9.0],
            [5.0, -6.0],
        ],
        dtype=torch.float32,
    )

    clipped = bottleneck._bound_coeffs_by_signal_norm(coeffs, signals)

    # Signal norms are [5, 4], so every coefficient should be clipped into [-norm, norm].
    assert torch.allclose(
        clipped,
        torch.tensor([[5.0, -5.0], [4.0, -4.0]], dtype=torch.float32),
        atol=1e-6,
    )


def test_prior_generation_masks_atom_and_coeff_token_ranges():
    torch.manual_seed(0)

    cfg = PriorConfig(
        vocab_size=9,
        H=1,
        W=1,
        D=4,
        atom_vocab_size=4,
        coeff_vocab_size=3,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    prior = Prior(cfg, bos_token_id=8, pad_token_id=7).eval()
    with torch.no_grad():
        for param in prior.parameters():
            param.zero_()

    toks = prior.generate(batch_size=16, show_progress=False)

    assert toks.shape == (16, 4)
    assert (toks[:, 0::2] < 4).all()
    assert ((toks[:, 1::2] >= 4) & (toks[:, 1::2] < 7)).all()
    assert not torch.isin(toks, torch.tensor([7, 8])).any()
