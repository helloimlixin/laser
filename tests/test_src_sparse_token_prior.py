import torch

from src.models.sparse_token_prior import (
    build_sparse_prior_from_cache,
    compute_quantized_rq_losses,
)
from src.models.spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig


def test_compute_quantized_rq_losses_keeps_atom_coeff_breakdown():
    per_token_ce = torch.tensor([[[1.0, 5.0, 3.0, 7.0]]], dtype=torch.float32)

    token_ce, atom_ce, coeff_ce, total = compute_quantized_rq_losses(
        per_token_ce,
        atom_loss_weight=2.0,
        coeff_loss_weight=0.25,
    )

    assert torch.isclose(token_ce, torch.tensor(4.0))
    assert torch.isclose(atom_ce, torch.tensor(2.0))
    assert torch.isclose(coeff_ce, torch.tensor(6.0))
    assert torch.isclose(total, torch.tensor((2.0 * 2.0 + 0.25 * 6.0) / 2.25))


def test_src_spatial_depth_prior_quantized_generation_preserves_unique_atoms():
    torch.manual_seed(0)

    cfg = SpatialDepthPriorConfig(
        vocab_size=5,
        atom_vocab_size=3,
        coeff_vocab_size=2,
        H=1,
        W=1,
        D=4,
        real_valued_coeffs=False,
        d_model=8,
        n_heads=2,
        n_spatial_layers=0,
        n_depth_layers=0,
        d_ff=16,
        dropout=0.0,
    )
    model = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    tokens = model.generate(batch_size=16, show_progress=False)

    assert tokens.shape == (16, 1, 4)
    atom_tokens = tokens[:, 0, 0::2]
    coeff_tokens = tokens[:, 0, 1::2]
    assert torch.all((atom_tokens >= 0) & (atom_tokens < cfg.atom_vocab_size))
    assert torch.all((coeff_tokens >= cfg.atom_vocab_size) & (coeff_tokens < cfg.vocab_size))
    for sample in atom_tokens:
        assert torch.unique(sample).numel() == sample.numel()


def test_build_sparse_prior_from_cache_uses_cache_shape_and_vocab_split():
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {"num_atoms": 3},
    }

    model = build_sparse_prior_from_cache(
        cache,
        architecture="spatial_depth",
        total_vocab_size=5,
        atom_vocab_size=None,
        coeff_vocab_size=None,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
        dropout=0.0,
        n_global_spatial_tokens=1,
    )

    assert model.cfg.H == 1
    assert model.cfg.W == 2
    assert model.cfg.D == 4
    assert model.atom_vocab_size == 3
    assert model.coeff_vocab_size == 2
