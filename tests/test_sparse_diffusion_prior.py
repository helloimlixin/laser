import pytest
import torch

from src.models.sparse_diffusion_prior import (
    SparseCoeffDiffusionModule,
    build_sparse_coeff_diffusion_prior_from_cache,
    infer_sparse_coeff_diffusion_cache_metadata,
)


def _real_cache(num_items=5, h=2, w=3, d=2, atom_vocab=7):
    generator = torch.Generator().manual_seed(123)
    tokens = torch.randint(0, atom_vocab, (num_items, h * w * d), generator=generator)
    coeffs = torch.randn(num_items, h * w * d, generator=generator)
    return {
        "tokens_flat": tokens,
        "coeffs_flat": coeffs,
        "shape": (h, w, d),
        "meta": {"num_atoms": atom_vocab, "quantize_sparse_coeffs": False},
    }


def test_sparse_coeff_diffusion_requires_real_valued_cache():
    cache = {
        "tokens_flat": torch.zeros(2, 4, dtype=torch.long),
        "shape": (1, 2, 2),
        "meta": {"num_atoms": 3},
    }

    with pytest.raises(ValueError, match="coeffs_flat"):
        infer_sparse_coeff_diffusion_cache_metadata(cache)


def test_sparse_coeff_diffusion_training_step_is_finite():
    torch.manual_seed(0)
    cache = _real_cache(num_items=4)
    prior = build_sparse_coeff_diffusion_prior_from_cache(
        cache,
        hidden_channels=16,
        atom_embed_dim=4,
        time_embed_dim=16,
        n_res_blocks=2,
        num_timesteps=8,
        support_bank=cache["tokens_flat"][:3],
    )
    module = SparseCoeffDiffusionModule(prior, learning_rate=1.0e-3)

    batch = (cache["tokens_flat"][:2], cache["coeffs_flat"][:2])
    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_sparse_coeff_diffusion_generate_sparse_codes_from_support_bank():
    torch.manual_seed(1)
    cache = _real_cache(num_items=6)
    prior = build_sparse_coeff_diffusion_prior_from_cache(
        cache,
        hidden_channels=16,
        atom_embed_dim=4,
        time_embed_dim=16,
        n_res_blocks=1,
        num_timesteps=6,
        support_bank=cache["tokens_flat"][:4],
    )
    module = SparseCoeffDiffusionModule(prior)

    atoms, coeffs = module.generate_sparse_codes(batch_size=3, steps=3)

    assert atoms.shape == (3, 6, 2)
    assert coeffs.shape == (3, 6, 2)
    assert atoms.dtype == torch.long
    assert int(atoms.min().item()) >= 0
    assert int(atoms.max().item()) < 7
    assert torch.isfinite(coeffs).all()
