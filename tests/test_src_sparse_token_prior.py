import torch

from src.models.sparse_token_prior import (
    SparseTokenPriorModule,
    build_sparse_prior_from_cache,
    build_sparse_prior_from_hparams,
    compute_quantized_rq_losses,
    infer_sparse_vocab_sizes,
)
from src.models.spatial_prior import (
    SpatialDepthPrior,
    SpatialDepthPriorConfig,
    build_spatial_depth_prior_config,
)


class _FakeSpatialPriorBottleneck:
    content_vocab_size = 9
    num_embeddings = 6
    n_bins = 4

    def __init__(self, coef_max):
        self.coef_max = coef_max

    @staticmethod
    def _dequantize_coeff(bins: torch.Tensor) -> torch.Tensor:
        return bins.to(torch.float32) - 1.5


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
        "meta": {
            "num_atoms": 3,
            "coef_max": 7.5,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
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
    assert model.cfg.coeff_max == 7.5
    assert torch.equal(model.cfg.coeff_bin_values, torch.tensor([-1.0, 0.0]))


def test_infer_sparse_vocab_sizes_uses_coeff_bin_values_length_when_count_is_missing():
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    total_vocab_size, atom_vocab_size, coeff_vocab_size = infer_sparse_vocab_sizes(
        cache,
        total_vocab_size=None,
        atom_vocab_size=None,
        coeff_vocab_size=None,
    )

    assert total_vocab_size == 5
    assert atom_vocab_size == 3
    assert coeff_vocab_size == 2


def test_build_sparse_prior_from_hparams_uses_saved_depth_layer_count():
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    model = build_sparse_prior_from_hparams(
        cache,
        hparams={
            "prior_architecture": "spatial_depth",
            "prior_d_model": 8,
            "prior_n_heads": 2,
            "prior_n_spatial_layers": 3,
            "prior_n_depth_layers": 2,
            "prior_n_global_spatial_tokens": 1,
            "prior_d_ff": 16,
            "prior_dropout": 0.0,
            "prior_atom_vocab_size": 3,
            "prior_coeff_vocab_size": 2,
            "prior_coeff_max": 7.5,
        },
    )

    assert model.cfg.n_spatial_layers == 3
    assert model.cfg.n_depth_layers == 2
    assert model.atom_vocab_size == 3
    assert model.coeff_vocab_size == 2


def test_build_sparse_prior_from_cache_supports_real_valued_coeffs():
    cache = {
        "tokens_flat": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        "coeffs_flat": torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32),
        "shape": (1, 2, 2),
        "meta": {
            "num_atoms": 5,
            "coeff_max": 6.0,
            "variational_coeffs": True,
            "variational_coeff_prior_std": 0.35,
            "variational_coeff_min_std": 0.05,
        },
    }

    model = build_sparse_prior_from_cache(
        cache,
        architecture="spatial_depth",
        total_vocab_size=None,
        atom_vocab_size=None,
        coeff_vocab_size=None,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
        dropout=0.0,
        n_global_spatial_tokens=0,
        autoregressive_coeffs=False,
    )

    assert model.real_valued_coeffs is True
    assert model.gaussian_coeffs is True
    assert model.autoregressive_coeffs is False
    assert model.atom_vocab_size == 5
    assert model.cfg.vocab_size == 5
    assert model.cfg.coeff_max == 6.0
    assert model.cfg.coeff_prior_std == 0.35
    assert model.cfg.coeff_min_std == 0.05


def test_build_sparse_prior_from_hparams_supports_real_valued_checkpoint_rebuild():
    cache = {
        "tokens_flat": torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        "coeffs_flat": torch.tensor([[0.1, -0.2, 0.3, -0.4]], dtype=torch.float32),
        "shape": (1, 2, 2),
        "meta": {
            "num_atoms": 5,
        },
    }

    model = build_sparse_prior_from_hparams(
        cache,
        hparams={
            "prior_architecture": "spatial_depth",
            "prior_d_model": 8,
            "prior_n_heads": 2,
            "prior_n_spatial_layers": 3,
            "prior_n_depth_layers": 2,
            "prior_n_global_spatial_tokens": 1,
            "prior_d_ff": 16,
            "prior_dropout": 0.0,
            "prior_atom_vocab_size": 5,
            "prior_real_valued_coeffs": True,
            "prior_gaussian_coeffs": True,
            "prior_autoregressive_coeffs": False,
            "prior_coeff_max": 7.0,
            "prior_coeff_prior_std": 0.4,
            "prior_coeff_min_std": 0.02,
        },
    )

    assert model.real_valued_coeffs is True
    assert model.gaussian_coeffs is True
    assert model.autoregressive_coeffs is False
    assert model.cfg.n_spatial_layers == 3
    assert model.cfg.n_depth_layers == 2
    assert model.cfg.coeff_max == 7.0


def test_sparse_token_prior_module_saves_prior_architecture_metadata():
    cfg = SpatialDepthPriorConfig(
        vocab_size=5,
        atom_vocab_size=3,
        coeff_vocab_size=2,
        H=1,
        W=2,
        D=4,
        real_valued_coeffs=False,
        d_model=8,
        n_heads=2,
        n_spatial_layers=2,
        n_depth_layers=1,
        n_global_spatial_tokens=1,
        d_ff=16,
        dropout=0.0,
    )
    module = SparseTokenPriorModule(prior=SpatialDepthPrior(cfg))

    assert module.hparams["prior_architecture"] == "spatial_depth"
    assert module.hparams["prior_n_heads"] == 2
    assert module.hparams["prior_n_spatial_layers"] == 2
    assert module.hparams["prior_n_depth_layers"] == 1


def test_sparse_token_prior_module_generate_sparse_codes_returns_atoms_and_coeffs_for_real_valued_priors():
    torch.manual_seed(0)

    cfg = SpatialDepthPriorConfig(
        vocab_size=6,
        atom_vocab_size=6,
        coeff_vocab_size=None,
        H=1,
        W=2,
        D=2,
        real_valued_coeffs=True,
        d_model=8,
        n_heads=2,
        n_spatial_layers=0,
        n_depth_layers=0,
        d_ff=16,
        dropout=0.0,
        gaussian_coeffs=False,
    )
    module = SparseTokenPriorModule(prior=SpatialDepthPrior(cfg)).eval()
    with torch.no_grad():
        for param in module.parameters():
            param.zero_()

    atom_ids, coeffs = module.generate_sparse_codes(batch_size=3, temperature=1.0)

    assert atom_ids.shape == (3, 2, 2)
    assert coeffs.shape == (3, 2, 2)
    assert torch.all((atom_ids >= 0) & (atom_ids < cfg.atom_vocab_size))


def test_build_spatial_depth_prior_config_prefers_bottleneck_coef_max():
    cfg = build_spatial_depth_prior_config(
        _FakeSpatialPriorBottleneck(coef_max=1.25),
        H=2,
        W=3,
        D=4,
        d_model=16,
        n_heads=2,
        n_spatial_layers=2,
        n_depth_layers=1,
        d_ff=32,
        dropout=0.0,
        n_global_spatial_tokens=0,
        real_valued_coeffs=False,
        coeff_max_fallback=7.0,
    )

    assert cfg.coeff_max == 1.25
    assert cfg.coeff_vocab_size == 4
    assert torch.equal(
        cfg.coeff_bin_values,
        torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32),
    )


def test_build_spatial_depth_prior_config_uses_fallback_when_bottleneck_coef_max_is_none():
    cfg = build_spatial_depth_prior_config(
        _FakeSpatialPriorBottleneck(coef_max=None),
        H=1,
        W=1,
        D=2,
        d_model=8,
        n_heads=2,
        n_spatial_layers=1,
        n_depth_layers=1,
        d_ff=16,
        dropout=0.0,
        n_global_spatial_tokens=0,
        real_valued_coeffs=True,
        coeff_max_fallback=9.5,
    )

    assert cfg.coeff_max == 9.5
    assert cfg.coeff_vocab_size is None
    assert cfg.coeff_bin_values is None
