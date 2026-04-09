import torch

import src.models.sparse_token_prior as sparse_token_prior_module
from src.models.gpt_prior import GPTPrior
from src.models.mingpt_prior import MinGPTQuantizedPriorConfig
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
from src.models.transformer_core import CausalSelfAttention


class _FakeSpatialPriorBottleneck:
    content_vocab_size = 9
    num_embeddings = 6
    n_bins = 4

    def __init__(self, coef_max):
        self.coef_max = coef_max

    @staticmethod
    def _dequantize_coeff(bins: torch.Tensor) -> torch.Tensor:
        return bins.to(torch.float32) - 1.5


class _FakeRealPrior(torch.nn.Module):
    real_valued_coeffs = True
    gaussian_coeffs = False

    def __init__(self):
        super().__init__()
        self.cfg = type("Cfg", (), {"H": 1, "W": 1, "D": 2, "vocab_size": 3})()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tok_grid, coeff_grid, mask_tokens=None, return_features=False):
        bsz, steps, depth = tok_grid.shape
        logits = torch.zeros(
            bsz,
            steps,
            depth,
            int(self.cfg.vocab_size),
            device=tok_grid.device,
            dtype=torch.float32,
        )
        coeff_pred = torch.zeros_like(coeff_grid) + self.anchor
        if return_features:
            feats = torch.zeros(bsz, steps, depth, 1, device=tok_grid.device, dtype=torch.float32)
            return logits, coeff_pred, feats
        return logits, coeff_pred


class _FakeQuantPrior(torch.nn.Module):
    real_valued_coeffs = False

    def __init__(self):
        super().__init__()
        self.cfg = type(
            "Cfg",
            (),
            {
                "H": 1,
                "W": 1,
                "D": 2,
                "vocab_size": 4,
            },
        )()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, tok_grid):
        bsz, steps, depth = tok_grid.shape
        logits = torch.zeros(
            bsz,
            steps,
            depth,
            int(self.cfg.vocab_size),
            device=tok_grid.device,
            dtype=torch.float32,
        )
        logits[..., 0] = 1.0 + self.anchor
        return logits


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


def test_real_valued_shared_step_applies_atom_loss_weight():
    batch = (
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([[1.0, -1.0]], dtype=torch.float32),
    )

    mod_a = SparseTokenPriorModule(
        prior=_FakeRealPrior(),
        atom_loss_weight=1.0,
        coeff_loss_weight=1.0,
        coeff_loss_type="mse",
    )
    mod_b = SparseTokenPriorModule(
        prior=_FakeRealPrior(),
        atom_loss_weight=3.0,
        coeff_loss_weight=1.0,
        coeff_loss_type="mse",
    )
    mod_a.log = lambda *args, **kwargs: None
    mod_b.log = lambda *args, **kwargs: None

    loss_a = mod_a._shared_step(batch, "train")
    loss_b = mod_b._shared_step(batch, "train")
    ce = torch.log(torch.tensor(3.0))
    coeff_mse = torch.tensor(1.0)

    assert torch.isclose(loss_a, ce + coeff_mse)
    assert torch.isclose(loss_b, 3.0 * ce + coeff_mse)


def test_real_valued_shared_step_does_not_log_duplicate_atom_ce_metric():
    batch = (
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([[1.0, -1.0]], dtype=torch.float32),
    )
    names = []

    mod = SparseTokenPriorModule(
        prior=_FakeRealPrior(),
        atom_loss_weight=1.0,
        coeff_loss_weight=1.0,
        coeff_loss_type="mse",
    )
    mod.log = lambda name, *args, **kwargs: names.append(name)

    mod._shared_step(batch, "train")

    assert "train/ce_loss" in names
    assert "train/atom_ce_loss" not in names


def test_real_valued_shared_step_logs_train_metrics_step_only():
    batch = (
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([[1.0, -1.0]], dtype=torch.float32),
    )
    calls = []

    mod = SparseTokenPriorModule(
        prior=_FakeRealPrior(),
        atom_loss_weight=1.0,
        coeff_loss_weight=1.0,
        coeff_loss_type="mse",
    )
    mod.log = lambda name, value, **kwargs: calls.append((name, kwargs))

    mod._shared_step(batch, "train")

    assert any(
        name == "train/loss"
        and kwargs.get("on_step") is True
        and kwargs.get("on_epoch") is False
        for name, kwargs in calls
    )


def test_sparse_token_prior_module_manual_lr_schedule_applies_first_step_without_scheduler():
    mod = SparseTokenPriorModule(
        prior=_FakeQuantPrior(),
        learning_rate=1e-3,
        warmup_steps=10,
        min_lr_ratio=0.01,
    )
    mod.__dict__["_trainer"] = type("TrainerStub", (), {"estimated_stepping_batches": 100})()

    optimizer = mod.configure_optimizers()
    mod._apply_scheduled_lrs(optimizer, step=0)

    assert optimizer.param_groups[0]["lr"] == 1e-4


def test_log_recon_images_uses_global_step_for_wandb_log():
    calls = []

    class _Experiment:
        def log(self, payload, **kwargs):
            calls.append((payload, kwargs))

    trainer_stub = type(
        "TrainerStub",
        (),
        {
            "is_global_zero": True,
            "global_step": 11,
            "logger": type("LoggerStub", (), {"experiment": _Experiment()})(),
        },
    )()

    mod = SparseTokenPriorModule(
        prior=_FakeQuantPrior(),
        stage1_decoder_bundle=object(),
    )
    mod.__dict__["_trainer"] = trainer_stub
    mod._stage1_decoder_bundle = object()

    old_decode = sparse_token_prior_module.decode_stage2_outputs
    sparse_token_prior_module.decode_stage2_outputs = lambda *args, **kwargs: torch.zeros(1, 3, 4, 4)
    try:
        mod._log_recon_images((torch.zeros(1, 2, dtype=torch.long),))
    finally:
        sparse_token_prior_module.decode_stage2_outputs = old_decode

    assert len(calls) == 1
    assert calls[0][1]["step"] == 11


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


def test_build_sparse_prior_from_cache_supports_gpt_aliases():
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    for arch in ("gpt", "mingpt"):
        model = build_sparse_prior_from_cache(
            cache,
            architecture=arch,
            total_vocab_size=5,
            atom_vocab_size=None,
            coeff_vocab_size=None,
            d_model=8,
            n_heads=2,
            n_layers=2,
            d_ff=16,
            dropout=0.0,
            n_global_spatial_tokens=0,
        )

        assert isinstance(model, GPTPrior)
        assert model.cfg.H == 1
        assert model.cfg.W == 2
        assert model.cfg.D == 4
        assert model.atom_vocab_size == 3
        assert model.coeff_vocab_size == 2


def test_build_sparse_prior_from_cache_threads_gpt_window_sites():
    cache = {
        "tokens_flat": torch.zeros(4, 16, dtype=torch.int32),
        "shape": (2, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    model = build_sparse_prior_from_cache(
        cache,
        architecture="gpt",
        total_vocab_size=5,
        atom_vocab_size=None,
        coeff_vocab_size=None,
        window_sites=2,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
        dropout=0.0,
        n_global_spatial_tokens=0,
    )

    assert isinstance(model, GPTPrior)
    assert model.cfg.window_sites == 2
    assert model.window_tokens == 8


def test_build_sparse_prior_from_cache_preserves_gpt_global_prefix_tokens():
    cache = {
        "tokens_flat": torch.zeros(4, 16, dtype=torch.int32),
        "shape": (2, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    model = build_sparse_prior_from_cache(
        cache,
        architecture="gpt",
        total_vocab_size=5,
        atom_vocab_size=None,
        coeff_vocab_size=None,
        window_sites=2,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=16,
        dropout=0.0,
        n_global_spatial_tokens=3,
    )

    assert isinstance(model, GPTPrior)
    assert model.cfg.n_global_spatial_tokens == 3
    assert model.global_spatial_tokens.shape == (1, 3, 8)


def test_build_sparse_prior_from_hparams_supports_gpt_checkpoint_rebuild():
    cache = {
        "tokens_flat": torch.zeros(4, 8, dtype=torch.int32),
        "shape": (1, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    for arch in ("gpt", "mingpt"):
        model = build_sparse_prior_from_hparams(
            cache,
            hparams={
                "prior_architecture": arch,
                "prior_d_model": 8,
                "prior_n_heads": 2,
                "prior_n_layers": 3,
                "prior_d_ff": 16,
                "prior_dropout": 0.0,
                "prior_atom_vocab_size": 3,
                "prior_coeff_vocab_size": 2,
            },
        )

        assert isinstance(model, GPTPrior)
        assert model.cfg.n_layers == 3
        assert model.atom_vocab_size == 3
        assert model.coeff_vocab_size == 2


def test_build_sparse_prior_from_hparams_preserves_gpt_window_sites():
    cache = {
        "tokens_flat": torch.zeros(4, 16, dtype=torch.int32),
        "shape": (2, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    model = build_sparse_prior_from_hparams(
        cache,
        hparams={
            "prior_architecture": "gpt",
            "prior_d_model": 8,
            "prior_n_heads": 2,
            "prior_n_layers": 3,
            "prior_d_ff": 16,
            "prior_dropout": 0.0,
            "prior_atom_vocab_size": 3,
            "prior_coeff_vocab_size": 2,
            "prior_window_sites": 2,
        },
    )

    assert isinstance(model, GPTPrior)
    assert model.cfg.window_sites == 2


def test_build_sparse_prior_from_hparams_preserves_gpt_global_prefix_tokens():
    cache = {
        "tokens_flat": torch.zeros(4, 16, dtype=torch.int32),
        "shape": (2, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    model = build_sparse_prior_from_hparams(
        cache,
        hparams={
            "prior_architecture": "gpt",
            "prior_d_model": 8,
            "prior_n_heads": 2,
            "prior_n_layers": 3,
            "prior_d_ff": 16,
            "prior_dropout": 0.0,
            "prior_atom_vocab_size": 3,
            "prior_coeff_vocab_size": 2,
            "prior_window_sites": 2,
            "prior_n_global_spatial_tokens": 3,
        },
    )

    assert isinstance(model, GPTPrior)
    assert model.cfg.window_sites == 2
    assert model.cfg.n_global_spatial_tokens == 3


def test_build_sparse_prior_from_hparams_prefers_saved_grid_shape():
    cache = {
        "tokens_flat": torch.zeros(4, 24, dtype=torch.int32),
        "shape": (3, 2, 4),
        "meta": {
            "num_atoms": 3,
            "coeff_bin_values": torch.tensor([-1.0, 0.0], dtype=torch.float32),
        },
    }

    model = build_sparse_prior_from_hparams(
        cache,
        hparams={
            "prior_architecture": "gpt",
            "prior_H": 2,
            "prior_W": 2,
            "prior_D": 4,
            "prior_d_model": 8,
            "prior_n_heads": 2,
            "prior_n_layers": 3,
            "prior_d_ff": 16,
            "prior_dropout": 0.0,
            "prior_atom_vocab_size": 3,
            "prior_coeff_vocab_size": 2,
        },
    )

    assert isinstance(model, GPTPrior)
    assert model.cfg.H == 2
    assert model.cfg.W == 2
    assert model.cfg.D == 4
    assert model.window_tokens == 8


def test_gpt_masked_logits_split_atom_and_coeff_vocab():
    model = GPTPrior(
        MinGPTQuantizedPriorConfig(
            vocab_size=7,
            H=1,
            W=1,
            D=4,
            atom_vocab_size=5,
            coeff_vocab_size=2,
            d_model=8,
            n_heads=2,
            n_layers=0,
            d_ff=16,
            dropout=0.0,
        )
    )
    logits = torch.zeros(1, 4, 7)

    masked = model._masked_logits(logits, start_pos=0)

    assert torch.isneginf(masked[0, 0, 5:]).all()
    assert torch.isneginf(masked[0, 1, :5]).all()
    assert torch.isneginf(masked[0, 2, 5:]).all()
    assert torch.isneginf(masked[0, 3, :5]).all()


def test_gpt_generation_blocks_duplicate_atoms_within_site():
    model = GPTPrior(
        MinGPTQuantizedPriorConfig(
            vocab_size=5,
            H=1,
            W=1,
            D=4,
            atom_vocab_size=3,
            coeff_vocab_size=2,
            d_model=8,
            n_heads=2,
            n_layers=0,
            d_ff=16,
            dropout=0.0,
        )
    ).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    model._sample_from_logits = lambda logits, **kwargs: logits.argmax(dim=-1)

    tokens = model.generate(batch_size=1, show_progress=False)

    assert torch.equal(tokens[0, 0], torch.tensor([0, 3, 1, 3], dtype=torch.long))


def test_windowed_attention_mask_is_site_local_and_causal():
    attn = CausalSelfAttention(d_model=4, n_heads=1, dropout=0.0, window_size=4)
    mask = attn._windowed_attention_mask(
        q_start=4,
        q_end=8,
        k_start=1,
        k_end=8,
        window_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[0, 0]

    allowed = mask.eq(0.0)
    assert torch.equal(allowed[0], torch.tensor([True, True, True, True, False, False, False]))
    assert torch.equal(allowed[-1], torch.tensor([False, False, False, True, True, True, True]))


def test_windowed_attention_mask_keeps_global_prefix_tokens_visible():
    attn = CausalSelfAttention(
        d_model=4,
        n_heads=1,
        dropout=0.0,
        window_size=4,
        n_global_prefix_tokens=2,
    )
    mask = attn._windowed_attention_mask(
        q_start=6,
        q_end=8,
        k_start=0,
        k_end=8,
        window_size=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        n_global_prefix_tokens=2,
    )[0, 0]

    allowed = mask.eq(0.0)
    assert torch.equal(allowed[0], torch.tensor([True, True, False, True, True, True, True, False]))
    assert torch.equal(allowed[1], torch.tensor([True, True, False, False, True, True, True, True]))


def test_windowed_attention_trims_cached_kv_state_during_decode():
    attn = CausalSelfAttention(d_model=4, n_heads=1, dropout=0.0, window_size=3).eval()
    kv_cache = None
    seen = []

    for _ in range(6):
        _, kv_cache = attn(torch.zeros(1, 1, 4), kv_cache=kv_cache)
        seen.append(int(kv_cache[0].shape[2]))

    assert seen == [1, 2, 3, 3, 3, 3]


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
