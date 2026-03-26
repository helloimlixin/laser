import inspect
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import laser as laser_module


class _DummyLPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, recon, target):
        self.batch_sizes.append(recon.shape[0])
        return (recon - target).square().mean(dim=(1, 2, 3), keepdim=True)


def _build_model(**overrides):
    params = {
        "in_channels": 3,
        "num_hiddens": 16,
        "num_embeddings": 8,
        "embedding_dim": 4,
        "sparsity_level": 1,
        "num_residual_blocks": 1,
        "num_residual_hiddens": 8,
        "commitment_cost": 0.25,
        "learning_rate": 1e-3,
        "beta": 0.9,
        "perceptual_weight": 0.0,
        "compute_fid": False,
        "coherence_weight": 0.0,
        "log_images_every_n_steps": 0,
        "diag_log_interval": 0,
    }
    params.update(overrides)
    model = laser_module.LASER(**params)
    model.log = lambda *args, **kwargs: None
    return model


def test_train_lpips_uses_full_batch(monkeypatch):
    monkeypatch.setattr(laser_module, "LPIPS", _DummyLPIPS)
    model = _build_model(perceptual_weight=1.0)
    batch = torch.randn(4, 3, 16, 16)

    loss, recon, x = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == x.shape == batch.shape
    assert model.lpips.batch_sizes == [4]


def test_validation_visual_cache_respects_flag():
    batch = torch.randn(4, 3, 16, 16)
    disabled = _build_model(enable_val_latent_visuals=False)
    enabled = _build_model(enable_val_latent_visuals=True)
    trainer_stub = type("TrainerStub", (), {"is_global_zero": True})()
    disabled.__dict__["_trainer"] = trainer_stub
    enabled.__dict__["_trainer"] = trainer_stub

    disabled._maybe_store_val_batch(batch)
    enabled._maybe_store_val_batch(batch)

    assert disabled._val_vis_batch is None
    assert enabled._val_vis_batch is not None


def test_train_psnr_is_logged_to_progress_bar():
    model = _build_model()
    batch = torch.randn(4, 3, 16, 16)
    log_calls = []

    def _record_log(name, value, **kwargs):
        log_calls.append((name, kwargs))

    model.log = _record_log

    loss, recon, x = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == x.shape == batch.shape
    assert any(
        name == "train/psnr" and kwargs.get("prog_bar") is True
        for name, kwargs in log_calls
    )


def test_logged_sparsity_uses_fixed_support_budget_not_thresholded_coeffs():
    model = _build_model(num_embeddings=8, sparsity_level=2)
    batch = torch.randn(2, 3, 16, 16)
    sparse_codes = laser_module.SparseCodes(
        support=torch.tensor(
            [
                [[[0, 1]]],
                [[[2, 3]]],
            ],
            dtype=torch.long,
        ),
        values=torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.0, 0.0]]],
            ],
            dtype=torch.float32,
        ),
        num_embeddings=8,
    )
    logged = {}

    model.forward = lambda x: (x, x.new_zeros(()), sparse_codes)
    model.log = lambda name, value, **kwargs: logged.setdefault(name, value)

    loss, recon, x = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == x.shape == batch.shape
    assert float(logged["train/sparsity"]) == 0.25
    assert float(logged["train/effective_sparsity"]) == 0.0625


def test_dictionary_is_always_in_optimizer_with_optional_lr_override():
    model = _build_model(dict_learning_rate=5e-4)

    optimizer = model.configure_optimizers()
    dict_group = next(
        group
        for group in optimizer.param_groups
        if any(param is model.bottleneck.dictionary for param in group["params"])
    )

    assert model.bottleneck.dictionary.requires_grad is True
    assert dict_group["lr"] == 5e-4


def test_laser_patch_dictionary_learning_runs_end_to_end():
    model = _build_model(patch_based=True, patch_size=2, patch_stride=1)
    batch = torch.randn(2, 3, 16, 16)

    loss, recon, x = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == x.shape == batch.shape


def test_laser_patch_toggle_can_fall_back_to_per_site_dictionary_learning():
    model = _build_model(patch_based=False, patch_size=8, patch_stride=4)
    batch = torch.randn(2, 3, 16, 16)

    loss, recon, x = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == x.shape == batch.shape


def test_laser_decode_from_tokens_forwards_quantized_decode_args():
    model = _build_model()
    calls = {}

    def _tokens_to_latent(tokens, **kwargs):
        calls["tokens"] = tokens
        calls["kwargs"] = kwargs
        return torch.zeros(tokens.size(0), model.hparams.embedding_dim, 4, 4)

    model.bottleneck.tokens_to_latent = _tokens_to_latent
    model.decode = lambda z_q: z_q + 1.0

    tokens = torch.zeros(2, 3, 3, 2, dtype=torch.long)
    images = model.decode_from_tokens(
        tokens,
        latent_hw=(4, 4),
        atom_vocab_size=8,
        coeff_vocab_size=5,
        coeff_bin_values=torch.linspace(-1.0, 1.0, steps=5),
        coeff_max=2.0,
        coeff_quantization="uniform",
        coeff_mu=0.0,
    )

    assert images.shape == (2, model.hparams.embedding_dim, 4, 4)
    assert torch.allclose(images, torch.ones_like(images))
    assert calls["tokens"] is tokens
    assert calls["kwargs"]["latent_hw"] == (4, 4)
    assert calls["kwargs"]["atom_vocab_size"] == 8
    assert calls["kwargs"]["coeff_vocab_size"] == 5
    assert float(calls["kwargs"]["coeff_max"]) == 2.0
    assert calls["kwargs"]["coeff_quantization"] == "uniform"


def test_laser_encode_to_tokens_quantizes_sparse_codes():
    model = _build_model(num_embeddings=8, sparsity_level=2)
    sparse_codes = laser_module.SparseCodes(
        support=torch.tensor([[[[1, 4]]]], dtype=torch.long),
        values=torch.tensor([[[[-1.0, 0.5]]]], dtype=torch.float32),
        num_embeddings=8,
    )

    model.encode = lambda x: (torch.zeros(x.size(0), 4, 3, 5), torch.zeros(()), sparse_codes)

    tokens, latent_hw = model.encode_to_tokens(
        torch.randn(1, 3, 16, 16),
        coeff_vocab_size=5,
        coeff_max=1.0,
    )

    assert latent_hw == (3, 5)
    assert tokens.shape == (1, 1, 1, 4)
    assert tokens.tolist() == [[[[1, 8, 4, 11]]]]


def test_laser_init_no_longer_exposes_removed_sparse_coding_knobs():
    params = inspect.signature(laser_module.LASER.__init__).parameters

    assert "perceptual_batch_size" not in params
    assert "use_online_learning" not in params
    assert "use_backprop_only" not in params
    assert "sparse_solver" not in params
    assert "iht_iterations" not in params
    assert "iht_step_size" not in params
    assert "lista_layers" not in params
    assert "lista_tied_weights" not in params
    assert "lista_initial_threshold" not in params
    assert "multi_res_dct_weight" not in params
    assert "multi_res_dct_levels" not in params
    assert "multi_res_grad_weight" not in params
    assert "multi_res_grad_levels" not in params
    assert "patch_flatten_order" not in params
    assert "per_pixel_sparse_coding" not in params
    assert "fista_alpha" not in params
    assert "fista_tolerance" not in params
    assert "fista_max_steps" not in params
    assert "use_pattern_quantizer" not in params
    assert "num_patterns" not in params
    assert "pattern_commitment_cost" not in params
    assert "pattern_ema_decay" not in params
    assert "pattern_temperature" not in params
    assert "orthogonality_weight" not in params
    assert "dictionary_update_mode" not in params
    assert "dict_ema_decay" not in params
    assert "dict_ema_eps" not in params
    assert "fast_omp" not in params
    assert "omp_diag_eps" not in params
    assert "omp_cholesky_eps" not in params
    assert "sparse_coding_scheme" not in params
    assert "lista_steps" not in params
    assert "lista_step_size_init" not in params
    assert "lista_threshold_init" not in params
