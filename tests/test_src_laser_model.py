import inspect

import torch


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
        "num_hiddens": 32,
        "num_embeddings": 8,
        "embedding_dim": 4,
        "sparsity_level": 1,
        "num_residual_blocks": 1,
        "num_residual_hiddens": 8,
        # Attention U-Net backbone params (tiny config for 16x16 test inputs).
        "resolution": 16,
        "num_downsamples": 1,
        "channel_multipliers": (1, 1),
        "commitment_cost": 0.25,
        "learning_rate": 1e-3,
        "beta": 0.9,
        "perceptual_weight": 0.0,
        "compute_fid": False,
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


def test_lpips_range_uses_datamodule_normalization_stats():
    model = _build_model()
    cfg = type("Cfg", (), {"mean": (0.25, 0.5, 0.75), "std": (0.25, 0.5, 0.125)})()
    datamodule = type("DataModule", (), {"config": cfg})()
    model.__dict__["_trainer"] = type("TrainerStub", (), {"datamodule": datamodule})()

    unit = torch.tensor([[[[0.0, 1.0]], [[0.25, 0.75]], [[0.5, 1.0]]]])
    mean = torch.tensor(cfg.mean).view(1, 3, 1, 1)
    std = torch.tensor(cfg.std).view(1, 3, 1, 1)
    normalized = (unit - mean) / std

    lpips_input = model._image_to_lpips_range(normalized)

    assert torch.allclose(lpips_input, unit * 2.0 - 1.0)


def test_laser_adversarial_training_adds_discriminator_optimizer():
    model = _build_model(
        adversarial_weight=0.05,
        discriminator_channels=8,
        discriminator_layers=1,
    )

    assert model.automatic_optimization is False
    assert model.discriminator is not None
    assert len(model.configure_optimizers()) == 2

    batch = torch.randn(2, 3, 16, 16)
    loss, recon, x = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == x.shape == batch.shape
    assert model._effective_adversarial_weight("train") == 0.05


def test_laser_adversarial_quality_gate_waits_for_reconstruction_mse():
    model = _build_model(
        adversarial_weight=0.05,
        adversarial_start_recon_mse=0.01,
        adversarial_quality_ema_decay=0.0,
        discriminator_channels=8,
        discriminator_layers=1,
    )
    batch = torch.zeros(2, 3, 16, 16)
    sparse_codes = laser_module.SparseCodes(
        support=torch.zeros(2, 1, 1, 1, dtype=torch.long),
        values=torch.ones(2, 1, 1, 1),
        num_embeddings=model.hparams.num_embeddings,
    )

    assert model._effective_adversarial_weight("train") == 0.0
    model.forward = lambda x: (x + 0.2, x.new_zeros(()), sparse_codes)
    loss, recon, target = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == target.shape == batch.shape
    assert model._effective_adversarial_weight("train") == 0.0

    model.forward = lambda x: (x, x.new_zeros(()), sparse_codes)
    loss, recon, target = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == target.shape == batch.shape
    assert model._effective_adversarial_weight("train") == 0.05


def test_laser_adversarial_warmup_uses_configured_steps():
    model = _build_model(
        adversarial_weight=0.05,
        adversarial_start_step=10,
        adversarial_warmup_steps=10,
        discriminator_channels=8,
        discriminator_layers=1,
    )

    assert model._effective_adversarial_weight_at_step("train", 9) == 0.0
    assert model._effective_adversarial_weight_at_step("train", 10) == 0.0
    assert abs(model._effective_adversarial_weight_at_step("train", 15) - 0.025) < 1.0e-8
    assert abs(model._effective_adversarial_weight_at_step("train", 20) - 0.05) < 1.0e-8


def test_manual_optimizer_clipping_uses_model_side_clip_value():
    model = _build_model(
        adversarial_weight=0.05,
        discriminator_channels=8,
        discriminator_layers=1,
    )
    param = next(model.parameters())
    for model_param in model.parameters():
        model_param.grad = None
    param.grad = torch.full_like(param, 10.0)
    optimizer = torch.optim.Adam([param], lr=1e-3)
    model.manual_gradient_clip_val = 0.75
    model.__dict__["_trainer"] = type("TrainerStub", (), {"gradient_clip_val": 0.0})()

    model._clip_manual_optimizer(optimizer)

    assert param.grad.norm() <= 0.7501


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


def test_validation_step_skips_duplicate_recon_grid_for_cached_latent_visuals():
    # With latent visuals enabled, validation_step skips the simple recon grid
    # (the richer grid is logged at validation epoch end instead). With them
    # disabled, validation_step logs the grid once at batch 0.
    def _run(enable):
        model = _build_model(enable_val_latent_visuals=enable, log_images_every_n_steps=1)
        model.__dict__["_trainer"] = type("TrainerStub", (), {"is_global_zero": True})()
        batch = torch.randn(4, 3, 16, 16)
        calls = []
        model.compute_metrics = lambda batch, prefix="val": (
            torch.tensor(0.0),
            batch if isinstance(batch, torch.Tensor) else batch[0],
            batch if isinstance(batch, torch.Tensor) else batch[0],
        )
        model.log_images = lambda *args, **kwargs: calls.append(kwargs.get("prefix", "val"))
        model.validation_step(batch, 0)
        model.validation_step(batch, 1)
        return calls

    assert _run(enable=True) == []        # skipped for epoch-end latent visuals
    assert _run(enable=False) == ["val"]  # logged once at batch 0


def test_log_images_uses_global_step_for_wandb_log(recording_wandb_trainer):
    model = _build_model(log_images_every_n_steps=1)
    trainer_stub, calls = recording_wandb_trainer(global_step=7)
    model.__dict__["_trainer"] = trainer_stub

    x = torch.rand(2, 3, 8, 8)
    recon = torch.rand(2, 3, 8, 8)
    model.log_images(x, recon, prefix="train")

    # The wandb_media helpers fold the step into the payload as
    # "trainer/global_step" (rather than an experiment.log(step=...) kwarg).
    assert calls, "log_images should emit at least one wandb log"
    assert all(payload["trainer/global_step"] == 7 for payload, _ in calls)


def test_log_images_is_idempotent_per_prefix_and_step(recording_wandb_trainer):
    model = _build_model(log_images_every_n_steps=1)
    trainer_stub, calls = recording_wandb_trainer(global_step=11)
    model.__dict__["_trainer"] = trainer_stub

    x = torch.rand(2, 3, 8, 8)
    recon = torch.rand(2, 3, 8, 8)
    model.log_images(x, recon, prefix="train")
    model.log_images(x, recon, prefix="train")
    model.log_images(x, recon, prefix="val")

    # The same (prefix, step) logs its media exactly once: the repeated "train"
    # call is deduplicated by _claim_media_log, so each grid key appears once.
    image_keys = [key for payload, _ in calls for key in payload if key.endswith("/images")]
    assert image_keys.count("train/images") == 1
    assert image_keys.count("val/images") == 1
    assert all(payload["trainer/global_step"] == 11 for payload, _ in calls)


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
    assert any(
        name == "train/psnr"
        and kwargs.get("on_step") is True
        and kwargs.get("on_epoch") is False
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
    assert not any("/diag/" in name for name in logged)
    removed_metrics = {
        "train/input_mean",
        "train/input_std",
        "train/recon_mean",
        "train/recon_std",
        "train/dict_norm_max",
        "train/dict_norm_min",
        "train/dict_norm_mean",
        "train/coeff_abs_max",
        "train/coeff_abs_mean",
        "train/coeff_active_abs_mean",
        "train/coeff_clip_frac",
        "train/sparsity_reg_loss",
        "train/weighted_sparsity_reg_loss",
        "train/bottleneck_objective",
        "train/e_latent_loss",
        "train/dl_latent_loss",
    }
    assert not (removed_metrics & set(logged))




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


def test_laser_manual_lr_schedule_applies_first_step_without_scheduler():
    model = _build_model(learning_rate=1e-3, warmup_steps=10, min_lr_ratio=0.01)
    model.__dict__["_trainer"] = type("TrainerStub", (), {"estimated_stepping_batches": 100})()

    optimizer = model.configure_optimizers()
    model._apply_scheduled_lrs(optimizer, step=0)

    assert model._lr_total_steps == 100
    assert optimizer.param_groups[0]["lr"] == 1e-5


def test_laser_lr_schedule_does_not_force_lightning_step_estimate_property():
    model = _build_model(learning_rate=1e-3, warmup_steps=10, min_lr_ratio=0.01)

    class TrainerStub:
        max_steps = -1
        max_epochs = 5
        num_training_batches = float("inf")

        @property
        def estimated_stepping_batches(self):
            raise AssertionError("estimated_stepping_batches should not be forced")

    model.__dict__["_trainer"] = TrainerStub()

    optimizer = model.configure_optimizers()
    model._apply_scheduled_lrs(optimizer, step=0)

    assert model._lr_total_steps == 1
    assert optimizer.param_groups[0]["lr"] == 1e-3


def test_laser_lr_schedule_ignores_infinite_batch_estimate():
    model = _build_model()
    trainer = type(
        "TrainerStub",
        (),
        {
            "estimated_stepping_batches": None,
            "max_steps": -1,
            "max_epochs": 2,
            "num_training_batches": float("inf"),
        },
    )()

    total_steps, source = model._resolve_lr_total_steps(trainer)

    assert total_steps == 1
    assert source == "fallback"


def test_laser_patch_dictionary_learning_runs_end_to_end():
    model = _build_model(patch_based=True, patch_size=2, patch_stride=2)
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


def test_laser_ddpm_backbone_runs_end_to_end():
    model = _build_model(
        backbone="ddpm",
        resolution=16,
        num_hiddens=32,
        num_residual_blocks=1,
        num_downsamples=1,
        channel_multipliers=(1, 1),
        use_mid_attention=True,
        patch_based=False,
    )
    batch = torch.randn(2, 3, 16, 16)

    loss, recon, x = model.compute_metrics(batch, prefix="train")

    assert torch.isfinite(loss)
    assert recon.shape == x.shape == batch.shape
    assert isinstance(model.pre_bottleneck, torch.nn.Identity)
    assert isinstance(model.post_bottleneck, torch.nn.Identity)
    assert model.infer_latent_hw((16, 16)) == (8, 8)


def test_laser_unet_alias_selects_ddpm_backbone():
    model = _build_model(
        backbone="unet",
        resolution=16,
        num_hiddens=32,
        num_residual_blocks=1,
        num_downsamples=1,
        channel_multipliers=(1, 1),
        patch_based=False,
    )

    assert model.backbone == "ddpm"


def test_laser_decode_from_tokens_forwards_quantized_decode_args():
    model = _build_model()
    calls = {}

    def _reconstruct(atom_ids, coeffs, *, latent_hw=None):
        calls["atom_ids"] = atom_ids
        calls["coeffs"] = coeffs
        calls["latent_hw"] = latent_hw
        return torch.zeros(atom_ids.size(0), model.hparams.embedding_dim, 4, 4)

    model.reconstruct_latent_from_atoms_and_coeffs = _reconstruct
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
    assert calls["latent_hw"] == (4, 4)
    assert calls["atom_ids"].shape == (2, 3, 3, 1)
    assert calls["coeffs"].shape == (2, 3, 3, 1)
    assert torch.equal(calls["atom_ids"], torch.zeros_like(calls["atom_ids"]))
    assert torch.allclose(calls["coeffs"], torch.full_like(calls["coeffs"], -1.0))


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
    # Collapsed to a single plain gradient-trained dictionary (June 2026): the
    # online-K-SVD, dictionary-through-decoder, usage-EMA and dead-atom-revival
    # knobs were all removed.
    assert "dictionary_through_decoder" not in params
    assert "dead_atom_revival_steps" not in params
    assert "dictionary_usage_ema_decay" not in params
    assert "dictionary_usage_grad_scale" not in params
    assert "dictionary_usage_grad_min" not in params
    assert "dictionary_usage_grad_max" not in params
    assert "dictionary_ksvd_lr" not in params
    assert "dictionary_ksvd_update_every" not in params
    assert "dictionary_ksvd_min_usage" not in params
    assert "dictionary_ksvd_max_atoms_per_step" not in params
    assert "online_ksvd_enabled" not in params
    assert "online_ksvd_start_step" not in params
    assert "online_ksvd_interval_steps" not in params
    assert "online_ksvd_stop_step" not in params
    assert "online_ksvd_max_samples" not in params
    assert "online_ksvd_max_atoms" not in params
    assert "online_ksvd_blend" not in params
    assert "dict_ema_decay" not in params
    assert "dict_ema_eps" not in params
    assert "fast_omp" not in params
    assert "omp_diag_eps" not in params
    assert "omp_cholesky_eps" not in params
    assert "omp_residual_tolerance" not in params
    assert "bounded_omp_refine_steps" not in params
    assert "sparse_coding_scheme" not in params
    assert "lista_steps" not in params
    assert "lista_step_size_init" not in params
    assert "lista_threshold_init" not in params
