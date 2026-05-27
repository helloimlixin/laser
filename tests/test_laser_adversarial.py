"""End-to-end smoke for the gated PatchGAN adversarial path in LASER.

These run a real Lightning Trainer so the manual-optimization wiring (two
optimizers, clip_gradients, dictionary maintenance, warmup gating, adaptive
weight) is exercised exactly as in training, on tiny CPU tensors.
"""

import lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.laser import LASER


def _tiny_model(**overrides):
    kwargs = dict(
        in_channels=3,
        num_hiddens=32,  # U-Net GroupNorm requires widths divisible by 32
        num_embeddings=64,
        embedding_dim=8,
        sparsity_level=4,
        num_residual_blocks=1,
        num_residual_hiddens=32,
        commitment_cost=0.05,
        learning_rate=1e-3,
        beta=0.9,
        backbone="vqgan",
        resolution=32,
        num_downsamples=2,
        channel_multipliers=[1, 2, 2],
        attn_resolutions=[],
        use_mid_attention=False,
        perceptual_weight=0.0,
        compute_fid=False,
        recon_mse_weight=1.0,
        recon_l1_weight=0.0,
        patch_based=True,
        patch_size=2,
        patch_stride=2,
        log_images_every_n_steps=0,
    )
    kwargs.update(overrides)
    return LASER(**kwargs)


def _loader(n=8):
    x = torch.randn(n, 3, 32, 32).clamp(-1, 1)
    return DataLoader(TensorDataset(x), batch_size=4)


def test_adversarial_disabled_keeps_automatic_optimization():
    model = _tiny_model(adversarial_weight=0.0)
    assert model.automatic_optimization is True
    assert model.discriminator is None


def test_adversarial_enabled_switches_to_manual_optimization():
    model = _tiny_model(adversarial_weight=0.8)
    assert model.automatic_optimization is False
    assert model.discriminator is not None
    opts = model.configure_optimizers()
    assert isinstance(opts, list) and len(opts) == 2


def test_adversarial_training_runs_through_warmup_into_active():
    torch.manual_seed(0)
    # disc_start_step=2 so we cross the warmup boundary within a short fit. Use
    # epoch-based stopping: under manual optimization Lightning counts global_step
    # per optimizer.step(), so two optimizers/batch would truncate max_steps early
    # (which is why scheduling/gating uses the model's own per-batch counter).
    model = _tiny_model(adversarial_weight=0.8, disc_start_step=2, grad_clip_val=1.0)
    # Snapshot a critic weight to confirm the discriminator actually updates once
    # warmup completes (batches 2 and 3 are "active").
    disc_w0 = next(model.discriminator.parameters()).detach().clone()
    trainer = pl.Trainer(
        max_epochs=2,  # loader yields 2 batches/epoch -> 4 training batches
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=2,
    )
    trainer.fit(model, train_dataloaders=_loader(), val_dataloaders=_loader())

    # The per-batch counter advanced once per training batch (not per optimizer step).
    assert int(model._manual_train_step.item()) == 4
    # The critic trained after warmup.
    disc_w1 = next(model.discriminator.parameters()).detach()
    assert not torch.allclose(disc_w0, disc_w1)
    # Everything stayed finite (no NaN blow-up from the GAN term).
    assert all(torch.isfinite(p).all() for p in model.discriminator.parameters())
    assert all(torch.isfinite(p).all() for p in model.decoder.parameters())


def test_token_budget_invariant_under_resolution_swap():
    """The FFHQ recipe doubles latent resolution but holds the stage-2 token
    sequence length fixed by enlarging the bottleneck patch (grid = latent/stride).

    Mirrors the real 256px case at 64px:
      * "old":  f8  -> 8x8 latent,  patch/stride 2 -> 4x4 = 16 patches
      * "new":  f4  -> 16x16 latent, patch/stride 4 -> 4x4 = 16 patches
    """
    common = dict(num_embeddings=64, embedding_dim=8, sparsity_level=4, resolution=64)
    old = _tiny_model(
        num_downsamples=3, channel_multipliers=[1, 2, 2, 4],
        patch_size=2, patch_stride=2, **common,
    )
    new = _tiny_model(
        num_downsamples=2, channel_multipliers=[1, 2, 2],
        patch_size=4, patch_stride=4, **common,
    )
    assert old.infer_latent_hw((64, 64)) == (8, 8)
    assert new.infer_latent_hw((64, 64)) == (16, 16)

    x = torch.randn(1, 3, 64, 64).clamp(-1, 1)
    tok_kwargs = dict(coeff_vocab_size=16, coeff_max=8.0)
    old_tokens, _ = old.encode_to_tokens(x, **tok_kwargs)
    new_tokens, _ = new.encode_to_tokens(x, **tok_kwargs)
    # Identical sequence length despite the finer latent grid.
    assert old_tokens.shape[-1] == new_tokens.shape[-1]


def test_adaptive_weight_is_nonnegative_and_detached():
    torch.manual_seed(0)
    model = _tiny_model(adversarial_weight=0.8, disc_start_step=0)
    model.train()
    x = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    recon, bottleneck_loss, _ = model(x)
    recon_loss = torch.nn.functional.mse_loss(recon, x)
    g_loss = -model.discriminator(recon).mean()
    w = model._adaptive_disc_weight(recon_loss, g_loss)
    assert w.item() >= 0.0
    assert not w.requires_grad
