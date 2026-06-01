"""Regression: online_ksvd must update the dictionary on the NON-adversarial path.

Bug (fixed): `online_ksvd_update_()` was only called from
`_adversarial_training_step` (the manual-optimization route). In a non-adversarial
run (`adversarial_weight=0`) Lightning uses automatic optimization, and in
`online_ksvd` mode the dictionary is `requires_grad=False` (so it is excluded from
the optimizer). With the K-SVD call missing from `optimizer_step`, NOTHING updated
the atoms — the dictionary stayed frozen at its initialization for the whole run.

These tests run a real Lightning Trainer on tiny CPU tensors so the
`optimizer_step` wiring is exercised exactly as in training.
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
        commitment_cost=0.25,
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


def _fit_a_few_steps(model, max_steps=4):
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model, train_dataloaders=_loader())


def test_online_ksvd_mode_setup():
    """In online_ksvd mode the dictionary is frozen for autograd and excluded
    from the optimizer, so K-SVD is its ONLY update route."""
    model = _tiny_model(adversarial_weight=0.0, dictionary_update_mode="online_ksvd")
    assert model.automatic_optimization is True  # non-adversarial -> automatic path
    assert model.bottleneck.dictionary_update_mode == "online_ksvd"
    assert model.bottleneck.dictionary.requires_grad is False

    optimizer = model.configure_optimizers()
    in_optim = any(
        any(p is model.bottleneck.dictionary for p in group["params"])
        for group in optimizer.param_groups
    )
    assert not in_optim, "ksvd dictionary must not be in the optimizer"


def test_online_ksvd_updates_dictionary_on_automatic_path():
    """The regression: a non-adversarial run must still move the atoms."""
    torch.manual_seed(0)
    model = _tiny_model(adversarial_weight=0.0, dictionary_update_mode="online_ksvd")
    dict0 = model.bottleneck.dictionary.detach().clone()

    _fit_a_few_steps(model, max_steps=4)

    dict1 = model.bottleneck.dictionary.detach()
    assert not torch.allclose(dict0, dict1), (
        "online_ksvd dictionary did not change after training on the automatic "
        "(non-adversarial) path — the K-SVD update is not wired into optimizer_step"
    )
    # Atoms stay unit-norm (normalize_dictionary_ runs after the update).
    norms = dict1.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert torch.isfinite(dict1).all()


def test_gradient_mode_also_updates_dictionary_on_automatic_path():
    """Contrast: gradient mode keeps requires_grad=True, is in the optimizer,
    and updates via backprop on the same path."""
    torch.manual_seed(0)
    model = _tiny_model(adversarial_weight=0.0, dictionary_update_mode="gradient")
    assert model.bottleneck.dictionary.requires_grad is True
    dict0 = model.bottleneck.dictionary.detach().clone()

    _fit_a_few_steps(model, max_steps=4)

    dict1 = model.bottleneck.dictionary.detach()
    assert not torch.allclose(dict0, dict1)
    assert torch.isfinite(dict1).all()
