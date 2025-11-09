import os
import sys
import pytest
import torch
import lightning as pl

# Add the project root to sys.path so 'src.*' imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.dlvae import DLVAE
from src.data.celeba import CelebADataModule
from src.data.config import DataConfig


def _has_celeba_images() -> bool:
    # Try to resolve using same logic as CelebADataModule by probing common env/config
    candidates = [
        os.environ.get('CELEBA_DIR', ''),
        '/home/xl598/Data/celeba/img_align_celeba',
        '/home/xl598/Data/celeba',
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            for root, _, files in os.walk(c):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) for f in files):
                    return True
    return False


@pytest.mark.skipif(not _has_celeba_images(), reason="CelebA images not available (set CELEBA_DIR).")
def test_dlvae_tiny_train_on_celeba():
    torch.manual_seed(0)
    pl.seed_everything(0)

    # Minimal data config
    data_dir = os.environ.get('CELEBA_DIR', '')
    cfg = DataConfig(
        dataset='celeba',
        data_dir=data_dir,
        batch_size=8,
        num_workers=0,
        image_size=32,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        augment=False,
    )
    dm = CelebADataModule(cfg)
    dm.prepare_data()
    dm.setup()

    # Small model for speed
    model = DLVAE(
        in_channels=3,
        num_hiddens=32,
        num_embeddings=64,
        embedding_dim=16,
        sparsity_level=3,
        num_residual_blocks=1,
        num_residual_hiddens=16,
        commitment_cost=0.25,
        decay=0.99,
        perceptual_weight=0.0,
        learning_rate=1e-3,
        beta=0.9,
        compute_fid=False,
        omp_tolerance=1e-7,
        omp_debug=False,
    )

    before_dict = model.bottleneck.dictionary.detach().clone()

    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        limit_train_batches=2,
        limit_val_batches=1,
        deterministic=True,
    )
    trainer.fit(model, datamodule=dm)

    # Run a forward on a small batch to confirm shapes
    batch = next(iter(dm.train_dataloader()))
    x, _ = batch
    recon, dl_loss, coeffs = model(x)
    assert recon.shape == x.shape
    assert dl_loss.ndim == 0
    assert coeffs.shape[0] == model.bottleneck.num_embeddings

    after_dict = model.bottleneck.dictionary.detach().clone()
    # Ensure that at least some training occurred (weights changed)
    assert not torch.allclose(before_dict, after_dict)


