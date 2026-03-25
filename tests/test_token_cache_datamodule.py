import torch

from src.data.token_cache import TokenCacheDataModule


def test_token_cache_datamodule_splits_and_exposes_shape(tmp_path):
    cache_path = tmp_path / "tokens.pt"
    torch.save(
        {
            "tokens_flat": torch.arange(60, dtype=torch.int32).view(10, 6),
            "shape": (1, 3, 2),
            "meta": {"num_atoms": 4},
        },
        cache_path,
    )

    dm = TokenCacheDataModule(
        cache_path=str(cache_path),
        batch_size=4,
        num_workers=0,
        seed=123,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    dm.setup()

    assert dm.token_shape == (1, 3, 2)
    assert len(dm.train_dataset) == 6
    assert len(dm.val_dataset) == 2
    assert len(dm.test_dataset) == 2

    batch = next(iter(dm.train_dataloader()))
    assert batch.shape[-1] == 6
