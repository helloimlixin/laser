import torch

from src.data.token_cache import TokenCacheDataModule, load_token_cache


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


def test_token_cache_datamodule_crops_latent_windows(tmp_path):
    cache_path = tmp_path / "tokens_crop.pt"
    tokens = torch.arange(2 * 3 * 4 * 2, dtype=torch.int32).view(2, 3 * 4 * 2)
    torch.save(
        {
            "tokens_flat": tokens,
            "shape": (3, 4, 2),
            "meta": {"num_atoms": 8},
        },
        cache_path,
    )

    dm = TokenCacheDataModule(
        cache_path=str(cache_path),
        batch_size=1,
        num_workers=0,
        seed=7,
        validation_fraction=0.5,
        test_fraction=0.0,
        crop_h_sites=2,
        crop_w_sites=2,
    )
    dm.setup()

    assert dm.token_shape == (2, 2, 2)
    val_item = dm.val_dataset[0]
    src_idx = dm.val_dataset.indices[0]
    expected = tokens[src_idx].view(3, 4, 2)[0:2, 1:3, :].reshape(-1)
    assert torch.equal(val_item, expected.to(torch.long))


def test_load_token_cache_preserves_saved_order_by_default(tmp_path):
    cache_path = tmp_path / "tokens_raw.pt"
    saved = {
        "tokens_flat": torch.tensor([[7, 70, 2, 20]], dtype=torch.int32),
        "shape": (1, 1, 4),
        "meta": {"dataset": "toy"},
    }
    torch.save(saved, cache_path)

    got = load_token_cache(cache_path)

    assert torch.equal(got["tokens_flat"], saved["tokens_flat"])
    assert got["meta"] == saved["meta"]


def test_load_token_cache_can_opt_in_to_canonicalization(tmp_path):
    cache_path = tmp_path / "tokens_canon.pt"
    torch.save(
        {
            "tokens_flat": torch.tensor([[7, 70, 2, 20]], dtype=torch.int32),
            "shape": (1, 1, 4),
            "meta": {"dataset": "toy"},
        },
        cache_path,
    )

    got = load_token_cache(cache_path, canonicalize=True)

    assert torch.equal(got["tokens_flat"], torch.tensor([[2, 20, 7, 70]], dtype=torch.int32))
    assert got["meta"]["support_order"] == "atom_id"
