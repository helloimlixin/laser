"""Core token-cache behavior: deterministic splits, crops, loading, and sorting."""

import torch

from src.cache_sort import sort_token_pairs
from src.data.token_cache import TokenCacheDataModule, load_token_cache


def test_token_cache_datamodule_splits_and_crops(tmp_path):
    cache_path = tmp_path / "tokens.pt"
    tokens = torch.arange(4 * 3 * 4 * 2, dtype=torch.int32).view(4, 3 * 4 * 2)
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
        batch_size=2,
        num_workers=0,
        seed=7,
        validation_fraction=0.25,
        test_fraction=0.25,
        crop_h_sites=2,
        crop_w_sites=2,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    assert dm.token_shape == (2, 2, 2)
    assert len(dm.train_dataset) == 2
    assert len(dm.val_dataset) == 1
    assert len(dm.test_dataset) == 1
    assert batch.shape == (2, 8)


def test_token_cache_load_and_pair_sorting(tmp_path):
    cache_path = tmp_path / "tokens_unsorted.pt"
    saved = {
        "tokens_flat": torch.tensor([[7, 70, 2, 20]], dtype=torch.int32),
        "shape": (1, 1, 4),
        "meta": {"dataset": "toy"},
    }
    torch.save(saved, cache_path)

    raw = load_token_cache(cache_path)
    # Old caches may have unsorted sparse support; canonicalization is opt-in so
    # normal loading does not silently rewrite token order.
    canonical = load_token_cache(cache_path, canonicalize=True)

    assert torch.equal(raw["tokens_flat"], saved["tokens_flat"])
    assert torch.equal(sort_token_pairs(saved["tokens_flat"]), torch.tensor([[2, 20, 7, 70]], dtype=torch.int32))
    assert torch.equal(canonical["tokens_flat"], torch.tensor([[2, 20, 7, 70]], dtype=torch.int32))
    assert canonical["meta"]["support_order"] == "atom_id"
