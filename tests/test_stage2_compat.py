from pathlib import Path

import torch

from src.stage2_compat import (
    ensure_stage2_cache_metadata,
    ensure_quantized_cache_metadata,
    resolve_stage1_checkpoint_from_token_cache,
)


def test_resolve_stage1_checkpoint_from_token_cache_prefers_sibling_stage1_dir(tmp_path: Path):
    stage1_ckpt = tmp_path / "run_001" / "stage1" / "ae_best.pt"
    token_cache = tmp_path / "run_001" / "stage2" / "tokens_cache.pt"
    stage1_ckpt.parent.mkdir(parents=True, exist_ok=True)
    token_cache.parent.mkdir(parents=True, exist_ok=True)
    stage1_ckpt.write_bytes(b"stub")
    token_cache.write_bytes(b"stub")

    resolved = resolve_stage1_checkpoint_from_token_cache(token_cache)

    assert resolved == stage1_ckpt.resolve()


def test_ensure_quantized_cache_metadata_reads_coeff_bins_from_stage1_checkpoint(tmp_path: Path):
    stage1_ckpt = tmp_path / "run_001" / "stage1" / "ae_best.pt"
    token_cache = tmp_path / "run_001" / "stage2" / "tokens_cache.pt"
    stage1_ckpt.parent.mkdir(parents=True, exist_ok=True)
    token_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bottleneck.coef_bin_centers": torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32),
        },
        stage1_ckpt,
    )
    token_cache.write_bytes(b"stub")

    enriched = ensure_quantized_cache_metadata(
        {
            "tokens_flat": torch.zeros(2, 4, dtype=torch.int32),
            "shape": (1, 1, 4),
            "meta": {},
        },
        token_cache_path=token_cache,
        output_root=tmp_path / "outputs",
    )

    meta = enriched["meta"]
    assert meta["stage1_checkpoint"] == str(stage1_ckpt.resolve())
    assert meta["coeff_vocab_size"] == 3
    assert meta["n_bins"] == 3
    assert torch.equal(meta["coeff_bin_values"], torch.tensor([-1.0, 0.0, 1.0]))


def test_ensure_stage2_cache_metadata_marks_real_valued_caches_without_coeff_bins(tmp_path: Path):
    token_cache = tmp_path / "run_001" / "stage2" / "tokens_cache.pt"
    token_cache.parent.mkdir(parents=True, exist_ok=True)
    token_cache.write_bytes(b"stub")

    enriched = ensure_stage2_cache_metadata(
        {
            "tokens_flat": torch.zeros(2, 4, dtype=torch.int32),
            "coeffs_flat": torch.zeros(2, 4, dtype=torch.float32),
            "shape": (1, 1, 4),
            "meta": {"coeff_max": 5.0},
        },
        token_cache_path=token_cache,
        output_root=tmp_path / "outputs",
    )

    meta = enriched["meta"]
    assert meta["quantize_sparse_coeffs"] is False
    assert meta["coeff_max"] == 5.0
    assert meta["coef_max"] == 5.0
