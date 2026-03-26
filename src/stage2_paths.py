"""Helpers for inferring maintained stage-1/stage-2 artifact paths."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, Optional


def _sorted_existing(paths: Iterable[Path]) -> list[Path]:
    existing = [path for path in paths if path.exists() and path.is_file()]
    return sorted(existing, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)


def latest_matching_file(patterns: Iterable[str | Path]) -> Optional[Path]:
    candidates: list[Path] = []
    for pattern in patterns:
        pattern = str(pattern)
        if any(ch in pattern for ch in "*?[]"):
            candidates.extend(Path(path) for path in glob.glob(pattern))
        else:
            candidates.append(Path(pattern))
    existing = _sorted_existing(path.resolve() for path in candidates)
    return existing[0] if existing else None


def infer_latest_stage1_checkpoint(
    *,
    output_root: str | Path = "outputs",
    model_type: str = "laser",
) -> Optional[Path]:
    root = Path(output_root).expanduser().resolve()
    return latest_matching_file(
        [
            root / "checkpoints" / "run_*" / model_type / "last.ckpt",
            root / "checkpoints" / "run_*" / model_type / "*.ckpt",
        ]
    )


def infer_latest_stage2_checkpoint(
    *,
    ar_output_dir: str | Path = "outputs/ar",
) -> Optional[Path]:
    root = Path(ar_output_dir).expanduser().resolve()
    return latest_matching_file(
        [
            root / "checkpoints" / "*" / "last.ckpt",
            root / "checkpoints" / "*" / "*.ckpt",
        ]
    )


def default_token_cache_filename(
    *,
    dataset: str,
    split: str,
    image_size: int,
    coeff_bins: int,
    coeff_quantization: str = "uniform",
) -> str:
    dataset = str(dataset).strip().lower()
    split = str(split).strip().lower()
    quant = str(coeff_quantization).strip().lower()
    parts = [dataset, split, f"img{int(image_size)}", f"cb{int(coeff_bins)}"]
    if quant != "uniform":
        parts.append(quant)
    return "__".join(parts) + ".pt"


def token_cache_dir(ar_output_dir: str | Path = "outputs/ar") -> Path:
    return Path(ar_output_dir).expanduser().resolve() / "token_cache"


def default_token_cache_path(
    *,
    ar_output_dir: str | Path = "outputs/ar",
    dataset: str,
    split: str,
    image_size: int,
    coeff_bins: int,
    coeff_quantization: str = "uniform",
) -> Path:
    return token_cache_dir(ar_output_dir) / default_token_cache_filename(
        dataset=dataset,
        split=split,
        image_size=image_size,
        coeff_bins=coeff_bins,
        coeff_quantization=coeff_quantization,
    )


def infer_latest_token_cache(
    *,
    ar_output_dir: str | Path = "outputs/ar",
    dataset: Optional[str] = None,
    split: Optional[str] = None,
) -> Optional[Path]:
    root = token_cache_dir(ar_output_dir)
    filters = []
    if dataset:
        filters.append(str(dataset).strip().lower())
    if split:
        filters.append(str(split).strip().lower())

    matches = []
    for path in root.glob("*.pt"):
        stem = path.stem.lower()
        if all(token in stem for token in filters):
            matches.append(path.resolve())
    existing = _sorted_existing(matches)
    return existing[0] if existing else None
