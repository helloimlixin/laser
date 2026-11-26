"""Helpers for inferring maintained stage-1/stage-2 artifact paths."""

from __future__ import annotations

import glob
import stat
from pathlib import Path
from typing import Iterable


def _latest_existing_file(paths: Iterable[Path]) -> Path | None:
    latest: Path | None = None
    latest_key: tuple[float, str] | None = None
    for path in paths:
        try:
            info = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        key = (info.st_mtime, str(path))
        if latest_key is None or key > latest_key:
            latest, latest_key = path, key
    return latest


def latest_matching_file(patterns: Iterable[str | Path]) -> Path | None:
    candidates: list[Path] = []
    for pattern in patterns:
        pattern = str(pattern)
        if any(ch in pattern for ch in "*?[]"):
            candidates.extend(Path(path) for path in glob.glob(pattern))
        else:
            candidates.append(Path(pattern))
    return _latest_existing_file(path.resolve() for path in candidates)


def infer_latest_stage1_checkpoint(
    *,
    output_root: str | Path = "outputs",
    model_type: str = "laser",
) -> Path | None:
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
) -> Path | None:
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
    coeff_mu: float = 0.0,
    support_order: str = "atom_id",
) -> str:
    dataset = str(dataset).strip().lower()
    split = str(split).strip().lower()
    quant = str(coeff_quantization).strip().lower()
    parts = [dataset, split, f"img{int(image_size)}", f"cb{int(coeff_bins)}"]
    if quant != "uniform":
        parts.append(quant)
        mu = float(coeff_mu)
        if mu > 0.0:
            mu_token = f"{mu:g}".replace("-", "m").replace(".", "p")
            parts.append(f"mu{mu_token}")
    order = str(support_order or "atom_id").strip().lower()
    if order not in {"", "none", "atom_id"}:
        parts.append(f"order{order}")
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
    coeff_mu: float = 0.0,
    support_order: str = "atom_id",
) -> Path:
    return token_cache_dir(ar_output_dir) / default_token_cache_filename(
        dataset=dataset,
        split=split,
        image_size=image_size,
        coeff_bins=coeff_bins,
        coeff_quantization=coeff_quantization,
        coeff_mu=coeff_mu,
        support_order=support_order,
    )


def infer_latest_token_cache(
    *,
    ar_output_dir: str | Path = "outputs/ar",
    dataset: str | None = None,
    split: str | None = None,
) -> Path | None:
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
    return _latest_existing_file(matches)
