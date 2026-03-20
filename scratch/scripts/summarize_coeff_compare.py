#!/usr/bin/env python3
"""Summarize coefficient-comparison sweeps from W&B run summaries."""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple


RUN_RE = re.compile(
    r"^(?P<prefix>.+?)_"
    r"(?P<mode>rv|qb\d+_c[0-9p]+)_"
    r"p(?P<patch>\d+)s(?P<stride>\d+)_"
    r"a(?P<atoms>\d+)_k(?P<sparsity>\d+)$"
)


class RunRow(NamedTuple):
    run_name: str
    timestamp: str
    mode: str
    patch: int
    stride: int
    atoms: int
    sparsity: int
    stage1_val_loss: Optional[float]
    stage1_val_recon: Optional[float]
    stage1_psnr: Optional[float]
    stage1_ssim: Optional[float]
    stage2_epoch_loss: Optional[float]
    stage2_train_loss: Optional[float]
    stage2_ce_loss: Optional[float]
    coeff_term_name: str
    coeff_term_value: Optional[float]
    coeff_raw_name: str
    coeff_raw_value: Optional[float]
    summary_path: Path


def _fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}"


def _parse_run_name(run_name: str) -> Dict[str, object]:
    match = RUN_RE.match(run_name)
    if not match:
        raise ValueError(f"Unrecognized run name format: {run_name}")
    groups = match.groupdict()
    return {
        "mode": groups["mode"],
        "patch": int(groups["patch"]),
        "stride": int(groups["stride"]),
        "atoms": int(groups["atoms"]),
        "sparsity": int(groups["sparsity"]),
    }


def _maybe_float(data: Dict[str, object], key: str) -> Optional[float]:
    value = data.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _find_latest_summaries(root: Path) -> List[Path]:
    latest_by_run = {}  # type: Dict[str, Path]
    for summary_path in root.glob("*/**/latest-run/files/wandb-summary.json"):
        rel = summary_path.relative_to(root)
        if len(rel.parts) < 2:
            continue
        run_name = rel.parts[0]
        timestamp = rel.parts[1]
        current = latest_by_run.get(run_name)
        if current is None:
            latest_by_run[run_name] = summary_path
            continue
        current_ts = current.relative_to(root).parts[1]
        if timestamp > current_ts:
            latest_by_run[run_name] = summary_path
    return sorted(latest_by_run.values())


def _load_rows(root: Path) -> List[RunRow]:
    rows = []  # type: List[RunRow]
    for summary_path in _find_latest_summaries(root):
        rel = summary_path.relative_to(root)
        run_name = rel.parts[0]
        timestamp = rel.parts[1]
        meta = _parse_run_name(run_name)
        with summary_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        mode = str(meta["mode"])
        if mode == "rv":
            coeff_term_name = "weighted_coeff_loss"
            coeff_term_value = _maybe_float(data, "stage2/weighted_coeff_loss")
            coeff_raw_name = str(data.get("stage2/coeff_loss_type", "coeff_loss"))
            coeff_raw_value = _maybe_float(data, "stage2/coeff_mse_loss")
        else:
            coeff_term_name = "weighted_coeff_ce"
            coeff_term_value = _maybe_float(data, "stage2/weighted_coeff_ce_loss")
            coeff_raw_name = "coeff_ce"
            coeff_raw_value = _maybe_float(data, "stage2/coeff_ce_loss")
        rows.append(
            RunRow(
                run_name=run_name,
                timestamp=timestamp,
                mode=mode,
                patch=int(meta["patch"]),
                stride=int(meta["stride"]),
                atoms=int(meta["atoms"]),
                sparsity=int(meta["sparsity"]),
                stage1_val_loss=_maybe_float(data, "stage1/val_loss"),
                stage1_val_recon=_maybe_float(data, "stage1/val_recon_loss"),
                stage1_psnr=_maybe_float(data, "stage1/val_psnr"),
                stage1_ssim=_maybe_float(data, "stage1/val_ssim"),
                stage2_epoch_loss=_maybe_float(data, "stage2/epoch_loss"),
                stage2_train_loss=_maybe_float(data, "stage2/train_loss"),
                stage2_ce_loss=_maybe_float(data, "stage2/ce_loss"),
                coeff_term_name=coeff_term_name,
                coeff_term_value=coeff_term_value,
                coeff_raw_name=coeff_raw_name,
                coeff_raw_value=coeff_raw_value,
                summary_path=summary_path,
            )
        )
    return sorted(rows, key=lambda row: (row.mode, row.atoms, row.sparsity))


def _iter_matched_pairs(rows: Iterable[RunRow]) -> List[Tuple[RunRow, RunRow]]:
    grouped = {}  # type: Dict[Tuple[int, int], Dict[str, RunRow]]
    for row in rows:
        grouped.setdefault((row.atoms, row.sparsity), {})[row.mode] = row
    pairs = []  # type: List[Tuple[RunRow, RunRow]]
    for key in sorted(grouped):
        modes = grouped[key]
        if "qb512_c4p0" in modes and "rv" in modes:
            pairs.append((modes["qb512_c4p0"], modes["rv"]))
    return pairs


def _best_by(rows: Iterable[RunRow], key: str) -> Optional[RunRow]:
    filtered = [row for row in rows if getattr(row, key) is not None]
    if not filtered:
        return None
    return min(filtered, key=lambda row: getattr(row, key))


def _render_table(rows: Iterable[RunRow]) -> str:
    header = (
        "| run | mode | atoms | k | stage1 val_loss | stage1 psnr | "
        "stage1 ssim | stage2 epoch_loss | coeff term | coeff raw |"
    )
    sep = "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    body = []
    for row in rows:
        body.append(
            "| {run} | {mode} | {atoms} | {k} | {val_loss} | {psnr} | {ssim} | {epoch_loss} | {coeff_term} | {coeff_raw} |".format(
                run=row.run_name,
                mode=row.mode,
                atoms=row.atoms,
                k=row.sparsity,
                val_loss=_fmt(row.stage1_val_loss),
                psnr=_fmt(row.stage1_psnr, 3),
                ssim=_fmt(row.stage1_ssim, 4),
                epoch_loss=_fmt(row.stage2_epoch_loss),
                coeff_term=_fmt(row.coeff_term_value),
                coeff_raw=_fmt(row.coeff_raw_value),
            )
        )
    return "\n".join([header, sep, *body])


def _render_observations(rows: List[RunRow]) -> List[str]:
    observations = []  # type: List[str]
    pairs = _iter_matched_pairs(rows)
    if pairs:
        q_wins = sum(
            1 for q_row, rv_row in pairs
            if q_row.stage1_val_loss is not None
            and rv_row.stage1_val_loss is not None
            and q_row.stage1_val_loss < rv_row.stage1_val_loss
        )
        observations.append(
            f"Quantized stage-1 val_loss beats real-valued in {q_wins}/{len(pairs)} matched atoms/sparsity pairs."
        )
    best_stage1 = _best_by(rows, "stage1_val_loss")
    if best_stage1 is not None:
        observations.append(
            "Best stage-1 run: "
            f"{best_stage1.run_name} "
            f"(val_loss={_fmt(best_stage1.stage1_val_loss)}, "
            f"psnr={_fmt(best_stage1.stage1_psnr, 3)}, "
            f"ssim={_fmt(best_stage1.stage1_ssim, 4)})."
        )
    best_stage2 = _best_by(rows, "stage2_epoch_loss")
    if best_stage2 is not None:
        observations.append(
            "Lowest stage-2 epoch_loss in this sweep: "
            f"{best_stage2.run_name} "
            f"(epoch_loss={_fmt(best_stage2.stage2_epoch_loss)})."
        )
    outlier = None
    for row in rows:
        if row.mode == "rv" and (row.coeff_term_value or 0.0) > 1.0:
            if outlier is None or (row.coeff_term_value or 0.0) > (outlier.coeff_term_value or 0.0):
                outlier = row
    if outlier is not None:
        observations.append(
            "Largest real-valued coefficient penalty: "
            f"{outlier.run_name} "
            f"({outlier.coeff_term_name}={_fmt(outlier.coeff_term_value)}, "
            f"{outlier.coeff_raw_name}={_fmt(outlier.coeff_raw_value)})."
        )
    observations.append(
        "Stage-2 losses are only approximately comparable across modes because quantized runs use coefficient CE while real-valued runs use a weighted coefficient regression term."
    )
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root",
        type=Path,
        help="Root directory that contains coeff-compare runs.",
    )
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        print(f"Missing run root: {run_root}", file=sys.stderr)
        return 1

    rows = _load_rows(run_root)
    if not rows:
        print(f"No coeff-compare summaries found under {run_root}", file=sys.stderr)
        return 1

    print(f"# Coeff Compare Summary\n")
    print(f"Run root: `{run_root}`\n")
    print(_render_table(rows))
    print("\n## Observations\n")
    for line in _render_observations(rows):
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
