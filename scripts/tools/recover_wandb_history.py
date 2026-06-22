#!/usr/bin/env python3
"""Recover scalar W&B history from a local .wandb file into a chartable run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def _nested_key(item: Any) -> str:
    nested = getattr(item, "nested_key", "")
    if isinstance(nested, str):
        return nested
    return "/".join(str(part) for part in nested)


def _json_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def iter_scalar_history(path: Path):
    store = DataStore()
    store.open_for_scan(str(path))
    last_global_step = None
    fallback_step = 0
    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue

        row: dict[str, Any] = {}
        raw_step = None
        for item in record.history.item:
            key = _nested_key(item)
            if not key:
                continue
            value = _json_value(item.value_json)
            if key == "_step":
                raw_step = value
                continue
            if key.startswith("_"):
                continue
            if not _is_scalar(value):
                continue
            row[key] = value

        if "trainer/global_step" in row:
            last_global_step = row["trainer/global_step"]
        elif last_global_step is not None:
            row["trainer/global_step"] = last_global_step

        if not row:
            continue

        step = getattr(record.history.step, "num", 0) or raw_step or fallback_step
        try:
            step = int(step)
        except Exception:
            step = fallback_step
        fallback_step = max(fallback_step + 1, step + 1)
        yield step, row


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_") or "metric"


def _plot_metric(rows: list[tuple[int, dict[str, Any]]], *, metric: str, x_key: str):
    data = []
    last_x = None
    for step, row in rows:
        if x_key in row:
            last_x = row[x_key]
        x_value = row.get(x_key, last_x if x_key == "trainer/global_step" else step)
        value = row.get(metric)
        if value is None or x_value is None:
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        if not isinstance(x_value, int | float) or isinstance(x_value, bool):
            continue
        data.append([x_value, value])
    if not data:
        return None
    table = wandb.Table(data=data, columns=[x_key, metric])
    return wandb.plot.line(table, x_key, metric, title=metric)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wandb_file", type=Path)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--group", default="")
    parser.add_argument("--id", default="")
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument(
        "--plot-summary",
        action="store_true",
        help="Also upload W&B custom line plots backed by Tables.",
    )
    parser.add_argument(
        "--plot-key",
        action="append",
        default=[],
        help="Metric key to upload as a custom plot. Repeat as needed.",
    )
    parser.add_argument("--x-key", default="trainer/global_step")
    args = parser.parse_args()

    rows = list(iter_scalar_history(args.wandb_file))
    if not rows:
        raise SystemExit(f"No scalar history rows found in {args.wandb_file}")

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.id or None,
        resume="allow" if args.id else None,
        name=args.name,
        group=args.group or None,
        tags=["history-recovered"],
        config={
            "source_wandb_file": str(args.wandb_file),
            "source_run_id": args.source_run_id,
            "source_url": args.source_url,
        },
    )
    wandb.define_metric("trainer/global_step")
    wandb.define_metric("*", step_metric="trainer/global_step")

    logged = 0
    for step, row in rows:
        wandb.log(row, step=step)
        logged += 1
    plotted = 0
    if args.plot_summary:
        plot_keys = args.plot_key or [
            "train/loss",
            "train/psnr",
            "train/ssim",
            "train/audio_lsd",
            "train/adversarial_generator_loss",
            "train/weighted_adversarial_generator_loss",
            "train/disc_loss",
            "val/loss",
            "val/psnr",
            "val/ssim",
            "val/rfid",
            "val/audio_lsd",
            "val/adversarial_generator_loss",
            "val/weighted_adversarial_generator_loss",
        ]
        payload = {}
        for metric in plot_keys:
            plot = _plot_metric(rows, metric=metric, x_key=args.x_key)
            if plot is None:
                continue
            payload[f"plots/{_safe_key(metric)}"] = plot
            plotted += 1
        if payload:
            wandb.log(payload, step=rows[-1][0])
    run.finish()
    print(f"Recovered {logged} scalar history rows and {plotted} plots to {run.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
