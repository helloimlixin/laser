#!/usr/bin/env python3
"""Keep a Stage-2 W&B run live while its one-time token cache is built."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import wandb


CACHE_PROGRESS = re.compile(r"cache rank 0: ([\d,]+)/([\d,]+)")


def last_cache_progress(log_path: Path):
    if not log_path.is_file():
        return 0, 0
    current = total = 0
    for match in CACHE_PROGRESS.finditer(log_path.read_text(errors="replace")):
        current = int(match.group(1).replace(",", ""))
        total = int(match.group(2).replace(",", ""))
    return current, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        resume="allow",
        name=args.name,
    )
    run.define_metric("pipeline/cache_items")
    run.define_metric("pipeline/cache_fraction")
    cache_log = args.run_root / "logs" / "token_cache.log"
    cache_report = (
        args.run_root
        / "token_cache"
        / "lsun_bedroom_train_a16384k4_compound_pairs.validation.json"
    )
    status_path = args.run_root / "status.tsv"
    pipeline_pid_path = args.run_root / "pipeline.pid"
    last_items = -1
    try:
        while True:
            current, total = last_cache_progress(cache_log)
            if current != last_items:
                run.log({
                    "pipeline/cache_items": current,
                    "pipeline/cache_total": total,
                    "pipeline/cache_fraction": current / total if total else 0.0,
                    "pipeline/cache_active": 1,
                })
                last_items = current

            status = status_path.read_text(errors="replace") if status_path.is_file() else ""
            if cache_report.is_file() or "\ttoken_cache\tcomplete\t" in status:
                run.log({
                    "pipeline/cache_items": total or current,
                    "pipeline/cache_total": total or current,
                    "pipeline/cache_fraction": 1.0,
                    "pipeline/cache_active": 0,
                    "pipeline/cache_complete": 1,
                })
                break
            if "\ttoken_cache\tfailed\t" in status or "\tdriver\tfailed\t" in status:
                run.log({"pipeline/cache_active": 0, "pipeline/cache_failed": 1})
                break
            if pipeline_pid_path.is_file():
                pipeline_pid = int(pipeline_pid_path.read_text().strip())
                try:
                    Path(f"/proc/{pipeline_pid}").stat()
                except FileNotFoundError:
                    run.log({"pipeline/cache_active": 0, "pipeline/pipeline_exited": 1})
                    break
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
