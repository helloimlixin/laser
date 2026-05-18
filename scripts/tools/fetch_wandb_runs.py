#!/usr/bin/env python3
"""Fetch compact W&B run metadata for launch reproduction."""

from __future__ import annotations

import argparse
import json

import wandb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="W&B run paths, e.g. entity/project/run_id")
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    for run_path in args.runs:
        run = api.run(run_path)
        payload = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "group": run.group,
            "job_type": run.job_type,
            "tags": list(run.tags or []),
            "created_at": str(run.created_at),
            "url": run.url,
            "summary": dict(run.summary),
            "config": dict(run.config),
        }
        print("===RUN===", run_path)
        print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
