#!/usr/bin/env python3
"""Sequential H100 sweep for fully autoregressive real-valued FFHQ stage-2 runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts" / "run_ozbyadv50_stage2_realcoeff_h100.py"
TOKEN_CACHE = (
    REPO
    / "outputs"
    / "ffhq_ozbyadv50_stage2_realcoeff_h100"
    / "stage2"
    / "token_cache"
    / "ffhq__train__img256__laser_real.pt"
)
SWEEP_ROOT = REPO / "outputs" / "ffhq_ozbyadv50_stage2_realcoeff_ar_sweep_h100"


@dataclass(frozen=True)
class Candidate:
    name: str
    coverage: str
    smoothing: str


DEFAULT_CANDIDATES = (
    Candidate("cov000_smooth000", "0.0", "0.0"),
    Candidate("cov001_smooth000", "0.01", "0.0"),
    Candidate("cov002_smooth000", "0.02", "0.0"),
    Candidate("cov001_smooth005", "0.01", "0.005"),
)


def q(value: object) -> str:
    return shlex.quote(str(value))


def parse_fids(log_path: Path) -> list[float]:
    pattern = re.compile(r"Generation FID for checkpoint monitor:\s*([0-9.]+)")
    values: list[float] = []
    if not log_path.is_file():
        return values
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            values.append(float(match.group(1)))
    return values


def run_candidate(args: argparse.Namespace, candidate: Candidate, stamp: str, results_path: Path) -> int:
    output_dir = SWEEP_ROOT / candidate.name / "stage2"
    log_dir = SWEEP_ROOT / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stamp}_{candidate.name}.log"
    wandb_id = f"ozbyadv50-realcoeff-ar-{candidate.name}-sweep{int(args.epochs)}-{stamp}"
    wandb_name = f"ffhq-stage2-realcoeff-ar-{candidate.name}-ozbyadv50-sweep{int(args.epochs)}-{stamp}"

    cmd = [
        sys.executable,
        str(LAUNCHER),
        "--no-background",
        "--epochs",
        str(int(args.epochs)),
        "--stage2-batch-size",
        str(int(args.batch_size)),
        "--output-dir",
        str(output_dir),
        "--token-cache",
        str(TOKEN_CACHE),
        "--autoregressive-coeffs",
        "--atom-coverage-weight",
        candidate.coverage,
        "--atom-label-smoothing",
        candidate.smoothing,
        "--wandb-id",
        wandb_id,
        "--wandb-name",
        wandb_name,
        "--wandb-group",
        f"ffhq-ozbyadv50-stage2-realcoeff-ar-sweep-{stamp}",
        "--wandb-resume",
        "never",
        "--checkpoint-save-top-k",
        "3",
        "--checkpoint-every-n-epochs",
        str(int(args.fid_every)),
        "--checkpoint-upload-every-n-epochs",
        str(int(args.fid_every)),
        "--checkpoint-monitor",
        "s2/generation_fid",
        "--checkpoint-mode",
        "min",
        "--generation-fid-every-n-epochs",
        str(int(args.fid_every)),
        "--generation-fid-num-samples",
        str(int(args.fid_samples)),
        "--generation-metric-num-samples",
        str(int(args.fid_samples)),
        "--compute-generation-fid",
        "--sample-every-n-epochs",
        str(int(args.sample_every)),
        "--sample-num-images",
        str(int(args.sample_num_images)),
        "--sample-temperature",
        str(args.sample_temperature),
        "--sample-top-k",
        str(int(args.sample_top_k)),
        "--sample-coeff-mode",
        "mean",
    ]

    print(f"[sweep] starting {candidate.name}", flush=True)
    print("[sweep] " + " ".join(q(part) for part in cmd), flush=True)
    with log_path.open("wb") as log:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT)

    fids = parse_fids(log_path)
    result = {
        "candidate": candidate.name,
        "coverage": candidate.coverage,
        "smoothing": candidate.smoothing,
        "returncode": int(proc.returncode),
        "wandb_id": wandb_id,
        "wandb_name": wandb_name,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "fid_values": fids,
        "best_fid": min(fids) if fids else None,
        "last_fid": fids[-1] if fids else None,
    }
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"[sweep] finished {candidate.name}: rc={proc.returncode} best_fid={result['best_fid']}", flush=True)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--fid-every", type=int, default=20)
    parser.add_argument("--fid-samples", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--sample-num-images", type=int, default=64)
    parser.add_argument("--sample-temperature", default="0.6")
    parser.add_argument("--sample-top-k", type=int, default=128)
    parser.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = SWEEP_ROOT / f"sweep_results_{stamp}.jsonl"
    print(f"[sweep] results: {results_path}", flush=True)
    print(f"[sweep] fid_samples={args.fid_samples} every={args.fid_every} epochs", flush=True)
    print(
        f"[sweep] sample_num_images={args.sample_num_images} every={args.sample_every} epochs",
        flush=True,
    )

    for candidate in DEFAULT_CANDIDATES:
        code = run_candidate(args, candidate, stamp, results_path)
        if code != 0 and bool(args.stop_on_failure):
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
