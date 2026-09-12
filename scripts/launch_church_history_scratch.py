#!/usr/bin/env python3
"""Launch durable, separate-GPU Church scratch arms with private W&B auth."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/lsun-church-history-scratch-20260910")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--microbatch", type=int, default=128)
    p.add_argument("--arms", nargs="+", choices=["control", "recovery"], default=["control", "recovery"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb-key-stdin", action="store_true")
    args = p.parse_args()
    args.output = args.output.resolve()
    initial = args.output / "initial-prior.pt"
    metadata = json.loads(initial.with_suffix(".json").read_text())
    assert metadata["initialization"] == "random" and metadata["seed"] == 4701
    for arm in args.arms:
        directory = args.output / arm
        if directory.exists() and not args.resume:
            raise FileExistsError(directory)
        status_path = directory / "status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            process = Path(f"/proc/{status['pid']}/cmdline")
            if process.exists() and str(directory).encode() in process.read_bytes():
                raise RuntimeError(f"{arm} is already running with PID {status['pid']}")
    environment = os.environ.copy()
    if args.wandb_key_stdin:
        environment["WANDB_API_KEY"] = getpass.getpass("W&B API key: ")
    if not environment.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY or supply --wandb-key-stdin")
    sources = ["scripts/train_church_history_scratch.py", "scripts/launch_church_history_scratch.py",
               "src/coefficient_history_training.py", "scripts/train_church_bar_coefficients.py",
               "scripts/train_official_rqtransformer_laser_stage2.py", "tests/test_coefficient_history_training.py"]
    snapshot = args.output / "source-snapshot"
    hashes = {}
    for name in sources:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (snapshot / "sha256.json").write_text(json.dumps(hashes, indent=2))
    launches = []
    for arm in args.arms:
        gpu = "0" if arm == "control" else "1"
        run_id = f"church-history-scratch-{arm}-20260910"
        command = [sys.executable, "-u", str(ROOT / "scripts/train_church_history_scratch.py"),
                   "--arm", arm, "--output", str(args.output / arm), "--initial-prior", str(initial),
                   "--epochs", str(args.epochs), "--batch-size", "512", "--microbatch", str(args.microbatch),
                   "--wandb-id", run_id, "--upload-final-checkpoint"]
        if args.resume:
            command.append("--resume")
        child_env = {**environment, "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "8",
                     "TORCH_HOME": "/workspace/tmp/official-rqvae-eval-cache", "PYTHONUNBUFFERED": "1"}
        logfile = args.output / f"{arm}.log"
        with logfile.open("ab") as stdout:
            child = subprocess.Popen(command, cwd=ROOT, env=child_env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        entry = {"arm": arm, "pid": child.pid, "gpu": int(gpu), "command": command,
                 "log": str(logfile), "started_unix": time.time(), "epochs": args.epochs,
                 "wandb_url": f"https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}"}
        launches.append(entry)
        (args.output / f"launch-{arm}.json").write_text(json.dumps(entry, indent=2))
        print(json.dumps(entry), flush=True)
    (args.output / "launches.json").write_text(json.dumps(launches, indent=2))


if __name__ == "__main__":
    main()
