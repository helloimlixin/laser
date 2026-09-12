#!/usr/bin/env python3
"""Launch the new repair-layer experiment with durable logs and private auth."""
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
    p.add_argument("--root-output", type=Path, default=ROOT / "outputs/lsun-church-neural-repair-20260910")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb-key-stdin", action="store_true")
    args = p.parse_args()
    output = args.root_output.resolve()
    assert (output / "pairs/complete.json").is_file()
    run = output / "train"
    if run.exists() and not args.resume:
        raise FileExistsError(run)
    if args.resume:
        status = json.loads((run / "status.json").read_text())
        cmdline = Path(f"/proc/{status['pid']}/cmdline")
        if cmdline.exists() and str(run).encode() in cmdline.read_bytes():
            raise RuntimeError("The repair trainer is already running")
    env = os.environ.copy()
    if args.wandb_key_stdin:
        env["WANDB_API_KEY"] = getpass.getpass("W&B API key: ")
    if not env.get("WANDB_API_KEY"):
        raise RuntimeError("W&B credentials are required for this launcher")
    sources = ["src/sparse_latent_repair.py", "scripts/build_church_repair_pairs.py",
               "scripts/train_church_latent_repair.py", "scripts/launch_church_latent_repair.py",
               "tests/test_sparse_latent_repair.py"]
    snapshot = output / "source-snapshot"
    hashes = {}
    for name in sources:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    (snapshot / "sha256.json").write_text(json.dumps(hashes, indent=2))
    run_id = "church-neural-repair-20260910"
    cmd = [sys.executable, "-u", str(ROOT / "scripts/train_church_latent_repair.py"),
           "--output", str(run), "--pairs", str(output / "pairs"), "--steps", str(args.steps),
           "--microbatch", str(args.microbatch), "--wandb-id", run_id]
    if args.resume:
        cmd.append("--resume")
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="8",
               TORCH_HOME="/workspace/tmp/official-rqvae-eval-cache",
               LASER_VGG16_WEIGHTS="/workspace/tmp/laser-vgg/vgg16-397923af.pth")
    with (output / "train.log").open("ab") as log:
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"pid": process.pid, "command": cmd, "gpu": args.gpu, "started_unix": time.time(),
              "log": str(output / "train.log"), "wandb_url": f"https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}"}
    (output / "train-launch.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
