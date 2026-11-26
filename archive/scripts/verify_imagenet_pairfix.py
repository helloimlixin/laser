"""Check fresh/resumed five-GPU updates and generation before production."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    b = args.base.resolve()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4", LASER_RUNTIME_ROOT=str(b / "runtime"),
        LASER_RUN_BASE=str(b), LASER_PERSISTENT_BASE=str(args.evidence),
        LASER_ACCUMULATION="2", LASER_COMPILE_BLOCKS="1", LASER_COMPILE_OBJECTIVE="1", LASER_LOCAL_PREFLIGHT="1",
        OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TORCHINDUCTOR_COMPILE_THREADS="2",
        TORCHINDUCTOR_CACHE_DIR=str(b / "inductor-cache"), NCCL_NVLS_ENABLE="0",
        TORCH_HOME="/tmp/laser-imagenet-stage2/torch-cache", PYTHONUNBUFFERED="1")
    recipe = yaml.safe_load((b / "recipe.yaml").read_text())
    options = recipe["options"]
    options.update(output=str(b / "preflight/train"), checkpoint_dir=str(b / "preflight/train/checkpoints"),
        wandb_mode="disabled", sample_grid_every=0, fid_every=0, save_step_freq=0, upload_checkpoints=False)
    phases = [("preflight", False, 20, False), ("preflight_resume", True, 2, False),
              ("generation", True, 0, True)]
    records = []
    for phase, resume, steps, generate in phases:
        options.update(resume=resume, max_optimizer_steps=steps)
        options.pop("smoke_test", None)
        options.pop("generation_smoke_test", None)
        options["generation_smoke_test" if generate else "smoke_test"] = True
        config = b / (phase + "-recipe.yaml")
        config.write_text(yaml.safe_dump(recipe, sort_keys=False))
        env["LASER_PHASE"] = phase
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=5",
                   str(b / "entry-production.py"), "--config", str(config)]
        start = time.monotonic()
        print(json.dumps(dict(phase=phase, status="starting")), flush=True)
        with (b / (phase + ".log")).open("w") as log:
            completed = subprocess.run(command, env=env, cwd=b / "runtime", stdout=log, stderr=subprocess.STDOUT)
        record = dict(phase=phase, exit_code=completed.returncode, seconds=time.monotonic() - start)
        records.append(record)
        (b / "preflight-progress.json").write_text(json.dumps(records, indent=2))
        print(json.dumps(record), flush=True)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    import torch
    checkpoint = torch.load(b / "preflight/train/checkpoints/last.pt", map_location="cpu", mmap=True, weights_only=False)
    assert checkpoint["global_step"] == 22
    assert checkpoint["batch_idx"] == 44
    assert {int(s["step"]) for s in checkpoint["optimizer"]["state"].values()} == {22}
    fresh = [json.loads((b / "preflight/verification" / f"startup-rank{rank}.json").read_text()) for rank in range(5)]
    finite = [json.loads((b / "preflight/verification" / f"step20-rank{rank}.json").read_text()) for rank in range(5)]
    resumed = [json.loads((b / "preflight_resume/verification" / f"startup-rank{rank}.json").read_text()) for rank in range(5)]
    assert all(r["fresh"] and r["optimizer_states"] == 0 and r["optimizer_step_before"] == 0 for r in fresh)
    assert all(r["finite"] and r["optimizer_step"] == 20 for r in finite)
    assert all(not r["fresh"] and r["optimizer_step_before"] == 20 for r in resumed)
    assert "Generation smoke test passed" in (b / "generation.log").read_text()
    proof = dict(passed=True, phases=records, fresh=fresh, finite_step20=finite, resumed=resumed,
                 checkpoint_step=22, microbatches_completed=44,
                 optimizer_states=len(checkpoint["optimizer"]["state"]),
                 production_loads_preflight_weights=False)
    for target in (b / "preflight-summary.json", args.evidence / "preflight-summary.json"):
        target.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(dict(passed=True, checkpoint_step=22, optimizer_states=proof["optimizer_states"])), flush=True)


if __name__ == "__main__":
    main()
