#!/usr/bin/env python3
"""Wait for the pilot and independent codes, then evaluate trained repair weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def active(pid):
    status = Path(f"/proc/{pid}/stat")
    return status.exists() and status.read_text().split()[2] != "Z"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    args = p.parse_args()
    experiment = args.experiment.resolve()
    train = experiment / "train"
    generation = experiment / "independent-50000"
    train_pid = json.loads((experiment / "train-launch.json").read_text())["pid"]
    generation_pid = json.loads((experiment / "independent-50000-launch.json").read_text())["pid"]
    while active(train_pid) or active(generation_pid):
        time.sleep(15)
    if not (train / "results.json").exists() or not (generation / "results.json").exists():
        raise RuntimeError("A prerequisite failed or paused; inspect the training and generation logs")
    training = json.loads((train / "results.json").read_text())
    candidates = ["final"]
    if 0 < training["selected_step"] < training["trained_steps"]:
        candidates.append("best")
    rows = {}
    for candidate in candidates:
        destination = experiment / f"independent-50000-{candidate}-repair"
        command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_church_latent_repair.py"),
                   "--checkpoint", str(train / (candidate + ".pt")),
                   "--codes", str(generation / "generated-codes.pt"), "--output", str(destination)]
        subprocess.run(command, check=True)
        rows[candidate] = json.loads((destination / "metrics.json").read_text())
    result = {"selected_step": training["selected_step"], "trained_steps": training["trained_steps"],
              "independent_generation": json.loads((generation / "results.json").read_text()), "evaluations": rows}
    (experiment / "independent-50000-comparison.json").write_text(json.dumps(result, indent=2))
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(entity="helloimlixin-rutgers", project="laser", id="church-neural-repair-20260910",
                         resume="must", dir=str(experiment))
        for name, metrics in rows.items():
            run.summary[f"fid50k/{name}/baseline"] = metrics["baseline"]["fid"]
            run.summary[f"fid50k/{name}/repair"] = metrics["repair"]["fid"]
            run.summary[f"fid50k/{name}/delta"] = metrics["fid_delta"]
            run.summary[f"diversity/{name}/baseline_lpips"] = metrics["baseline"]["image_pair_lpips"]
            run.summary[f"diversity/{name}/repair_lpips"] = metrics["repair"]["image_pair_lpips"]
            run.log({f"independent_pairs_{name}": wandb.Image(str(experiment / f"independent-50000-{name}-repair/paired-images.png"))})
        artifact = wandb.Artifact("church-neural-repair-independent-evaluation-20260910", type="evaluation")
        artifact.add_file(str(experiment / "independent-50000-comparison.json"))
        run.log_artifact(artifact)
        trained_model = wandb.Artifact("church-neural-repair-trained-final-20260910", type="model")
        trained_model.add_file(str(train / "final.pt"))
        trained_model.add_file(str(train / "config.json"))
        run.log_artifact(trained_model)
        run.finish()
    print(json.dumps({"phase": "independent_evaluation_complete", **result}), flush=True)


if __name__ == "__main__":
    main()
