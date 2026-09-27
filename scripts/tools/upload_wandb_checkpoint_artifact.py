#!/usr/bin/env python3
"""Upload an immutable checkpoint snapshot as a W&B model artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import wandb


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: object, target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--entity", default="helloimlixin-rutgers")
    parser.add_argument("--project", default="laser")
    parser.add_argument("--uploader-run-id", required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stat = checkpoint.stat()
    print(f"hashing checkpoint {checkpoint}", flush=True)
    digest = sha256_file(checkpoint)
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    epoch = int(payload.get("epoch", -1))
    step = int(payload.get("global_step", -1))
    batch_index = int(payload.get("batch_idx", -1))
    del payload

    aliases = ["latest", "last", f"epoch-{epoch}", f"step-{step}"]
    source_entity, source_project, source_run_id = args.source_run.split("/", 2)
    metadata = {
        "source_run": args.source_run,
        "source_run_url": (
            f"https://wandb.ai/{source_entity}/{source_project}/runs/{source_run_id}"
        ),
        "checkpoint_epoch": epoch,
        "checkpoint_step": step,
        "checkpoint_batch_idx": batch_index,
        "checkpoint_sha256": digest,
        "checkpoint_size_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "aliases": aliases,
        "snapshot_policy": "hardlink from atomically replaced last.pt",
    }
    manifest_path = checkpoint.parent / "wandb_artifact_manifest.json"
    atomic_json(
        {
            **metadata,
            "checkpoint": str(checkpoint),
            "artifact_name": args.artifact_name,
            "status": "prepared_immutable_snapshot",
        },
        manifest_path,
    )

    wb = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.uploader_run_id,
        resume="allow",
        name=f"{args.artifact_name} step {step} upload",
        group=f"{args.source_run.rsplit('/', 1)[-1]}-checkpoint-uploads",
        job_type="checkpoint_upload",
        notes=f"Immutable checkpoint upload for {args.source_run}.",
        config=metadata,
        settings=wandb.Settings(init_timeout=180),
    )
    artifact = wandb.Artifact(
        args.artifact_name,
        type="model",
        description=f"Stage-2 checkpoint from {args.source_run}, epoch {epoch}, step {step}.",
        metadata=metadata,
    )
    artifact.add_file(
        str(checkpoint),
        name="last.pt",
        policy="immutable",
        skip_cache=True,
    )
    artifact.add_file(
        str(manifest_path),
        name="wandb_artifact_manifest.json",
        policy="immutable",
        skip_cache=True,
    )
    logged = wb.log_artifact(artifact, aliases=aliases)
    logged.wait()
    result = {
        **metadata,
        "checkpoint": str(checkpoint),
        "artifact_name": logged.name,
        "artifact_version": logged.version,
        "artifact_qualified_name": f"{args.entity}/{args.project}/{logged.name}",
        "artifact_id": logged.id,
        "uploader_run_id": wb.id,
        "uploader_run_url": wb.url,
        "status": "complete",
    }
    atomic_json(result, manifest_path)
    wb.log({
        "checkpoint/epoch": epoch,
        "checkpoint/global_step": step,
        "checkpoint/size_bytes": stat.st_size,
        "checkpoint/upload_complete": 1,
    })
    wb.finish()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
