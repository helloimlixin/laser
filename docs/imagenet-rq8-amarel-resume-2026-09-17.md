# ImageNet RQ8 Amarel continuation

## Current recovery — September 18

Job **61690091** requests **8 A100/L40S GPUs (4 nodes × 2)**, partition `gpu`,
constraint `ampere|adalovelace`, 12 CPUs and 96 GiB RAM per node, 72 hours.
It was PENDING for Priority when submitted at 01:35 EDT. Flexible GPU type
probes predicted September 20 at 11:01 EDT; scheduler estimates can change.
Shorter 6/12/24-hour requests did not improve the tested forecasts.

The user clarified that external runs are paid and must not block Amarel
launches. The previous job 61676713 failed at 23:12 EDT on September 17 because
its external-writer guard refused to start. That guard is removed. The Amarel
continuation uses its own linked W&B run
`helloimlixin-rutgers/laser/imagenet-rfid421-rq8-refit-480m-20260913-amarel`
to prevent competing metrics/checkpoint writers. No active training allocation
was cancelled during this repair.

Artifact v91 at step 19100 passed full CPU restore validation: exact tokenizer
state, original train/validation manifests, strict model reload, finite weights
and optimizer, 268 optimizer states, scaler, and scheduler LR
0.00039367945089940074. The six-rank source's portable update cursor is now
mapped correctly to 8/16 ranks; 14 tests passed. A live resolver check selected
newer compatible v92 at step 19404 while the source was running. Allocation
startup resolves the newest reviewed checkpoint, preferring this destination's
own artifacts after its first save. A frozen verified v91 is retained as a
fallback for unreviewed source revisions. Source archives/manifests, original
files, test/preflight evidence, and scheduling probes are in
`RUN_DIR/recovery-20260918` and `compatible-artifacts.json`.

The prepared Amarel fresh-image pipeline is retained; the paid source uses
finite cached image views. The architecture, tokenizer geometry, objective,
global batch 2048, optimizer/scheduler state, and 100-epoch target are retained.

The detached monitor runs on **amarel4**, tmux server `laser-rq8-monitor`,
session `rq8-monitor`, using the same persistent Codex session and updated user
instructions. It preserves the pending allocation and repairs launch failures.
No paid-machine connection is configured; paid compute has not been stopped.
CC3M still needs its original checkpoint/source/cache bundle.

## Original setup record (superseded where stated above)

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-rq8-refit-480m-20260913

Submitted SLURM job **61676713** on September 17, 2026. The selected allocation is
eight L40S GPUs, one per node, on `gpu`, with six CPUs and 96 GiB system RAM per
node and a 72-hour time limit. Eight allocation shapes covering 8/16 GPUs and
A100/L40S were probed. The 8-node, 1-L40S-per-node shape had the earliest estimate.
The job was pending for priority at submission; allocated GPU identity and actual
training progress remain to be verified after it starts.

Launcher: `scripts/submit_imagenet_rq8_refit_resume.sh`.
Prepared run directory:
`/scratch/xl598/runs/laser/imagenet-rfid421-rq8-refit-480m-20260913-amarel`.

The newest recoverable artifact is
`helloimlixin-rutgers/laser/imagenet-rfid421-rq8-refit-480m-20260913-checkpoints:v75`,
step **15,800**, epoch 25, with complete model, AdamW, cosine scheduler, AMP,
rank RNG, tokenizer, and best-FID checkpoint states. W&B history reached step
16,029 in its latest continuation, so resumption from v75 replays 229 updates.
At allocation start the launcher resolves `latest` once for all nodes and
rejects an incompatible source bundle or an already-running W&B run.

The exact frozen tokenizer state, both original dataset manifest hashes,
657,198,081 model parameters, strict model reload, finite model/optimizer tensors,
and scheduler restoration were verified. Restored LR is 0.00042544096344272953.
The released fresh-image training pipeline remains active. Per-GPU batch is 32;
accumulation is eight on 8 GPUs or four on 16 GPUs, preserving global batch 2,048
and the 100-epoch target. Resizing the world uses independent rank RNG seeds.

The recovered source bundle is isolated in the run directory. It has bounded
changes for padded 100-image preview gathers on 8/16 ranks, FID generation batches
of 25 and Inception batches of 32, node-local W&B staging, allocation validation,
and a checkpointed stop ten minutes before the allocation expires. Both existing
samplers, FID sample counts, class grids, and evaluation cadence are retained.
Twelve focused tests passed in the PyTorch 2.4.1/CUDA 12.1 container. CPU checkpoint
and dataset checks passed; a GPU training preflight awaits the allocation.

Receipts include `ready.json`, `cpu-preflight.json`, `source_resume_artifact.json`,
`source-manifest-amarel.json`, `scheduling-probes.json`, and `submission.json`.
Node-local inputs are checked against one resolved artifact digest. Checkpoints
remain on shared storage and immutable artifact uploads stage on `/mnt/scratch`.

Monitor with `squeue -j 61676713`, the run directory's `slurm-61676713.out` and
`slurm-61676713.err`, and `train/status.json`. The live `initialization.json` should
report the resolved resume step and world size; then `stage2/optimizer_step`
should increase in W&B.

## Detached monitoring

On September 17 the user authorized an autonomous Codex monitor to diagnose,
fix, and relaunch launch failures. It runs on `amarel3` in detached tmux session
`rq8-monitor` (server `laser-rq8-monitor`), polling Slurm every minute and resuming
a persistent Codex session for inspections and recovery. Its full instructions,
live heartbeat, status history, logs, and stop instructions are under the run
directory's `monitor/README.md`. It follows replacement jobs and preserves the
same W&B run, global batch, and 8/16 A100/L40S constraints.

## DataLoader startup recovery — September 18, 11:39 EDT

Replacement job **61693182** requests 8 A100/L40S GPUs (4×2), `gpu`, `ampere|adalovelace`, 12 CPUs/96 GiB per node, 72 hours. It is normally pending for Priority. Job 61690091 restored step 31000 (epoch 49) on eight L40S GPUs, but DataLoader workers failed with `AF_UNIX path too long` before any optimizer update. Repeated GPU/process observations confirmed the stall; only that failed allocation was cancelled.

The launcher now uses a short node-local TMPDIR. Socket reproduction and a multiprocessing DataLoader test passed. The destination checkpoint, including optimizer, scheduler, scaler, RNG/cursor, tokenizer and manifests, passed full validation; best checkpoints were copied to shared storage. Because the initial artifact upload remained incomplete, `local-resume.json` points to an immutable validated destination snapshot in `recovery-20260918-afunix/support`. The stager verifies its manifest/source/immutable-input digests and staged bytes, and prefers newer reviewed destination artifacts when available. A real staging test and four negative integrity tests passed. Training source and source archives were unchanged. Backups, validation, preserved checkpoints, scheduling probes, cancellation evidence and receipts are in `RUN_DIR/recovery-20260918-afunix`. Existing initialization/upload records are historical until the replacement starts. Next verify finite loss, optimizer progress and durable destination uploads. Paid source activity did not block recovery and paid billing was not changed.
