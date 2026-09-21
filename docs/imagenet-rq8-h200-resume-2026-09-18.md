# ImageNet RQ8 continuation on six H200s

The later [verified token-cache continuation](imagenet-rq8-token-cache-2026-09-18.md)
supersedes the fresh-image input mode below at step 16,275. The same launcher
now resumes that faster cached configuration.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-rq8-refit-480m-20260913

The newest committed W&B checkpoint bundle was recovered from
`imagenet-rfid421-rq8-refit-480m-20260913-checkpoints:latest`, at optimizer
step **15,800**, epoch 25, 150 completed global updates into that epoch.
The previous process reported step 16,029 before stopping; those final 229
updates were not in the latest committed backup. The original local checkpoint
was substantially older, at step 9,202, and was preserved.

The recovered bundle includes full model, AdamW, cosine scheduler, AMP scaler,
rank RNG states, four best FID checkpoints, exact tokenizer and coefficients,
and its training source. Tokenizer state SHA-256 remains
`64535e9a0774a1cd2ceb208930398e8623109f1fa2cafbac75a80cbe5f555f95`.
The continuation uses the latest run's released ImageNet loader with fresh
random augmentations, FP32 encoding and targets, FP16 transformer computation,
temperature 0.0625, and the original 100-epoch training target.

Six NVLink-connected H200s are available. The container has approximately
61 CPU cores of quota, so training uses six loader workers and four compute
threads per rank. The effective batch remains exactly **2,048**. Each update
assigns 342 images to two ranks and 341 to four ranks, in three approximately
114-image microbatches. Gradients are weighted by actual image count before
DDP averaging. The source sampler's one-image epoch padding and shuffled global
batch order are preserved, including the cursor partway through the epoch.
New checkpoints store a portable global-update cursor.

The stable six-GPU benchmark completed six successful optimizer updates at a
median **566.4 images/second**, including fresh image loading and encoding,
with peak tensor allocation **56.15 GiB** on rank zero. A larger 171-image
microbatch failed during the large classifier's CUDA backward pass on both
tested PyTorch versions and is not used. The restored AMP scale backed off
once on the first six-GPU update, then remained stable. Hardware resizing
changes the random streams and floating-point trajectory.

The isolated environment uses PyTorch 2.9.1 / torchvision 0.24.1 with CUDA 12.8,
matching the checkpoint's PyTorch version with CUDA suitable for this machine.
The original project environment is retained. Installation commands are from
the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/).
Source is isolated under
`outputs/imagenet-rfid421-rq8-refit-20260913/recovery-20260918/runtime/`;
the downloaded original source is retained separately in `recovered-source/`.

Validation partitions cover all 50,000 images exactly once on six ranks.
Generation retains 4,000/50,000 samples, the original and temperature-0.9
samplers, original single-image decoding and file-based FID. The 100-image
preview supports uneven rank partitions. Seven CPU regression tests and a
six-process gradient comparison passed, including a short final batch.
The FID Inception checkpoint and evaluation dependencies load successfully.
The full-model preflight also saved and reloaded the portable six-rank state,
verified finite model and optimizer tensors, and generated the complete 10×10
preview. Production's checkpoint at step 15,803 matches the preflight model,
optimizer, scheduler, and scaler exactly.

Production output is
`outputs/imagenet-rfid421-rq8-refit-20260913/train-h200x6-20260918/`.
Its `status.json`, `training.log`, `checkpoints/last.pt`, and
`checkpoint-upload.json` record progress and committed backups. Latest
checkpoints are saved every 100 successful updates, after the first three,
at epoch boundaries, and on graceful termination. Best FID states and support
files remain included in the existing W&B artifact collection.
Production launched detached as torchrun PID 7259. W&B confirmed the same run
as running on six GPUs, and observed production throughput was approximately
565 images/second with all six GPUs at 100% utilization during updates.

Resume after the process has stopped:

```bash
python scripts/tools/launch_imagenet_rq8_h200.py
```

The launcher verifies the isolated source and recorded preflight, reads the
private W&B credential, and starts a detached six-GPU process in the same W&B
run. It rejects a duplicate active launch. The local environment is at
`/tmp/laser-imagenet-h200-env`; its packages and source verification are recorded
under `recovery-20260918`. Recreate this environment if the container is replaced.
