# ImageNet RQ8 continuation with the stage-1-compatible cache

The later [checkpoint-retention update](imagenet-rq8-checkpoint-retention-2026-09-18.md)
adds online latest-plus-best-three retention while keeping this input and
compute configuration.

The same W&B run, `imagenet-rfid421-rq8-refit-480m-20260913`, switches from
online image encoding to cached inputs at optimizer step **16,275**, epoch 26.
The complete checkpoint was preserved before stopping the old trainer during
FID generation. No optimizer updates were lost. The epoch-26 validation had
finished; the interrupted generation is not a completed FID measurement.
Normal validation and generation remain scheduled every two epochs.

The existing complete cache was verified and staged into RAM at
`/dev/shm/laser-rq8-cache-20260918`. Its persistent source remains
`outputs/imagenet-rfid421-rq8-refit-20260913/cache`. The eight arrays occupy
173,879,614,398 bytes (161.94 GiB): two training views for all 1,281,167 images,
one validation view for all 50,000 images, FP32 encoder latents, uint32 token
IDs, and class labels. Every array was checked for its shape, dtype, finite
values, and token/label ranges during staging; SHA-256 digests were recorded.

The dataset and transform files match the source SHA-256 values embedded in
the native 4.2151 rFID stage-1 checkpoint from
`imga16384k4altbn64-b128-b300-20260830000755`. The recovered stage-1 configuration
selects `imagenet256x256`. Loading uses PIL RGB and the released transforms:

- Training: resize the shorter side to 256, random 256×256 crop, horizontal
  flip with probability 0.5, tensor conversion, and normalization to [-1, 1].
- Validation: resize the shorter side to 256, center crop to 256, released
  final resize to 256×256, tensor conversion, and the same normalization.

All image paths, WNIDs, and labels match the saved manifests and torchvision
ImageNet dataset. Across 65 images in each of two training views and one
validation view, all 195 pixel tensors matched exactly. Direct FP32 encoding
differed from cached latents by at most 3.10e-6, and all **49,920** token IDs
matched exactly. The original tokenizer checkpoint, refitted coefficient
levels, and frozen tensor-state identity also match.

The two deterministic crop/flip views use seed `421 + view*10000019 + index`.
Each epoch selects `(epoch + image_index) % 2`. This replaces fresh image
augmentation on every visit with two fixed views. Full-vocabulary stochastic
RQ targets are still regenerated on every visit at temperature 0.0625; hard
cached tokens are used for validation, not as a replacement training objective.
The native stage-1 4.2151 rFID identifies the image pipeline and source
checkpoint. The converted RQ8 tokenizer's prior full-validation rFID is
4.4079; caching does not change that conversion or claim native rFID.

Twelve-update benchmarks on all six H200s, excluding two warm-up updates:

| Input / compute configuration | Images/sec | Seconds/update | Peak rank-0 tensor memory |
| --- | ---: | ---: | ---: |
| Previous fresh-image pipeline | 566.4 | 3.62 | 56.15 GiB |
| RAM cache, original 128-row compute chunks | 902.9 | 2.268 | 56.11 GiB |
| RAM cache, 1,024-row compute chunks | 1,011.3 | 2.025 | 57.14 GiB |

The selected configuration uses two loader workers and four compute threads
per GPU. Effective batch remains exactly 2,048, with three approximately
114-image microbatches per rank. Larger target/soft-CE chunks preserve the
objective: the numerical probe produced identical targets, deterministic
codes, and logit gradients. Both training benchmarks had zero additional AMP
skips. The scheduler, optimizer, scaler, six rank RNG states, global sampler
order, and epoch cursor are restored from the saved checkpoint.

Nine CPU regression checks and the six-process gradient comparison passed.
The full preflight saved step 16,278 with finite model and optimizer tensors.
Production resumed from 16,275, and its step-16,278 model, optimizer, scheduler,
and scaler match that preflight exactly by component SHA-256. W&B confirmed
the same run as active with cached inputs on six GPUs; observed live throughput
was approximately 1,013 images/sec. The step-16,275 backup committed as artifact
version 79, and the first cached checkpoint entered the durable upload queue.

The isolated runtime and audit results are under
`outputs/imagenet-rfid421-rq8-refit-20260913/cache-stage1-20260918/`.
The original fresh-image runtime remains under `recovery-20260918/`.
Production output and the W&B run/artifact collection remain unchanged.
Backups include the cache provenance, stage-1 configuration, runtime manifest,
verification record, and source archive. A surviving detached checkpoint
uploader is allowed to finish before the new trainer starts another upload.

Resume after stopping the trainer with:

```bash
python scripts/tools/launch_imagenet_rq8_h200.py
```

The launcher verifies the recorded runtime, restores a missing RAM cache from
the persistent arrays with digest checks, and resumes from the latest full
checkpoint. As before, the Python environment is `/tmp/laser-imagenet-h200-env`
and must be recreated if the container is replaced. The previous launcher is
preserved as `cache-stage1-20260918/launch-online-images.py` for reference.
