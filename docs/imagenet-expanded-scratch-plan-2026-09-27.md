# Queued Stage-2 scratch experiment

Status: authorized and queued. The user instructed: ‘ok do this after current run’.
A detached completion watcher is running and will launch the scratch experiment
automatically after the current run completes successfully.

The current continuation introduced the expanded teacher at optimizer step
57,151, shortly after epoch 90. A scratch experiment would apply this teacher
from the first optimizer update, removing the earlier teacher changes from the
training history. This does not establish that scratch training will improve
generation FID.

## Prepared configuration

Recipe: `configs/stage2/imagenet-rfid421-combination-expanded-scratch.yaml`.
Teacher snapshot: `configs/stage2/imagenet-rfid421-combination-expanded-targets.json`.

- Randomly initialize the same Stage-2 architecture and create fresh Adam state,
  optimizer step count, RNG trajectory, and learning-rate schedule. Both
  checkpoint-resume and weights-only Stage-2 initialization are disabled.
- Keep the frozen rFID 4.21 Stage-1 tokenizer, coefficient scales, architecture,
  training seed, fresh image augmentation, and expanded support teacher.
- Retain eight H200s, per-GPU batch 252, global batch 2,016, accumulation one,
  635 optimizer updates per epoch, and 63,500 updates over 100 epochs.
- Restart the configured two-epoch warmup and 100-epoch cosine schedule from
  step zero. Retain FID50k every five epochs, previews every 200 updates, and
  recoverable checkpoints every 200 updates. Skip the untrained startup preview.
- Use separate scratch output/checkpoint directories and the separate
  W&B run ID `imagenet-rfid421-combination-expanded-scratch-next`.

At the measured approximately 1,367 images/s, 63,500 updates take approximately
26 hours of training alone, or 208 H200 GPU-hours. FID evaluation, previews,
checkpointing, and startup add time. This is a measured-throughput extrapolation,
not a monetary quote or a guaranteed wall-clock duration.

## Automatic handoff

An isolated frozen runtime, audited scratch entry point, and recovery supervisor
are installed under `/mnt/laser-imagenet-combination-expanded-scratch-next`.
The completion watcher is `queue_after_current.py` in that directory.

The watcher verifies that the current job completed its
63,500 updates and final FID evaluation, that its final/best checkpoints are
durable, and that the proposed scratch directories are unused. Scratch-local W&B, staging,
and upload-cache paths are configured. The launcher preserves the validated eight-GPU runtime settings,
including BF16, compiled attention/objective, and expandable CUDA allocations.

At startup, the entry point audits `initial_global_step == 0`,
`initial_optimizer_state_count == 0`, and the new warmup schedule. The supervisor resumes only the
new scratch run's own recovery checkpoints after any subsequent interruption.
A one-shot start marker prevents duplicate random initializations. If the parent
fails or stops before completion, the watcher continues waiting. No scratch
training or new W&B run starts until the completion gate passes.

Queue state: `outputs/imagenet-rfid421-combination-expanded-scratch-next/queue-status.json`.
Watcher PID at arming: 96802. Ten completion/duplication guard
tests passed, all 346 frozen runtime files and both frozen assets passed their
checksum checks, and native config parsing plus the fresh warmup start were
verified without launching GPU training. The final parent checkpoint will be
checked for all finite model/Adam values and matching local/persistent SHA-256
checksums before the handoff. All eight GPUs must be released first.

To cancel this queued scratch job before handoff, create
`/mnt/laser-imagenet-combination-expanded-scratch-next/cancel.request`. This leaves
the current training run alone.
