# FFHQ RQ-recipe throughput tuning on two H200 GPUs
Subsequent [precision and validation tuning](ffhq-k4-rqrecipe150-precision-2026-09-26.md) enabled training TF32 convolutions and measured 153.29 images/s. The configuration below describes the preceding FP32 run.


The original RQ-VAE FFHQ split contains **60,000 unique training images and 10,000 validation images**. This was checked by downloading and counting the [training list](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/rqvae/img_datasets/assets/ffhqtrain.txt) and [validation list](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/rqvae/img_datasets/assets/ffhqvalidation.txt). Their hashes match this run's preserved lists. The [released FFHQ stage-1 recipe](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/configs/ffhq/stage1/ffhq256-rqvae-8x8x4.yaml) uses 32 images per GPU; four GPUs give global batch 128. Training lasts 150 epochs.

Selected **64 images per GPU × 2 GPUs × accumulation 1 = global batch 128**, with activation checkpointing disabled. Measured throughput is **66.60 images/s versus 50.07 images/s**, a **33.0% increase**. Allocated memory peaked at 127.46 GB/GPU; reserved memory was 138.35 GB/GPU. FP32, disabled TF32, LR 4e-5, Adam states, scheduler clocks, model architecture, and data transforms are retained.

| Batch/GPU | Accumulation | Activation checkpointing | Images/s | Peak allocated GB/GPU |
| ---: | ---: | :---: | ---: | ---: |
| 32 | 2 | on | 50.07 | 27.19 |
| 32 | 2 | off | 64.59 | 64.61 |
| 64 | 1 | on | 51.44 | 52.61 |
| 96 | 1 | on | 51.58 | 78.04 |
| 128 | 1 | on | 46.62 | 97.91 |
| 64 | 1 | off | 66.60 | 127.46 |

Measurements use both GPUs, real training images, and the complete generator/discriminator/dictionary update. Each candidate loads the same step-17,000 checkpoint, performs four warm-up updates, then measures 16 updates. Startup and cuDNN autotuning are excluded. All six candidates passed finite-parameter/loss, optimizer/scheduler-counter, and dictionary-synchronization checks. Benchmark weights are discarded. These short throughput tests do not establish equal final reconstruction quality.

The discriminator uses ordinary BatchNorm over independent consecutive groups of 32 inside each 64-image device batch. Its state-dict layout and affine parameters remain compatible. CPU tests establish exact outputs, input/affine gradients, and running-state updates versus separately processing reference groups for batches 32/48/64/96/128. The smaller final tail follows ordinary BatchNorm behavior. The larger generator microbatch changes its adaptive GAN-weight calculation, and augmentation worker ordering changes, so this is a recorded batch rebase rather than bitwise continuation.

Production resumes the same [W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-rqrecipe150-20260924) from the latest saved checkpoint: epoch index 36, optimizer step 17,000, checkpoint ID `3b1906dd9a4749d9bc2bf2b70905306c`. The prior process had reached approximately step 17,060; those 60 unsaved updates are replayed. The source checkpoint is preserved in W&B Files as `throughput-source-step17000.pt`.

The resume cursor converts 232 old 32-image microbatches into 116 new 64-image batches, preserving 7,424 already-consumed samples per rank. Partial metric sums/counters are rescaled while code counts are retained. Sampler tests verify the consumed prefix and remaining sample sequence are identical relative to the saved checkpoint. Full 10,000-image rFID/FLIP validation, coefficient maps, inherited best-three checkpoints, and the existing zoom watcher continue.

Runtime: `/tmp/laser-ffhq-k4-rqrecipe150-20260924`. Supervisor: `throughput_supervisor.py`. Live status: `throughput-supervisor-status.json`; logs: `throughput-training-attempt*.log`; restored-state audit: `throughput-resume-audit.json`. The detached supervisor permits four bounded attempts. Recovery checkpoints remain every 200 optimizer updates, with versioned uploads each epoch.

Evidence, source snapshots, configuration, tests, and measurements are under `outputs/ffhq-k4-rqrecipe150-throughput-20260925` and in the original W&B run. Credentials are excluded from these files.
