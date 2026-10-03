# Church throughput: RQ paper optimization, September 24, 2026

The repaired Church run reached FID 11.6910, but its effective batch of 128
limited training throughput. Continue its best checkpoint using the original
RQ paper's optimization settings while retaining the validated LASER coefficient
objective, representation, tokenizer, cache, and sampler.

## Reference and chosen settings

The [RQ paper, Appendix C](https://arxiv.org/html/2203.01941v2#A3)
specifies global batch 2048, learning rate 0.0005, AdamW betas (0.9, 0.95),
weight decay 0.0001, gradient clipping at 1, and a 300-epoch cosine schedule
without warmup. The [released Church configuration](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml)
instead specifies batch 256, with the same learning rate and epoch count.
This run uses the published paper's optimization settings.

Four H100 80GB GPUs each process microbatches of 128, with four accumulation
steps: 128 × 4 × 4 = 2048 images per optimizer update. The unchanged trainer
uses fused AdamW, BF16, DDP, and compiled coefficient pair attention. The full
prebuilt FP32 token cache stays in RAM.

## Measured throughput

Two steady-state windows per setting used all four GPUs and the actual model,
coefficient target construction, backward pass, DDP, clipping, and optimizer.
The benchmark used learning rate zero to preserve source model weights and
excluded preview generation, FID, checkpoint persistence, and logging.

| Setting | Batch per GPU | Accumulation | Global batch | Mean images/s | Peak allocated GiB/GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous repaired run | 32 | 1 | 128 | 1,410.8 | 14.14 |
| Released YAML batch | 64 | 1 | 256 | 1,847.8 | 21.03 |
| Paper batch, selected | 128 | 4 | 2048 | 2,263.6 | 36.33 |

The selected layout improves compute throughput by approximately 60%.
At 61 complete optimizer updates per epoch, compute alone takes approximately
55 seconds per epoch. FID and checkpoint time are additional. The raw samples,
script, and logs are preserved in the run's `audit` directory.

Production verification through global step 5140 measured 2,285.3 images/s across
19 steady windows, with all four GPUs at 99–100% utilization in the recorded
sample. This corresponds to 54.7 seconds of training compute per epoch. Losses
and gradients were finite, and every logged learning rate matched the restarted
18,300-update cosine schedule. Startup and the step-5000 preview/save window
are excluded from this steady-state rate.

## Continuation and schedule

The source is the full epoch-5, step-4930 checkpoint of
`church-compound-ffhq-adapt-4h100-20260924`, with FID 11.691015814491152.
Optimizer state and all four RNG streams are retained. An explicit schedule
restart identifier starts a new 300-additional-epoch cosine phase at 0.0005:
18,300 optimizer updates, ending at epoch 305. The global step remains monotonic;
the new scheduler step starts at zero. Reloading a checkpoint from this phase
continues its saved scheduler instead of restarting again.

The prior run's final step-8146 checkpoint and best checkpoint remain preserved,
with online W&B file checksums verified. The new run owns a separate copy of the
best checkpoint and its best-checkpoint paths are rebased to its own directory.
The first recovery save exposed a shared bootstrap payload behind the new
run's initial best/last links. The best checkpoint was restored from the intact
source into an independent payload, and both retained paths were verified.
The step-5000 recovery includes the optimizer, four RNG streams, and scheduler
step 70 of 18,300. Its successful save did not interrupt the training process.

Full last and best-FID checkpoints continue uploading online. Periodic recovery
saves occur every 1,000 optimizer updates; previews every 500; 50,000-sample FID
every five epochs; regular epoch checkpoints every ten epochs. Changing the global
batch reduces how often recovery saves and previews interrupt an epoch.

## Preserved LASER adaptations

- Full autoregression over both atom IDs and sparse coefficients, with dictionary
  atom-vector conditioning and the existing two-layer pair transformer.
- Validated Church stage-1 checkpoint (recorded reconstruction FID 2.6393) and
  the cache encoded from original LSUN Church images.
- Physical coefficient targets, temperature 0.125, atom loss weight 1.5,
  geometry loss disabled.
- Atom top-k 250 and coefficient top-p 0.85; the unchanged original RQ FID
  evaluator and reference statistics.

These are adaptations to LASER's atom/coefficient representation, so this is not
a reproduction of the original model's discrete residual-code objective.
The source FID 11.6910 predates this optimization change; quality under the new
schedule must be measured by subsequent evaluations. The first 50,000-image
evaluation after five additional epochs (epoch 10, global step 5235) measured
FID **12.7802**. This confirms the throughput gain has not yet produced a quality
improvement. The 11.6910 best remains retained; training continues under the
requested paper optimization schedule.

## Validation and recovery

The unchanged frozen runtime passed 41 focused tests. A 12-update four-GPU
training preflight at the production batch and learning rate produced finite
model and optimizer tensors and saved all four RNG streams. Generation from
the reloaded checkpoint passed at 128 images per GPU, peaking at 56.32 GiB
allocated and 59.03 GiB reserved per GPU. The reloaded scheduler correctly
continued at update 12 of 18,300.

Run directory: `outputs/church-rq-paper-opt-20260924`.
Configuration: `configs/stage2/lsun-church-rqopt-calibrated-4h100.yaml`.
Recovery entry point: `outputs/church-rq-paper-opt-20260924/resume.py`.
Runtime manifest, source archive, configuration, assets, preflight receipts,
benchmark, and checkpoint handoff are persisted in the run directory.
The retained best and step-5000 recovery checkpoint were both verified online
by size and MD5. Subsequent checkpoints, including epoch 10, upload asynchronously.

W&B: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-compound-rqopt-b2048-4h100-20260924
