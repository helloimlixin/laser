The ImageNet continuation uses the existing [W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-no-geometry-scratch-20260924). This trains the 1.45B-parameter stage-2 autoregressive prior. The stage-1 tokenizer is frozen; there is no discriminator optimizer to rescale.

The throughput revision resumes the durable checkpoint at update **25,259**, preserving all 837 AdamW states and the cosine scheduler. Training uses **252 images per GPU × 4 H200s × 2 accumulation steps = effective batch 2,016**. Compared with the previous four-GPU launch, physical batch increases 50%; compared with the original seven-H100 launch, it increases from 96 to 252 per GPU. Effective batch is unchanged, so peak LR 0.0004921875, AdamW betas (0.9, 0.95), weight decay 0.0001, and the 63,500-update cosine schedule remain intact. The per-rank epoch cursor maps from 1,482 to 988 without skipping or replaying samples at this restart. Hardware layout and compilation change the stochastic trajectory; this is not a bitwise-equivalent continuation.

Profiling found that the tiled FlashAttention kernel was inefficient for the six four-token depth-attention layers. Its backward operation also caused the previous illegal-memory-access failure at batch 252. The replacement computes the same causal softmax with FP32 accumulation. Transformer block training and the compound objective are compiled; autoregressive cached sampling retains its eager path. Parameter names and checkpoint structure are preserved. Completed microbatch logits are released after backward, and DDP uses gradient bucket views.

| Four-GPU configuration | Effective batch | Warm training images/sec | Result |
|---|---:|---:|---|
| Previous implementation, 168/GPU × 3 accumulation | 2,016 | ~1,112 | Baseline |
| Optimized, 252/GPU × 2 accumulation | 2,016 | **1,721** | Selected; 30 resumed updates with diagnostics |
| Optimized, 252/GPU × 4 accumulation | 4,032 | 1,744 | Only 1.28% faster |
| Optimized, 336/GPU × 3 accumulation | 4,032 | — | Out of memory in the full distributed loop |

The selected configuration is about **55% faster** than the previous launch. Peak allocated CUDA memory during the diagnostic-enabled probe was 117.19 GiB per GPU, with 120.95 GiB reserved. A standalone single-GPU test fit batch 336 at 125.00 GiB and 464.64 images/sec, versus 458.42 images/sec at batch 252; it did not represent the full distributed trainer's memory requirement. The full 336/GPU test exhausted memory during backward. Larger effective-batch probes used a constant restored current LR strictly to compare execution performance; their updates were discarded. Their small speed gain does not justify changing the established optimization schedule.

The source recovery initially used the run's uploaded `last.pt`, verified against W&B MD5 `180e961f77ad28b1fd60a13805675599`, at update 25,000, epoch index 39, cursor 705. The prior run had logged through 25,400, so that initial recovery replayed 400 updates. The later 25,259 checkpoint was saved durably before stopping the slower four-GPU run. Original source and best checkpoints remain preserved. Best historical FID50k is 34.38792896 at epoch 35. The original 100-epoch budget, FID50k every five epochs, and generation sampler remain in effect.

The [Church feature comparison](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-released-rq-vs-laser-features-20260925) motivated an ImageNet coefficient-noise audit. Exact finite-bin moments were measured at one random site from 16,384 training images with the actual tokenizer dictionary and 2,048 coefficient bins:

| Coefficient teacher | Added noise energy / clean latent energy | Relative covariance change | Physical standard deviation |
|---|---:|---:|---:|
| Original physical T=0.125 | 0.30558% | 0.24917% | 0.25 at every depth |
| Continued physical T=0.03125 | 0.07639% | 0.06229% | 0.125 at every depth |
| Hypothetical normalized T=0.25 | 15.00792% | 13.63705% | Depth-dependent |

Both sampled coefficient inputs and soft cross-entropy targets use physical temperature **0.03125**. This reduces added teacher noise energy fourfold. Deterministic cached OMP supports, coefficient scales, bins, and generation sampling are retained. This is a conservative objective correction; improved generated-image covariance, energy, or FID has not yet been demonstrated.

Validation includes 16 tests passing against both the edited repository and recovered production runtime: causal outputs and gradients, FP32/BF16 compiled block behavior, strict checkpoint keys, sampling fallbacks, and compiled objective gradients. Three four-GPU performance probes completed 30 actual resumed optimizer updates each with finite losses and gradients. The earlier continuation also passed 19 targeted noise and checkpoint/optimizer/scheduler/RNG/data-cursor tests.

Persistent files are in `outputs/imagenet-rfid421-resume-4h200-20260925/`. The `throughput-optimization/` subdirectory contains benchmark logs, configurations, tests, the scoped source patch, and a SHA256-identified runtime archive. The active runtime is `/mnt/laser-imagenet-resume/throughput-optimization/runtime`; the training log is `/mnt/laser-imagenet-resume/training.log`. Full checkpoints are staged locally and copied asynchronously to persistent storage using immutable files. The interval is now 200 updates, allowing the measured 118–139-second background copies to finish before the next save. The supervisor retries failures at most twice from the latest durable checkpoint. Configuration and audit evidence are attached to the original W&B run.

Live verification after relaunch: the original W&B run reports `running`, and training reached update **25,400** with finite gradients. Across all 14 warm ten-update intervals it sustained **1,711.63 images/sec** (latest interval 1,709.32), 53.9% above 1,112. The new 17,496,902,141-byte durable recovery file was loaded and checked: 837 model tensors, 837 optimizer states, four RNG streams, epoch index 39/cursor 1,270, scheduler position 25,400/63,500, LR 0.000322140900959458, batch 252 × 4 × 2, and physical noise T=0.03125. The scheduled epoch-40 FID evaluation has started. A checkpoint-copy wait at the epoch boundary is excluded from the warm training throughput; evaluation and checkpoint serialization also add wall-clock overhead.
