# Church compound RQ-Transformer with recurrent depth

The user canceled complete-site integers and requested a return to the compound RQ-Transformer with looped computation. This experiment restores four atom–coefficient pairs per latent site. Each atom prediction consumes preceding completed pairs; its coefficient prediction additionally consumes that atom. Both the spatial and depth streams retain coefficients and support together. There is no coefficient-pattern codebook, complete-site vocabulary, or added compression.

The starting architecture is the balanced FFHQ-derived compound model: width 768, 20 spatial blocks, 12 attention heads, a two-block coefficient micro-transformer, and four depth-specific coefficient classifiers. The frozen stage-1 checkpoint is `outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt`, previously adapted to Church. It is not retrained.

| Arm | Depth head | Effective block applications | Total parameters |
|---|---|---:|---:|
| Looped | Two shared blocks, repeated three times | 6 | 190,450,688 |
| Unrolled control | Six independently trainable blocks | 6 | 218,802,176 |

The arms start from the same function: construct the full model with the same seed, copy the first two depth blocks cyclically across the six positions, and retain either shared weights or independent copies. This control uses repeated initialization, unlike the earlier model's independently initialized depth blocks. Forward block count is matched; parameter count and optimizer work differ. This is a weight-sharing comparison, not a claim of equal total training cost.

Each recurrent pass has its own causal KV cache. Only weights are shared. Looping refines the current hidden state before a decision; it does not revise previously sampled pairs. There is no adaptive loop count or untrained increase in inference depth.

## Training and evaluation

Both arms train from scratch using the same 125,203 training images, 1,024 held-out training-source images, and 300 official validation images, with disjoint keys. Fresh pixel crops/flips are encoded by the frozen tokenizer. Coefficients retain the calibrated 2,048 signed scalar bins and depth scales; targets are nearest-bin hard targets. The earlier broad normalized soft-target noise is not restored.

Common settings: batch 128, microbatch 64, dropout .15, AdamW betas (.9, .95), weight decay .05, gradient clipping 1. Atom loss weight is 1.5; coefficient loss weight is 1. Distribution geometry weight .05 starts after epoch two and ramps over three epochs. Learning rate warms to 8e-5 over .5 epoch, decays exponentially to 1e-5 at epoch five, and follows a cosine tail toward 1e-6 on a fixed 60-epoch schedule. The bounded pilot runs at most 12 epochs without restarting that schedule.

Evaluate held-out likelihood and FID-4096 at epochs 1, 2, 4, 6, 8, 10, and 12. Stop after three checks without .01 likelihood improvement or .25 screening-FID improvement, once epoch four is reached. Keep the best screening checkpoint and evaluate it with an independent 4,096-sample seed at termination. Sampling uses atom top-k 2,048, coefficient top-p .5, and temperature 1 for both arms. These screening estimates are not FID-50k claims. Resume preserves optimizer, image order, pending evaluation, and both stopping monitors.

## Verification and provenance

Focused tests cover initial-function equivalence, summed gradients across shared uses, causal use of support and coefficients, target exclusion, valid distinct-support sampling, and full-grid cached/teacher equivalence. Actual GPU checks cover production batch sizes, deterministic pause/resume across GPU and worker changes, all 64 sites in FP32/BF16, and generation/checkpoint-selection lifecycle. Exact outcomes and source hashes are recorded in `outputs/church-looped-pair-20260911/verification.json` before launch.

Production entry points are `src/church_looped_pair.py`, `scripts/train_church_looped_pair.py`, and `scripts/launch_church_looped_pair.py`. The already-pending legacy launcher routes through a compatibility bridge after verification. It starts no integer model; the canceled draft and its smoke artifacts remain under `outputs/church-interleaved-pattern-20260911/abandoned-source` and adjacent verification directories. The old pattern-order pilots were gracefully paused with optimizer/data state; receipt: `outputs/church-interleaved-pattern-20260911/previous-ordering-pause.json`.

Run status and links are recorded in `outputs/church-looped-pair-20260911/launch.json` and `launch-health.json`. No quality improvement is established by smoke tests or successful startup.
