# ImageNet LASER VAR, 2026-09-14

**Stopped and superseded.** This was a pretrained-tokenizer adaptation, which
did not satisfy the requested from-scratch comparison. Its reconstruction
results are not evidence for scratch training. The active experiment is the
[matched scratch VQ/LASER comparison](imagenet-var-scratch.md). Preserve these
artifacts as an audit record; do not resume this run for that comparison.

This experiment replaces VAR's vector lookup with the repository's LASER
dictionary learning and OMP solver at every spatial scale. Production artifacts
are under `outputs/imagenet-laser-var-f16-20260914`. Training uses the public
`train.py` entry point and
[`configs/experiments/imagenet-laser-var.yaml`](../configs/experiments/imagenet-laser-var.yaml).
The W&B run is
[imagenet-laser-var-f16-k2-d16-20260914](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-laser-var-f16-k2-d16-20260914).

## Comparison and downsampling choice

**Choose spatial factor 16 at 256×256.** This gives the same 16×16 final latent
grid, 680 total spatial sites, and ten next-scale steps as released VAR. The
scale sizes are `1,2,3,4,5,6,8,10,13,16`. Factor 32 would give only 8×8 final
sites and introduce an additional compression change; factor 8 would quadruple
the final grid's sites and change transformer cost substantially. This choice
controls spatial layout, rather than claiming a measured optimum over factors.

The direct baseline is the released **VAR-d16: 310M parameters, FID 3.55,
200 epochs, effective batch 768**. The released larger d30 reports 1.97, or
1.80 with rejection sampling. Use the unrejected d16 result for the primary
comparison. These are established VAR references, not the current overall
leaderboard. [Released models, recipe, and evaluation instructions](https://github.com/FoundationVision/VAR).

For broader context, REPA-E reports 1.12 guided FID, while AdvFD reports 0.72
for JiT-H with distribution-based post-training that includes Inception features.
Their supervision, training costs, and evaluation qualifications differ from
this autoregressive experiment. A win over VAR-d16 would not establish overall
SOTA. [REPA-E](https://arxiv.org/abs/2504.10483),
[AdvFD](https://arxiv.org/html/2608.11205v1).

| Quantity at 256×256 | Released VAR | LASER VAR |
| --- | ---: | ---: |
| Final spatial grid | 16×16 | 16×16 |
| Transformer positions | 680 | 680 |
| Dictionary vectors | 4,096 | 4,096 normalized atoms |
| Sparse terms per site | 1 VQ ID | 2 atom/coefficient pairs |
| Nominal fixed-width bits/image | 8,160 | 28,560 |
| Prior parameters | about 310M | 311,665,153 |

LASER uses 257 signed, asinh-companded scalar levels including exact zero.
Its nominal budget is `680 × 2 × (12 atom bits + 9 coefficient bits)`.
Actual entropy-coded rate is unmeasured. This is a comparison at matching
spatial layout and almost matching prior size; it is **not a bitrate-matched
compression improvement**. Two small depth/head decisions per scale also add
sampling work, which must be reported when comparing generation speed.

## Model and training

The encoder, decoder, and residual convolutions initialize from released
`vae_ch160v4096z32.pth`. Its normalized embedding vectors initialize LASER's
dictionary; the embedding table is then removed. LASER OMP selects two distinct
atoms per site and refits their coefficients within each scale. Earlier scales
stay fixed. Scalar levels are calibrated on 512 randomly selected training
images, with a 20% range margin, then fixed. Both reconstruction training and
prior targets use the same discretized coefficients. Dictionary parameters and
residual convolutions learn from the multiscale fixed-code reconstruction loss;
the encoder receives commitment and straight-through reconstruction gradients.

The transformer is the pinned official FoundationVision VAR body. At each site,
its output boundary factorizes the probability as
`p(a1|history) p(c1|history,a1) p(a2|history,a1,c1) p(c2|history,a1,c1,a2)`.
Its coefficient and depth context projections use the physical dictionary
vectors. Duplicate atoms are masked in both training and sampling. Current-scale
target codes never enter the spatial transformer's inputs: those contain only
the accumulated reconstruction of earlier scales.

The tokenizer adapts for five ImageNet epochs with L1 + LPIPS reconstruction,
multiscale dictionary/commitment losses, and a PatchGAN hinge objective starting
at update 1,000. The backbone LR is 2e-5 and dictionary LR 1e-4. The effective
batch is 128: microbatch 32 on four GPUs. The discriminator uses one update per
generator update once active, with weight 0.1 in the generator objective.
Before starting the 200-epoch prior, the final matched 50k rFID must be at most
2.0 and at most 1.0 worse than the released VQ tokenizer on the same images.
If this quality check fails, the tokenizer checkpoint is retained and the
pipeline stops with `tokenizer_quality_rejected`, releasing the GPUs. The
initial unadapted replacement has poor reconstruction quality, so spending the
full prior budget without checking the trained tokenizer would be premature.

The fresh prior trains for 200 epochs with effective batch 768: microbatch 96
on four GPUs and accumulation 2. It retains the released d16 architecture,
AdamW betas (0.9, 0.95), LR 3e-4, four warmup epochs, `lin0` schedule with final
LR ratio 0.1, weight decay 0.05, and gradient clipping 1.0. Both stages use BF16
autocast; OMP, coefficient arithmetic, dictionary losses, and cross entropy are
FP32. TF32 is disabled. BF16 and the extra tokenizer adaptation are departures
from the released prior's FP16 recipe and must accompany reported comparisons.

All 1,281,167 training images and 50,000 validation images retain the official
sorted WNID labels. Training encodes fresh epoch-specific 288px Lanczos
resize/256px random crops, without horizontal flips, matching VAR's default
augmentation. Validation uses the corresponding center crop. The loader drops
the incomplete final effective batch, so no padded or partial optimizer update
changes batch size. There is no finite token-view cache.

## Evaluation and recovery

Matched reconstruction FID compares reconstructed and original validation
images using identical preprocessing. The initial 2,048-image diagnostic and
final 50,000-image evaluation also reconstruct the released VQ tokenizer on
those same images. Intermediate tokenizer evaluations use 2,048 images.
Reconstruction FID is not generated-image FID or a mathematical lower bound on it.

The prior logs held-out atom, coefficient, and joint NLL each epoch. Generation
uses CFG 1.5, atom top-k 900, top-p 0.96, no rejection or smoothing. Labels are
sample index modulo 1,000. A 4,096-image PyTorch FID diagnostic runs after epoch
one and every five epochs. Every 20 epochs and at the end, 50,000 uint8 images
(exactly 50 per class) are exported to an NPZ and scored with the unmodified
official ADM evaluator and `VIRTUAL_imagenet256_labeled.npz`. Its FID, sFID,
Inception Score, precision, and recall log under `adm/` in W&B. The diagnostic
PyTorch score is separately named and cannot substitute for official metrics.

Checkpoints save model/optimizer state, discriminator state when relevant, and
all-rank Python/NumPy/CPU/CUDA RNG states after update one, every 100 updates,
and each epoch. SIGTERM/SIGINT to a worker requests a checkpoint at the next
completed update; all ranks agree before exiting. The prior verifies its source
tokenizer checkpoint hash on resume. Changing that tokenizer requires a fresh
prior. The production source snapshot and command are recorded in `launch.json`.
Credentials are read from a mode-0600 file outside the repository and are absent
from configs, source snapshots, receipts, and logs.

```bash
cat outputs/imagenet-laser-var-f16-20260914/status.json
tail -f outputs/imagenet-laser-var-f16-20260914/production.log
```

The normal foreground launch/resume command is:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NCCL_NVLS_ENABLE=0 \
  .venv-imagenet-stage2/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --config configs/experiments/imagenet-laser-var.yaml
```

For an exact production restart after the previous process has exited, use the
`command`, `cwd`, and environment overrides in `launch.json`; they point to the
source snapshot. Do not start a second writer while the original run is active.

## Dependencies and verification

Install the official source at its pinned revision:

```bash
git clone https://github.com/FoundationVision/VAR.git third_party/FoundationVision_VAR
git -C third_party/FoundationVision_VAR checkout 78b95394fc5896192e3a003e4b295f8ea743c48f
python -m pip install -r requirements-var.txt
```

The production environment is the existing PyTorch 2.4.1+cu124 / torchvision
0.19.1 environment with the added Hydra and Hugging Face dependencies. Official
ADM evaluation uses a separate `.venv-var-eval` with TensorFlow CPU 2.17.1,
NumPy 1.23.5, and SciPy 1.14.1. The NumPy pin preserves the unmodified official
evaluator's use of `np.bool`. Its Inception feature extraction warmup passed.

Six focused tests cover quantized-code round trips, encoder/dictionary gradients,
scalar vocabulary integrity, scale and depth causality, joint NLL, reproducible
sampling/KV cleanup, and public recipe dispatch. Full four-GPU smoke runs cover
both stages, including active adversarial updates, checkpoint writes, validation,
and decoding. At the selected batch sizes the third updates measured about
226 tokenizer images/s and 413 prior images/s, peaking at 61.15 and 70.65 GiB
per GPU respectively. These are short benchmarks, not sustained throughput.

A broader test invocation passed 23 tests and exposed two pre-existing archived
CLI expectations (`ddpm` versus the archive's `rqvae`) plus a missing optional
Lightning dependency in this isolated environment. Those failures do not involve
the new backend and were not changed as part of this experiment.

No trained generation result or SOTA claim is available at launch.

## Launch observation

Production is detached under launcher PID 7850, using the 228-file source
snapshot recorded in `source-manifest.json`. An initial startup port collision
was resolved by selecting an explicit unused local rendezvous port; its log is
retained separately. All four H200s are active. Through update 30, the tokenizer
loss decreased from 1.2059 to 0.4374, at roughly 232 images/s after startup.
These are early training losses, not generation-quality results.

Before adaptation, the 2,048-image matched reconstruction diagnostic was
124.51 for LASER versus 9.58 for the released VQ tokenizer on exactly the same
images. These small-sample values are not comparable to published full-50k
rFID. Their large difference is the reason for checking final tokenizer quality
before spending the full prior budget.

The prior checkpoint resumed from update three through update four with its
optimizer and all four RNG states, then validated and sampled successfully.
Changing the tokenizer checkpoint correctly rejected prior recovery. Both saved
models and optimizer tensors were checked for finite values. The final focused
invocation passed 19 tests. The ADM evaluator completed all five metrics on an
identical-array smoke case, with near-zero numerical FID/sFID and precision and
recall equal to one; this is an evaluator check, not a model score.
