# Church control: released frozen tokenizer, fresh original transformer

This control tests whether the local stage-2 training implementation can recover
the quality of the original Church model when tokenizer fine-tuning is removed
as a variable. It uses the released Church RQ-VAE checkpoint directly, with no
LASER components or tokenizer updates. The existing LASER run continues alongside
it on the two H200 GPUs.

W&B run:
https://wandb.ai/helloimlixin-rutgers/laser/runs/church-original-rqvae-released-tokenizer-control-20260917

## Recipe

| Item | Control |
|---|---|
| Tokenizer | Frozen released Church RQ-VAE; 8×8×4 codes; shared 16,384-entry codebook |
| Tokenizer SHA-256 | `ba008ec2e192a6d4084a8fd511927a789c68a8459d6e8bfc22122ae02887b800` |
| Transformer | Original 370,087,936 parameters; 24 spatial + 4 depth layers, width 1,024 |
| Initialization | Fresh weights, seed 0, released normal-0.02 initializer; empty optimizer |
| Training population | All 126,227 Church training images; original RQVAE LSUN loader |
| Preprocessing | PIL RGB, bilinear short-side resize to 256, center crop, normalize to [-1,1] |
| Cache | FP32 continuous encoder latents; stochastic codes and soft targets regenerated each visit |
| Loss | Released soft-target cross-entropy; native residual stochastic sampling, temperature 0.5 |
| Global batch | 2,048 = 2 GPUs × 128 images × 8 accumulation steps |
| Duration | 300 epochs; 62 update attempts per epoch; 18,600 planned successful updates if none are skipped |
| Optimizer | AdamW, LR 5e-4, betas (0.9, 0.95), weight decay 1e-4, gradient clipping 1 |
| Schedule | Original cosine to zero over 18,600 successful updates; no adaptive FID multiplier |
| Regularization | Residual dropout 0.1; embedding/attention-probability dropout 0; stochastic soft targets |
| Precision | FP16 transformer autocast, FP32 tokenizer and cross-entropy, TF32 disabled |
| Preview | 100 images in a 10×10 grid every 200 successful optimizer steps; fixed seed 71000 |
| Sampling | Released Church settings: temperature 1, top-k 1,400, top-p 1 |
| Validation | All 300 official held-out images, epoch 1 and every 5 epochs |
| FID | Exactly 50,000 generated images, epoch 1 and every 10 epochs; seed 71000 + rank |
| FID reference | Same fresh statistics from all 126,227 unique training images used by current LASER |
| Checkpoints | Full resumable latest plus best three FID50k checkpoints; W&B upload at steps 1 and 25, each completed epoch, and graceful shutdown |

The distributed training sampler follows the existing convention and pads one
sample to make the rank lengths equal. The final accumulation group is weighted
by its actual image count. Cache construction and FID real statistics contain
each image once, without padding. AMP-skipped updates do not advance the
scheduler or trigger previews.

## Scope of reproduction

The architecture exactly matches the released checkpoint configuration after
applying upstream defaults. The paper specifies a stage-2 global batch of 2,048,
whereas the checked-in Church YAML specifies 256. This control follows the paper
for batch size and follows the released checkpoint for sampling parameters.

KakaoBrain did not release the stage-2 training loop. This is a local driver using
the released architecture, quantizer, stochastic targets, loss, initialization
helper, optimizer helper, scheduler, and cached sampler. In particular, the
original unpublished optimizer parameter grouping and initializer call site
cannot be independently recovered from weight-only checkpoints. This run is a
controlled reproduction attempt, not a claim that every unpublished detail is
identical.

The tokenizer checkpoint's training duration is not inferred from its config.
Its weights are used unchanged. Existing matched reconstruction tests already
verified this released tokenizer on all 300 held-out Church images.

## Verification and operation

The runtime is copied from the audited frozen source and has its own manifest
under `outputs/church-released-tokenizer-control-20260917/runtime/`. Startup
checks source hashes, tokenizer/cache hashes, exact architecture, dataset size,
and the shared FID reference. Cache preparation checks all 72 recorded pixel
probes from the original data loader and verifies tokenizer state immutability.

Preflight trains the full-size model on real data at global batch 2,048 for three
updates, saves update two, and resumes it in a separate process to repeat update
three. It also exercises held-out validation, Inception, the actual sampler,
and a 100-image preview. Production starts from freshly initialized weights;
preflight weights are never used for production. Final preflight results and
source hashes are recorded in `verification.json` in the output root.

The full preflight passed with zero AMP skips and peak GPU allocation of
36.292 GiB. Resuming update two reproduced update three exactly across all
1,844 model/optimizer/RNG tensors and the remaining checkpoint fields, including
scheduler, scaler, and data cursor. Both paths produced identical validation
results on all 300 held-out images. The 100-image preview and finite 2,048-dimensional
Inception features were verified. The related CPU regression suites passed all
32 tests. Checkpoint deserialization uses the frozen upstream package because
the optimizer metadata includes its configuration types.

Every retained checkpoint includes model, optimizer, scheduler, AMP scaler,
per-rank RNG states, deterministic data cursor, tokenizer/cache identity, and
FID ranking. A pending-evaluation marker permits recovery at an epoch boundary.
Checkpoint files remain pinned until W&B confirms the upload, and superseded
local FID states are removed only after the new artifact commits. The best-three
list fills as FID evaluations finish; it contains no unevaluated placeholders.

Launch after verification:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_church_released_control.py
```

Resume a stopped production process:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_church_released_control.py --resume
```

The launcher reuses the private existing W&B credential without putting it in
source, arguments, launch receipts, or documentation. Production progress is in
`train/status.json`, committed uploads in `train/checkpoint-upload.json`, and
logs in `training.log` under the output root.

Production was launched fresh on 2026-09-17. W&B confirmed its first full
checkpoint artifact as `COMMITTED`, containing `last.pt` (4,441,574,698 bytes)
and `selection.json`. Training advanced after that upload with zero AMP skips.
The remote check and startup state are recorded in `production-verification.json`.
No FID-ranked slots were populated at launch because the first FID evaluation
had not occurred yet.

Sources: [paper training details](https://arxiv.org/html/2203.01941#A3),
[official repository](https://github.com/kakaobrain/rq-vae-transformer),
[earlier retraining audit](church-published-reproduction-audit-2026-09-13.md), and
[matched stage-1 comparison](church-stage1-isolation-2026-09-17.md).
