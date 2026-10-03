LSUN Church unconditional stage 2 is running with the successful FFHQ compound recipe and
the original Church RQ-Transformer architecture and optimization schedule.

Online run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-compound-350m-4a100-20260923

The FFHQ reference is `ffhqcmp0804205803`, with recorded FID50k 8.1743927.
Its actual W&B configuration and the archived implementation were checked;
the archived source SHA-256 is
`9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f`.
The generic `ffhq-compound.yaml` constant-LR settings are not used.

The frozen tokenizer is `tokenizer.pt` from
`helloimlixin-rutgers/laser/church-laser-stage1-selection-20260920-selected-checkpoints:v0`.
It was selected by the earlier matched reconstruction comparison, from the
three-epoch Church fine-tune at learning rate 1e-5. Its SHA-256 is
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
The source stage-1 run reported full-training-population rFID 2.6393384;
that reconstruction score is not a generated FID for this stage-2 run.

Five A100 80 GB GPUs built the complete cache before training. It contains all
126,227 training images, each with 8×8×4 atom/coefficient pairs, using deterministic
resize/center-crop 256 preprocessing. Encoder and OMP computation and stored
coefficients are FP32. A fresh 256-image encoding matched the cached atom IDs
exactly, with zero coefficient error. All labels are zero. Each training rank
materializes the approximately 186 MiB cache in RAM; the frozen encoder and OMP
are not rerun during prior training.

FFHQ-style per-depth scaling maps each full-training coefficient maximum to 3:
`[7.66235399, 4.15803528, 2.63332391, 1.65121174]`. Coefficients are not clipped.
The prior retains normalized soft coefficient targets at temperature 0.5,
stochastic coefficient-token contexts, and 2,048 uniform bins per sparse depth.

Every event predicts an atom given earlier complete pairs, then its coefficient
given that history and the selected atom's frozen dictionary vector. Completed
pairs condition both later spatial sites and later within-site events. The
coefficient conditioner is the FFHQ two-layer causal micro-transformer with four
depth-specific classifiers. Pair embeddings include learned coefficient features,
dictionary vectors, and physical contributions. Atom loss weight is 1.5;
distribution geometry has weight 0.05, top-four candidates, an epoch-two start,
and a three-epoch ramp. Training retains the archived unmasked atom logits;
sampling enforces distinct atoms within a support.

The [original Church YAML](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml)
sets width 1,024, 24 spatial layers, four depth layers, 16 heads, 300 epochs,
global batch 256, and peak LR 5e-4. This run uses four GPUs with batch 64 each,
preserving that global batch without LR scaling. There are 493 updates per epoch
and a 147,900-update cosine horizon to zero. AdamW uses betas (0.9, 0.95),
weight decay 1e-4, gradient clipping at one, and no warmup.

BF16 autocast is explicitly enabled inside the RQ backbone for speed; weights
and optimizer state remain FP32. This fixes the legacy nested autocast scope,
which disabled mixed precision despite the outer BF16 context. The default
behavior for other recipes remains FP32. A CUDA numerical probe confirmed
BF16 atom and coefficient logits, finite gradients, and a maximum logit
difference of 0.005918 from FP32 on the test model. This is a numerical-precision
change from the archived FFHQ execution, with its model and objective preserved.

FID evaluates 50,000 unconditional images every five epochs using the original
RQ-VAE Inception implementation and official `lsun_256_church.npz` statistics.
Atom sampling uses temperature one, top-k 250, top-p one; coefficient sampling
uses temperature one and top-p 0.85. Generation batch is 128 per GPU. A 64-image
preview is logged every 500 updates. No best-FID checkpoint exists before the
first completed evaluation.

Full recovery checkpoints are saved every 100 updates and every epoch. A bounded
background worker persists periodic saves while training continues. This shared
drive could not reliably read a replaced multi-gigabyte file, so this run writes
immutable payload filenames and atomically replaces a small checkpoint symlink.
Completed replacement retires the previous latest payload; best checkpoints
retain their own payload. Copy failure preserves the previous checkpoint.
Online W&B
files are `last.pt` and `best-fid-01.pt`; both retain model, optimizer, scheduler,
epoch/cursor, and per-rank RNG state. Best FID is replaced only on improvement.
Uploads are asynchronous and can lag the newest local checkpoint.

Recipe: `configs/stage2/lsun-church-ffhq-compound-350m-4a100.yaml`.
Persistent directory: `outputs/church-ffhq-compound-350m-4a100-20260923`.
Local execution snapshot: `/mnt/laser-church/runtime`.
The persistent directory retains source hashes, source files, dependencies,
tokenizer/cache provenance, tests, and launch commands. Credentials are stored
privately in `/root/.netrc`, outside the repository and snapshots.

Twenty focused tests passed against the execution snapshot, covering archived
FFHQ logits/loss/gradient parity, full-pair causality, cached autoregressive
generation, checkpoint selection/upload, interrupted persistence, bounded retention,
and reference statistics. The full-size
preflight uses separate disposable weights and activates geometry immediately,
then validates checkpoint recovery and generation before production starts.

Runtime verification passed on all four ranks. The model contains 404,738,048
parameters. The preflight completed two updates with geometry enabled, reloaded
all model/optimizer/scheduler and four RNG states, and generated 128 finite images
per GPU. Peak allocated memory was 23.24 GiB during the first update and 56.30 GiB
during generation. The persisted 4.86 GB preflight recovery state matched its
local serialization by complete SHA-256, confirming the shared-storage fix.

Production began from random stage-2 weights and reached update 270 with finite
loss 8.4424 and gradient norm 0.1108. A ten-update compute window measured
728 images/second; checkpoint, preview, and FID overhead are additional.
The complete update-100 recovery checkpoint was verified online as `last.pt`:
4,857,546,697 bytes, MD5 `e7eb8674c59e81bb42c34bce71376f71`. An authenticated
HTTP range download returned status 206 and the expected PyTorch ZIP header.
See `online-checkpoint-verification.json` for the receipt. Subsequent checkpoints
replace the same online latest slot while training continues.

Inspect `status.json`, `training.log`, and `supervisor.log` in the persistent
directory. Run its `resume.py` to restart a stopped supervisor and restore local
assets from the persistent snapshot. Training resumes only this run's own
checkpoint. The detached processes require the machine to remain running.
