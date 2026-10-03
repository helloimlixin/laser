The existing [Church run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-compound-350m-4a100-20260923)
was recovered on four NVLink-connected NVIDIA H100 80GB GPUs. The first recovery
checkpoint contains step 21,600, epoch index 43, batch cursor 401, optimizer
state, cosine scheduler state, and four per-rank RNG streams. W&B history had
reached step 21,660, so 60 previously logged updates are replayed. The online
`last.pt` available at initial inspection was older, at step 21,300; recovery
therefore uses the newer persistent checkpoint. The named checkpoint files are
symlinks to intact payloads under `train/checkpoints/.checkpoint-data`.

Subsequent artifact investigation found an atom-conditional geometry-loss
mismatch and prepared a compatible continuation from step 26,203. See
[the diagnostic and correction report](church-coefficient-artifacts-2026-09-23.md).
The restart helper honors `resume-runtime.json` when a validated corrected
runtime has been selected; the original source snapshot remains preserved.

The complete source snapshot matches every entry in `runtime-manifest.json`.
Execution uses `/mnt/laser-church/runtime`, preserving the implementation that
started this run. The model has 404,738,048 parameters and retains full
autoregression over atom/coefficient pairs, dictionary atom vector conditioning,
the two-layer coefficient micro-transformer, depth-specific coefficient heads,
normalized stochastic soft coefficient targets, and the distribution geometry
objective. A focused archived-FFHQ comparison passes exact FP32 logits, loss,
and gradient equivalence. Other tests cover pair causality, cached sampling,
resume cursors/schedules, and checkpoint persistence/upload. All 38 selected
tests pass. Two unrelated archived test modules could not collect in this
minimal frozen runtime because one requires omitted archive imports and the
other imports upstream dataclasses incompatible with Python 3.12; neither is
part of the production training import path.

The prebuilt cache contains all 126,227 Church images with 8×8×4 pairs, FP32
coefficients, and per-depth scales `[7.66235399, 4.15803528, 2.63332391, 1.65121174]`.
Each rank loads it into RAM. The tokenizer SHA-256 remains
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
The encoder and sparse pursuit are not rerun during training.

Four-GPU DDP uses batch 64 per GPU, global batch 256, BF16 autocast, SDPA,
TF32 where applicable, and fused AdamW. Model/optimizer tensors stay FP32.
The original 300-epoch, 147,900-update cosine horizon is retained. The initial
resumed LR is 0.00047414473911719574. Initial H100 training windows measured
approximately 1,170–1,195 images/s excluding checkpoint, preview, and FID
overhead; first-update peak allocated memory was 26.30 GiB on each GPU.

The initial launch exposed checkpoint I/O as a bottleneck: 4.86 GB recovery
copies took 34–42 seconds, and saving every 100 updates caused training to wait
for the preceding background copy. A graceful stop saved step 21,835. The
tuned continuation uses the FFHQ reference cadence of every 500 updates and
every five epochs, while retaining all training and evaluation settings.
`last.pt` and `best-fid-01.pt` remain full recovery checkpoints uploaded online;
the best slot changes only when FID improves. Immutable persistent payloads,
local upload-cache hard links, and background periodic copies are retained.

After the graceful tuning stop, both online checkpoint files were verified
against their full local MD5 checksums. The uploaded latest checkpoint is step
21,835 (`22e4c108b1546386eee567e600030f0d`); the best checkpoint is unchanged.
The tuned continuation restored step 21,835 and all four RNG streams.
It subsequently reached step 22,000 with finite loss 6.1741 and gradient norm
0.4900; W&B reports `running` and the intended 500-step/five-epoch save cadence.
Recent compute windows measured 1,177–1,226 images/s. A new step-22,000 recovery
checkpoint committed to persistent storage and was queued for online upload,
and the corresponding preview image was generated successfully. See
`resume-h100-live.json` and `resume-h100-online-confirmed.json` in the run directory
for the live training receipt and confirmed online checkpoint checksums.

FID still uses 50,000 generated images every five epochs, batch 128 per GPU,
the original RQ-VAE Inception implementation, and the same official Church
reference statistics. The feature extractor was downloaded and tested on this
host; it produced finite 2048-dimensional features. The prior best checkpoint
is epoch 30, step 14,790, FID 15.458077954621018. Its online bytes match the
recovered checkpoint MD5 `be9bb1cd4237fc3689053f887d61bc5d`.

The hardware-specific recipe is
`configs/stage2/lsun-church-ffhq-compound-350m-4h100-resume.yaml`.
The resolved execution recipe, launch record, recovery receipts, and validation
receipts are stored under
`outputs/church-ffhq-compound-350m-4a100-20260923/resume-h100-*`.
The credential is read from a private file outside the repository or from the
existing W&B environment; it is absent from recipes and launch records.

Inspect `status.json`, `training.log`, and `resume-h100-supervisor.log` in the
run directory. To restart a stopped run on this four-GPU host:

```bash
python scripts/tools/resume_church_ffhq_h100.py
```

The launcher restores the frozen source/assets as necessary, validates the
latest full checkpoint, stages it locally for fast loading/upload, and launches
a detached supervisor. It refuses to start without an existing recovery state
or while this run already has a live supervisor. It always resumes the newest
committed checkpoint. The machine must remain running for training to continue.
