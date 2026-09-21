# Fresh Church transformer with adaptive learning rate

The user authorized a fresh stage-2 run after the requested previous run's
transformer checkpoint could not be recovered. The frozen tokenizer is still
the exact three-epoch Church finetune from that previous run.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3ep-scratch-adaptive-lr-20260916

The transformer has 386,882,561 parameters, random seed 0, and a new AdamW
optimizer. It loads no pretrained transformer or preflight weights. Both GPUs
verify identical initialization. The frozen tokenizer SHA-256 is
`1dbf4519a5c6248ec7f0ab3865511371a0370d247e3c5fc1bffb15492f476624`,
and its codebook SHA-256 is
`b7a4bac5f80da85a1ff341e6782d66439a2fff0bcd88b35aca529919fe4843dd`.

Settings:

- Two H200 GPUs; batch 256 per GPU; four accumulation steps; global batch 2,048.
- 300 epochs, 62 optimizer updates per epoch. FP16 model autocast, FP32 codebook
  geometry and cross entropy, TF32 disabled.
- Initial effective learning rate **2.5e-4**, implemented as a saved 0.5
  multiplier on the original 5e-4 cosine schedule. FID50k is evaluated at epoch
  1 and every 10 epochs. Two evaluations without at least 0.1 FID improvement
  halve the multiplier, with one evaluation of cooldown and a 1e-6 floor.
  Scheduler and controller state are checkpointed together. A quality gain
  from these changes remains unproven at launch.
- The original RQVAE LSUN loader's RGB, bilinear resize to short side 256,
  center crop, and normalization to [-1,1]. The 126,227 training keys, 300
  validation keys, and 72 pixel probes match the original receipts. The FP32
  latent cache hash matches; stochastic targets are recomputed each visit at
  the saved temperature 0.125. All 93 frozen dependency files are verified.
- FID50k uses all 126,227 unique training images as its real reference, with
  the released Inception implementation. Sampler temperature 1, top-k 1,400,
  top-p 1. Seeds and reference remain fixed for comparable LR decisions.
- Full resumable `last.pt` is saved/uploaded at update 25, after every epoch,
  and on graceful shutdown. Each W&B artifact includes the latest checkpoint
  and up to three best evaluated checkpoints, ranked by the same FID50k
  protocol. These checkpoints contain optimizer, scheduler, scaler, per-rank
  RNG, data cursor, and LR controller state. The first three ranked slots fill
  as evaluations finish. Immutable files remain pinned until upload commits;
  superseded local FID files are pruned afterward.
- A fixed-seed 100-image **10 × 10** preview runs every **200 successful
  optimizer steps**, with training RNG preserved. AMP-skipped updates do not
  trigger duplicate previews. FID diagnostic grids also use 10 × 10 images.
  This corrects the initial mistaken interval of 200 epochs; see
  `preview-step-correction/` in the output directory for the change receipts.

The cadence correction resumed this same run from its full step-621 checkpoint.
The restored learning rate was `0.00024931302825363265`, with multiplier 0.5.
An immediate 100-image grid was generated at step 621, and scheduled previews
continue at steps 800, 1000, 1200, and so on. Seven continuation tests passed,
including the optimizer-step cadence and preview RNG checks. W&B records
`preview_every_steps: 200` and clears the former epoch interval.

Before launch, the actual two-GPU full-batch preflight completed two fresh
updates, restored that checkpoint, produced a 100-image grid with the real
sampler/decoder, and completed update three. All updates were finite with zero
AMP skips; peak training allocation was approximately 54.88 GiB per GPU. The
released Inception backend produced finite 2,048-dimensional features in a
separate check. Nineteen continuation/recovery regression tests passed.

The production runtime is frozen under
`outputs/church-ft3ep-scratch-adaptive-20260916/runtime/`. Its source manifest,
launch receipt, logs, preflight results, and production status are in the same
parent directory. Production files are in `train/`; the committed checkpoint
upload receipt is `train/checkpoint-upload.json` once the first upload finishes.
See `verification.json` for the observed production startup and W&B status.

The initial fresh launch command was:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_church_ft3ep_continuation.py --from-scratch
```

An interruption must resume this new run from its own full `last.pt` using the
frozen driver with `--resume` and its new run ID, retaining the same batch size,
cache, and calibration. Do not repeat the fresh launch command to resume.
The launcher provides that operation as:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_church_ft3ep_continuation.py --resume-fresh-run
```
