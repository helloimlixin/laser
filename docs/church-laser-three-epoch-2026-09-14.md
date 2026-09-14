# Three-epoch Church LASER fine-tune followed by fresh stage 2

The user requested three tokenizer fine-tuning epochs followed by stage-2
training after the previous prior plateaued. The batch-2048 prior's FID50k
was 13.8845 at epoch 50 and 14.2560 at epoch 100. Its best FID4096 was
16.0568 at epoch 60, while validation soft CE rose from 7.9553 at epoch 15
to 11.7565 at epoch 100. The prior was checkpointed and paused at epoch
106.5968, step 6609; its last and best checkpoints remain preserved.

The new pipeline starts from the original ImageNet LASER checkpoint with
reported ImageNet rFID 4.210914. It performs **three total Church epochs**,
starting with fresh optimizers, then uses the epoch-three tokenizer. It does
not add three epochs to the existing one-epoch Church checkpoint. Stage 2
starts from random weights and an empty optimizer after all preparation.

Stage 1: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3ep-official-20260914

Stage 2, automatically created after fine-tuning/preparation:
https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3ep-rq32k-scratch-20260914

## Fine-tuning settings and source

Both generator and discriminator use Adam, LR 4e-6, betas (0.5, 0.9), no
weight decay, no warmup, and constant LR. Global batch is 128, with 64 images
per H200. There are 987 updates per epoch and 2961 total updates. The loss
uses the original MSE reconstruction term, latent weight 0.25, LPIPS weight
1.0, and hinge discriminator / vanilla generator loss with adaptive
adversarial weighting and discriminator weight 0.75. Training is FP32 with
TF32 disabled. Released activation checkpointing is enabled to bound memory.

LASER-specific settings retain the previously successful 16,384-atom,
four-coefficient bottleneck, progressive commitment, alternating dictionary
updates, and local-64 discriminator BatchNorm scope. Thus this is the official
Church optimization/GAN recipe applied to LASER, not an RQVAE tokenizer.

The paper describes one Church fine-tuning epoch; the released Church
checkpoint configuration lists three. The duration here follows the user's
explicit three-epoch request. Both sources agree on LR 4e-6.

The recorded successful Church source was recovered using root commit
`2ae6b2ec30ee2c66c871f03154f5fb41a13ac059`, RQVAE submodule base
`54c4bd9c1e0546f542f3bc6fc6401f9ddf8a7c4a`, and the successful one-epoch
W&B run's recorded patch. Applying the patch succeeds, and `main_stage1.py`
matches the W&B code file byte-for-byte. The current shared workspace code
is not used for fine-tuning.

The historical patch omitted untracked helper files. The required local-64
BatchNorm helper was restored as an identity that retains the released
BatchNorm modules; six recorded scope tests pass. The checkpoint helper was
recovered from an existing archived source. Unavailable coefficient-heatmap
logging is disabled; the original epoch reconstruction grids remain enabled.
These details and hashes are recorded in `source-recovery/supplemental-files.json`.
An initial smoke import failure is retained separately; it performed no
training updates. These supplements prevent a claim that the entire historical
source tree was recovered byte-for-byte.

## Automatic stage transition

`scripts/tools/run_church_laser_three_epoch_pipeline.py` supervises:

1. Three fine-tuning epochs with reconstruction evaluation and full GAN
   checkpoints. The released LSUN factory evaluates the training population;
   these reconstruction scores are not held-out validation results.
2. A tensor-only export of the completed third-epoch tokenizer. Full generator,
   discriminator, optimizer, scheduler and both ranks' RNG states remain saved.
3. Fresh FP32 encoder latents for all 126,227 training images and all 300 official
   validation images. Old latents and codebook levels are not reused.
4. A newly fitted 32,769-entry compact shared book: zero plus two coefficient
   levels for each atom. Fitting uses training indices 64000–68095, eight
   passes and prior weight 4. Directions come from the newly trained dictionary.
5. New stochastic-target calibration using training indices 62048–62175 and
   the preserved original-RQVAE relative-distortion control. The previous
   LASER temperature 0.125 is not imposed on the new geometry.
6. A matched 4096-image reconstruction screen and a full-batch stage-2
   preflight, including sampling, decoding and strict checkpoint reload.
7. A fresh 386,882,561-parameter RQ-Transformer: 24+4 layers, width 1024,
   global batch 2048, AdamW at 5e-4, 300-epoch cosine, top-k 1400, temperature
   1 and top-p 1. Best-FID and best-validation full states are retained.

Each token jointly identifies atom and coefficient, and both condition later
predictions. Stage 2 checks the new tokenizer, cache and book hashes. Preflight
weights are never used to initialize production. The supervisor records each
child PID, command, phase and failure; it stops the chain if a phase fails.

## Verification and result limits

The stage-1 smoke completed two full-batch generator/discriminator updates,
reconstruction evaluation, checkpoint export and strict model/discriminator/
optimizer reload. Both optimizers held LR 4e-6. It peaked at 54.72 GiB allocated
per GPU. A separate preparation smoke exercised new encoder extraction,
all 300 validation latents, codebook fitting, calibration and reconstruction.
Smoke populations and FIDs are explicitly diagnostic and are not quality
measurements. All production sources/configurations are fingerprinted before
launch and verified at each phase transition.

Three-epoch quality improvement is not established at launch. Pipeline status,
per-stage logs, source recovery and verification receipts live under
`outputs/church-laser-three-epoch-20260914`.

Sources: [paper training details](https://arxiv.org/html/2203.01941#A3),
[official repository and released checkpoint](https://github.com/kakaobrain/rq-vae-transformer).
