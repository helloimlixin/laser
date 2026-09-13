# Compact joint sparse-code RQTransformer on LSUN Church

The 32,769-token experiment is running from fresh random initialization on both
H200 GPUs. Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-compact-rq32k-scratch-20260913

Artifacts are under `outputs/church-compact-rq-stage2-20260913`. The larger
131,073-token run was gracefully paused at optimizer update 58,833, epoch
119.1356. Its complete model, optimizer, scheduler, scaler, and both ranks' RNG
states remain in its `train/last.pt`; the pause receipt includes its file hash.
The compact run loads neither that checkpoint nor its own preflight weights.

## Model and frozen tokenizer

- Vocabulary: 32,769 joint atom/coefficient choices. Each image retains
  8×8×4 = 256 tokens. Token zero is the unique zero vector; token
  `1 + atom * 2 + bin` selects `dictionary[:, atom] * levels[atom, bin]`.
- The two coefficient values are specific to each atom: one negative and one
  positive. They were fitted offline before this training experiment. Residual
  quantization never refits earlier selected contributions.
- Released Church architecture: 24 spatial layers, four depth layers, width
  1024, 16 attention heads. Total parameters: **386,882,561**. Both spatial and
  depth conditioning use the fixed codeword vectors, preserving dependence on
  both support and coefficient.
- The source is the frozen LASER Church sparse checkpoint
  `outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt`,
  SHA256 `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
  This is the sparse-tokenizer branch, distinct from the original RQVAE branch.
- Codebook: `outputs/church-compact-scaled-rq-20260913/adaptive2-8passes/compact-codebook.pt`,
  SHA256 `ed74fc4672bb8609d72cbc7797b90119846398d5cb8aa38ef8f8bf4694e68cae`.
- Initial stage-2 state SHA256:
  `ca0c5887b5e5b760a1a743e4742fa5184fc4e4558d35376ca18869eb9469084c`.
  Seed zero, identical on both ranks, empty optimizer. Production matches the
  preflight's initial state, rather than its trained state.

The matched 4,096-image reconstruction screen measured rFID **8.3923** versus
**8.5846** for the 131k vocabulary. PSNR decreased by 0.1126 dB. This screening
result does not establish generation quality or replace a 50k evaluation.

## Target calibration and cache provenance

Calibration uses the same 128 Church training images, indices 62,048–62,175,
and random seeds as the earlier temperature experiment. These are disjoint from
the coefficient fitting indices 64,000–68,095. The measured original RQ control
is reused from the earlier report, with its report and checkpoint hashes recorded.

| Tokenizer / training temperature | Sampled / hard latent reconstruction MSE |
| --- | ---: |
| Original Church RQ, 0.5, previously measured control | 1.0153084 |
| Compact 32k, 0.5 | 1.3012919 |
| Compact 32k, 0.25 | 1.0643316 |
| Compact 32k, **0.125** | **1.0130783** |

Selection retains the prior rule: the largest tested temperature at most 0.5
whose ratio is within 0.02 of the original RQ control. This controls relative
reconstruction distortion; it does not assert identical noise distributions.
There is no additive coefficient noise. Full-vocabulary stochastic RQ targets
are regenerated each visit, conditioning each residual depth on the sampled
prefix. Training target temperature is separate from generation temperature.

The verified FP32 encoder cache from the larger sparse run is reused for all
126,227 images. Its SHA256 is
`cad69321547c726feebedac84f9d9c2a231b54f85e4066884393bc1ec08b9312`.
It stores unquantized encoder outputs, so the changed book does not require
recaching. The new manifest records the original cache manifest, original
tokenizer identity, unchanged encoder checkpoint, and new frozen tokenizer
state separately. Both official validation shards are hash-checked links to the
same 300 validation latents. The new complete tokenizer state SHA256 is
`2a141c36fbf1bf1cc808f0b875a49a919764c80ac4504302783fc5f2785873f2`.

## Training and evaluation

The released Church optimization settings are retained for comparison with the
131k experiment: AdamW, initial LR 0.0005, betas (0.9, 0.95), weight decay
0.0001, gradient clipping 1.0, 300-epoch cosine schedule, no warmup. Batch size
is 128 per GPU, 256 globally, with 494 updates per epoch. Model computation uses
FP16 autocast and GradScaler; tokenizer geometry and loss use FP32 with TF32
disabled. Chunking soft targets and cross entropy retains the full vocabulary.

The existing FID learning-rate controller is retained. It halves the LR
multiplier after two consecutive 4,096-sample FIDs exceed the best by 0.25, or
three checks fail to improve by 0.1, with ten epochs between reductions. The
multiplier is applied once to the underlying cosine schedule.

All 300 validation images and fixed-seed 4,096-sample generation FID run after
epoch one, then every five epochs. A 50,000-sample FID runs every 50 epochs.
Generation uses the released cached sampler: temperature 1.0, top-k 250, top-p
1.0. Checkpoints contain full training state at update 25 and every epoch;
model archives are saved every ten epochs. Eight consecutive AMP skips save a
checkpoint and report failure.

## Verification

Thirteen relevant tests passed, including explicit expanded-book equivalence,
original RQ soft targets and cumulative commitment, causal dependence on both
atom and coefficient, exact chunked loss/gradient, LR control, and partial-batch
weighting. The production-size two-GPU preflight completed three updates with
finite gradients and zero AMP skips. Peak training allocation was 30.226 GiB
per GPU versus 56.103 GiB for the larger vocabulary at the same batch size.

The preflight checkpoint reloaded strictly; all model and optimizer tensors
were finite, all 460 optimizer entries and both RNG states were present.
Full-model chunked CE and the released dense CE both measured 10.0966196.
The released cached sampler produced valid token IDs and finite decoded images.
Those three-update images test integration, not generation quality.

The launcher verified and snapshotted 93 source files, including the pinned
released architecture at commit `341395e562ac347f5eb62db9f5f08b9f2cc42a60`.
The released repository does not contain its stage-2 training loop; this uses
the verified local driver and the released model, optimizer, scheduler, and
sampler. Existing run source files remain unchanged. The production audit is
recorded in `startup-verification.json`.
