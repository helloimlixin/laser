# LASER Church stage 2 with the published RQ-Transformer recipe

This experiment trains a fresh prior using the existing frozen LASER compact
tokenizer and the paper's Church optimization settings. It tests the batch and
schedule discrepancy found in the reproduction audit. It is an adaptation to
LASER; the original stage-2 training driver was not released.

The preceding LASER run was checkpointed and paused at epoch 74.7368,
optimizer step 36,909. Its full state is preserved in
`outputs/church-compact-rq-stage2-20260913/train/last.pt`, SHA256
`1306f70e2544142fe5dc5a0b0d4882905d0344d3d812953de8c4e6201ea16e7e`.
The new prior does not load those weights, optimizer state, or preflight weights.

Production run:
https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-rq32k-original-recipe-scratch-20260913

The run uses both H200 GPUs with 512 images per GPU and two accumulation
steps. The production-size preflight completed two updates without an
overflow, peaked at 101.29 GiB allocated per GPU, and passed strict checkpoint
reload plus cached sampling/decoding checks. Artifacts, configuration, and
the launch receipt are under `outputs/church-laser-original-recipe-20260913`.

| Setting | Previous compact run | New experiment |
|---|---|---|
| Global batch | 256 | 2048 |
| Scheduled updates per epoch | 494 | 62 |
| Cosine horizon | 148,200 updates | 18,600 updates / 300 epochs |
| Initial LR | 0.0005 | 0.0005 |
| Additional LR control | FID-triggered halvings | None; published cosine schedule |
| Generation top-k | 250 | 1400 |
| Generation temperature / top-p | 1 / 1 | 1 / 1 |
| Best checkpoint preservation | Only periodic epoch archives | Separate full best-FID and best-validation states |

AdamW uses betas (0.9, 0.95), weight decay 0.0001, gradient clipping 1.0,
and zero warmup. The released 24-spatial / 4-depth architecture, width 1024,
16 heads and residual dropout 0.1 remain unchanged. LASER's larger output
vocabulary makes this model 386,882,561 parameters.

## LASER adaptations retained

The source is the Church-fine-tuned LASER checkpoint
`outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt`,
SHA256 `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`.
The compact shared codebook has 32,769 entries: zero plus 16,384 atoms with
two calibrated coefficient levels each. It produces an 8x8x4 code map.
Each token identifies both atom and coefficient; both condition subsequent
predictions through the released physical codeword embeddings and cumulative
depth context.

The frozen book SHA256 is
`ed74fc4672bb8609d72cbc7797b90119846398d5cb8aa38ef8f8bf4694e68cae`.
The full frozen tokenizer state SHA256 is
`2a141c36fbf1bf1cc808f0b875a49a919764c80ac4504302783fc5f2785873f2`.
Encoder outputs are reused from the verified FP32 cache; Church's released
resize/center-crop transform is deterministic. Soft targets and stochastic
code sequences are recomputed at every visit across the full vocabulary.

LASER retains target temperature 0.125, calibrated to match the relative
residual distortion of RQVAE at 0.5. Copying the raw RQVAE temperature gives
30.1% more residual MSE in LASER; the calibrated value gives 1.31%, versus
1.53% in the original RQVAE calibration control. These are previous calibration
measurements, not a claim of identical target entropy. No new physical
coefficient perturbation is introduced.

## Verification and evaluation

The driver imports the old run's verified immutable source snapshot, including
the pinned released RQ models, rather than mutable shared-workspace adapters.
All 93 snapshot files are checked. Fresh seed-zero initialization matches the
previous experiment's initial weights exactly, with an empty optimizer on
both ranks. The checkpoint hash is recorded separately from tokenizer hashes.

Six existing tokenizer/loss tests pass. The full 18,600-step learning-rate
trajectory matches the analytic cosine. GPU preflights exercise actual
gradient accumulation, optimizer updates, cached sampling, decoding, and
strict model/optimizer checkpoint reloads. Concrete receipts are in
`outputs/church-laser-original-recipe-20260913/verification-*.json`.

Evaluation uses 4096 generated images at epoch 1 and every five epochs;
50,000-image FID is scheduled every 50 epochs. Both use top-k 1400, fixed
seed 71000 + rank, 100 samples per GPU, continuous FP32 decoded pixels,
FP32 Inception and the verified official Church reference. Evaluation RNG is
isolated from training. Latest full best-FID and best-validation checkpoints
are retained independently of the ten-epoch model archives. A run stops after
eight consecutive AMP failures, not a cumulative lifetime skip count.

New FIDs must be compared at the same sample count and sampling settings.
The previous compact run's epoch-50 FID50k of 13.1620 used top-k 250;
changing the sampler prevents attributing any future improvement solely to
training batch size. This launch itself does not establish a quality gain.

Sources: [paper, Appendix A.3 and B.1](https://arxiv.org/html/2203.01941),
[released Church training YAML](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/lsun-church/stage2/lsun-church256-sqgan-8x8x4-350M-simp.yaml).
The paper and checkpoint sampling config take precedence over the discrepant
batch/top-k values in that YAML.
