# Church: exact support and one calibrated joint coefficient pattern

A **2,048-entry joint coefficient codebook passes the existing reconstruction gate** while keeping all four atom IDs exact. Validation reconstruction LPIPS is **0.00912**, PSNR **34.866 dB**, and relative latent MSE **0.00295**, measured against continuous frozen-LASER reconstructions. This is a compression result. Subsequent generation remains poor: FID-4096 is **51.0480 at epoch 10** and **45.0732 at epoch 12**. See the [generation diagnosis](lsun-church-support-pattern-generation-2026-09-11.md).

## Complete-site integer

Each spatial site contains four 14-bit atom IDs and one 11-bit coefficient-pattern ID, for **67 nominal bits per site**. The pattern table recovers all four signed coefficient-bin IDs. Decoding the integer alone therefore determines the complete sparse code; no source support or continuous residual is supplied at decoding time.

The integer is stored as an arbitrary-width Python integer, because it exceeds signed int64. Eight-by-eight grids are serialized as 64 integers plus shape metadata. The original four atom/coefficient pairs require 100 nominal fixed-width bits. Shared dictionary and pattern-table storage is excluded from these counts, and they are not estimates of entropy-coded file size.

Training uses the internal atom and pattern fields. One integer in storage does not imply a 2^67-class softmax or eliminate the need to model the fields. Compared with four separate atom/coefficient pairs, this representation has four atom decisions and **one joint coefficient decision** per site.

## Vocabulary selection

We reserved the same 128 training-calibration images and fitted codebooks on the same 262,144 disjoint training sites as the previous integer probes. Sixteen Lloyd iterations use physical coefficient distances under each selected atom support's Gram matrix. Fitted physical patterns are quantized to the existing signed coefficient bins before assignment and evaluation.

Candidate sizes were declared as 512, 1,024, 2,048, 4,096, and 8,192. The search stopped at the smallest passing candidate, selected on calibration before any new holdout or validation evaluation. The gate remains LPIPS mean plus two standard errors ≤ 0.01 **and** relative latent MSE ≤ 0.005.

| Pattern vocabulary | Complete-site bits | Calibration LPIPS | LPIPS mean + 2 SE | Relative latent MSE | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 512 | 65 | 0.01695 | 0.01761 | 0.00615 | Fail |
| 1,024 | 66 | 0.01210 | 0.01260 | 0.00419 | Fail |
| **2,048** | **67** | **0.00901** | **0.00940** | **0.00298** | **Pass** |

The selected vocabulary gives holdout LPIPS **0.00910** on 256 fixed images and validation LPIPS **0.00912** on all 300 images. It uses 2,045 of 2,048 patterns on validation, with marginal coefficient-pattern entropy 10.843 bits. These usage statistics do not establish autoregressive learnability. All evaluated site integers were independently unpacked and checked against the original atom IDs and selected coefficient-bin IDs.

We additionally tested three fresh crop/flip views of each of the 128 calibration images, using the actual training augmentation seeds and BF16 encoder/FP32 OMP path. Their mean LPIPS is **0.00890**, LPIPS mean plus two standard errors **0.00925**, and relative latent MSE **0.00300**, with zero coefficient clipping. The standard error is computed across 128 images after averaging each image's three correlated views. This check also passes.

Stage 1 remains the existing one-epoch Church finetune from the ImageNet rFID 4.2109 checkpoint. Its SHA-256 is `93f19adc8ca2cfae0cf629d48d2d6f5f7039440b1a48ad1c7ba43cf17932a388`. No neural weights were updated to fit or validate these patterns. The selected pattern artifact SHA-256 is `0a76a13020f80e58264f80d8d0221a52d1590ce55b5fc65039ecc76324a2d0b4`.

## Reconstruction FID measured against original images

The selected 67-bit codec's reconstruction FID is **33.29565** on all **300 official Church validation images**, using the original RQ-VAE Inception backend. Unlike the LPIPS compression gate above, the reference here is the original image population, not continuous LASER reconstructions.

| Reconstruction path | rFID against the same 300 original images |
| --- | ---: |
| Continuous cached sparse codes, BF16 source encoder | 33.29745 |
| Original support with nearest scalar coefficient bins | 33.30422 |
| Exact support with the selected 2,048-pattern codebook | **33.29565** |
| Continuous sparse codes re-encoded in FP32 | 33.17963 |

The pattern-minus-continuous difference is **−0.00179**, so there is essentially no measured rFID change from this additional coefficient compression on this set. That tiny difference is not evidence of improved reconstruction quality. The image keys, transforms, image decoder, and real-image Inception statistics are matched across conditions. All pattern reconstructions were verified against the saved complete-site integers. Decoder and Inception arithmetic are FP32 with TF32 disabled; means/covariances use FP64. An independent low-rank SVD calculation agrees with the standard SciPy covariance-square-root result within 0.000052, and identical real features give exactly zero.

**Correction to the archived 4.90215 reference:** the September 9 Church stage-1 run used **126,227 training images in its loader named `validation`**. Its archived log reports `#train samples: 126227, #valid samples: 126227`, and the upstream `LSUN-church` dataset constructor uses the training LMDB for both loaders. That score was not measured on the 300-image official validation split. Population, sample count, and extraction protocol differ, so 4.90215 and 33.29565 are not directly comparable. Earlier descriptions of 4.90215 as Church validation rFID referred to the run's label without checking the underlying split.

- [Reconstruction FID results](../outputs/church-support-pattern-integer-20260911/reconstruction-fid/results.json).
- [Independent numerical check](../outputs/church-support-pattern-integer-20260911/reconstruction-fid/numerical-verification.json).
- [Archived-protocol audit](../outputs/church-support-pattern-integer-20260911/reconstruction-fid/protocol-audit.json).
- Reproduction: `scripts/evaluate_church_support_pattern_rfid.py`. The saved feature matrices allow recomputing the scores without re-encoding images.

## Stage-2 candidate

The prior uses the existing support-first RQ-Transformer path, with **201,472,512 trainable parameters**: width 768, 20 spatial blocks, six depth blocks, and 12 attention heads. It generates four distinct atoms, then a 2,048-way joint coefficient pattern conditioned on the complete support and preceding spatial sites. Completed spatial embeddings represent the decoded sparse latent. Within-site atom context contains cumulative selected atom vectors and no coefficient-pattern information.

The new Church subclass computes the four cumulative support states by explicit sequential additions. This preserves the intended context while avoiding a CUDA `cumsum` kernel that does not support strict deterministic execution. At sampling time the untruncated distribution is passed as `top_p=None`, avoiding an unnecessary cumulative-probability scan. The original shared trainer and transformer sources remain unchanged, preserving existing runs' resume compatibility.

- Random stage-2 initialization; no failed prior weights are transferred.
- Training population: all 125,203 cached training-image keys, with fresh pixel crops and horizontal flips each epoch. The pattern table was fitted without the 128 calibration images; stage-2 training may use those images after the table is fixed. The separate 1,024-image holdout and 300-image validation splits remain excluded from training.
- Frozen stage-1 encoder, dictionary, bins/scales, and decoder. Pattern targets are the deterministic nearest physical pattern under the selected support Gram metric. No soft-label perturbation is introduced.
- Objective: `(1.5 × sum of four atom NLLs + 4 × joint pattern NLL) / 10`. This gives 60% weight to mean atom NLL and 40% to the joint coefficient decision.
- Effective batch 128, microbatch 64; BF16 transformer and encoder, FP32 OMP/assignments and image decoder; gradient clipping at norm 1.
- AdamW, betas (0.9, 0.95), matrix weight decay 0.05, residual dropout 0.15.
- LR warms for half an epoch to 8e-5, decays exponentially to 1e-5 by epoch 5, then follows a cosine tail to 1e-6 at epoch 60.
- Maximum 60 epochs. Evaluate epoch 1 and every two epochs. Starting at epoch 10, stop if three consecutive checks fail to improve held-out objective by 0.01 **or** FID-4096 by 0.25. Each monitor preserves its cumulative-improvement reference across resume.
- Retain the best measured FID-4096 checkpoint and a separate best held-out checkpoint. Evaluate 50,000 samples every 20 epochs and confirm the selected checkpoint with 50,000 samples using an independent seed at termination.
- Screening samples use atom top-k 2,048, the full joint coefficient-pattern distribution, and temperature 1. Smaller-sample screening FID is not directly comparable with older FID-50k results.
- Save optimizer/data-stream state after the first update, every 250 updates, at epoch/evaluation boundaries, and on graceful pause. Fixed per-update RNG seeds and committed data positions make resume independent of DataLoader prefetch.

## Artifacts and verification

- [Calibration results](../outputs/church-support-pattern-integer-20260911/results.json).
- [Selected coefficient codebook](../outputs/church-support-pattern-integer-20260911/patterns-2048/codebook.pt).
- [Validation reconstructions](../outputs/church-support-pattern-integer-20260911/patterns-2048/validation-reconstructions.png): continuous reference, integer-decoded reconstruction.
- [Fresh-view check](../outputs/church-support-pattern-integer-20260911/augmentation-check.json).
- [Launch verification](../outputs/church-support-pattern-integer-20260911/verification.json).

Implementation: `src/support_pattern_integer_codec.py`, `src/church_support_pattern_training.py`, `scripts/calibrate_church_support_patterns.py`, `scripts/check_church_support_pattern_views.py`, `scripts/train_church_support_pattern.py`, and `scripts/launch_church_support_pattern.py`. Tests cover exact integers beyond int64, signed-bin decoding, geometric assignment, objective weighting and gradients, causal atom/pattern contexts, cached sampling, data-stream resume, and stopping state. GPU smoke checks use the actual 201M-parameter architecture.

All **20 focused tests pass**. An actual four-update GPU run exactly matches a run paused after update two and resumed through update four, including model weights, optimizer tensors, data stream, both stopping monitors, and pending evaluation state. This comparison crosses the epoch/evaluation boundary and changes DataLoader workers from zero to two. A separate negligible-LR run stops at epoch two, retains the epoch-one checkpoint, and completes independent-seed generation from that selected checkpoint. We checked 16,384 generated site integers for exact recovery of support/pattern fields, signed coefficient bins, and distinct atoms. These near-untrained smoke samples establish execution correctness, not generation quality.

The launcher requires the passing verification record and matching codebook/source hashes. The older calibrated hard-target prior was gracefully paused at step **55,001**, epoch **56.181**, and its 2.63 GB optimizer/data-stream checkpoint was saved. Its best and latest checkpoints are preserved. That prior's FID-50k worsened from 29.151 at epoch 25 to 30.079 at epoch 50.

## Production launch

**Paused for the matched ordering experiment at step 14,866, epoch 15.1850.** Model, optimizer, and exact data-stream state were saved and verified. The current comparison is documented in [Church pattern ordering](lsun-church-pattern-order-2026-09-11.md).

The new prior launched at **13:36:45 UTC on September 11**, on GPU 1, PID **40217**, from random stage-2 weights. Its FID-4096 curve at epochs 1, 2, 4, 6, 8, 10, and 12 is **119.313, 82.944, 61.835, 57.296, 54.164, 51.048, and 45.073**. It is still improving, but the inspected images have substantial structural defects. The epoch-10 frozen diagnosis finds weak later-atom predictions and coefficient-history drift even with correct support. This does not establish that the training curve has plateaued. Existing training and stopping rules continue; the diagnostic does not update neural weights.

- [W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-support-pattern2048-201m-20260911).
- [Launch health](../outputs/church-support-pattern-integer-20260911/launch-health.json).
- Training directory: `outputs/church-support-pattern-integer-20260911/train`.
- Durable log: `outputs/church-support-pattern-integer-20260911/train.log`.
- The calibrated soft-target baseline was also gracefully paused, at step 58,272, epoch 59.5224, to provide the second GPU for the matched ordering comparison.

The frozen epoch-10 sampling diagnosis improves FID-4096 from **51.0480 to 44.0590** by changing only joint-pattern nucleus sampling from the full distribution to **p=0.5**, retaining atom top-k 2048 and temperature 1. Structural defects remain in the images. `scripts/sample_church_support_pattern.py` provides this setting for frozen checkpoints and saves independently verified complete-site integers. The live trainer's sampling settings are unchanged, preserving the comparability of its existing curve. See the [full generation diagnosis](lsun-church-support-pattern-generation-2026-09-11.md).
