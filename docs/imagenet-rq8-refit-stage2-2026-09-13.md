# ImageNet eight-level coefficient refit and stage-2 run

The user chose eight coefficient levels to reduce model size and requested better
reconstruction fidelity. Refitting the eight values improves matched rFID-50k
from **4.492764 to 4.407900** without increasing the vocabulary or transformer.
The difference from the frozen source tokenizer is **+0.192783**. This does not
meet the preferred +0.1 target. The launch uses a disclosed **+0.20** gate.

Run ID: `imagenet-rfid421-rq8-refit-480m-20260913`.
Output: `outputs/imagenet-rfid421-rq8-refit-20260913`.
Run page: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-rq8-refit-480m-20260913

## Fixed ten-class sampling grids

The live sampler uses the earlier requested **10 rows × 8 samples** layout:
ostrich, bald eagle, lorikeet, Tibetan terrier, snow leopard, teapot, wombat,
red fox, Samoyed, and hotpot, in that order. Class names appear in the left
column and image tiles have no padding. The class IDs are
`9, 22, 90, 200, 289, 849, 106, 277, 258, 926`.

The preview uses fixed seed 20260814 (plus rank), temperature **0.9**, top-k 16384,
and top-p 0.92. Four GPUs each sample 20 images; the output restores the class
row order. It saves the PNG, exact sampled token tensors, and a JSON manifest
with class labels and checkpoint step. The preview preserves training RNG state
and model mode. The original layout helper is copied without changes into a
small preview module to avoid importing the earlier run's separate model stack.

W&B receives the grid under both `generation/class_conditional_samples` and
`samples/requested_classes_10x8`. It runs immediately after the checkpointed
restart and at the existing evaluation points (epoch 1, then every two epochs).
FID sampling covers all 1,000 classes. The original sampler's FID series is
retained, with the selected sampler evaluated and logged separately.

The source revision is retained under `source-revisions/grid10x8-20260913`.
The original preflight proof and source snapshot are preserved separately.
Fifteen focused tests passed, and an AST comparison verified that the optimizer,
scheduler, training, resume logic, and FID sampling population were unchanged.

## Sampling calibration for packed integers

The packed vocabulary has 131,073 entries. The released top-k of 16,384 covered
its original vocabulary but excludes candidates here. On 32 held-out validation
prefixes from the epoch-8 checkpoint, top-k followed by top-p 0.92 retains an
average of 87.50%, 83.47%, 79.90%, and 79.34% of the original probability mass
at depths 0–3. Removing top-k is therefore a meaningful change, but its benefit
must be measured. The numeric integer IDs remain categorical labels.

Eight settings were screened on 1,024 generated images each: the original
joint sampler, full-vocabulary nucleus sampling, temperature 0.9 with different
filters, and atom-first sampling with separate coefficient temperatures.
Atom-first sampling uses log-sum-exp over each atom's coefficient logits and
keeps exactly one zero candidate; without truncation or temperature changes,
the factorization reproduces the joint probabilities. None of the three
atom-first variants beat the original sampler in this screen.

The top candidates were checked on **4,096 images from the same immutable
epoch-8 checkpoint (step 5,008)**, using the same class labels, seeds, batch
sizes, decoder, and training-reference Inception statistics:

| Sampler | Temperature | Top-k | Top-p | Generation FID-4096 |
| --- | ---: | ---: | ---: | ---: |
| Original | 1.0 | 16,384 | 0.92 | 61.715942 |
| **Selected** | **0.9** | **16,384** | **0.92** | **59.232846** |
| Full-vocabulary nucleus | 0.9 | none | 0.92 | 60.353731 |

The baseline reproduces the training run's saved FID within 1e-7. The selected
setting reduces this measured FID by 4.02%. This is an early-checkpoint result,
not evidence that it will remain optimal later. Fixed ten-class grids for all
settings are available in the separate W&B studies:
[screen](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rq8-sampler-screen-epoch8-20260913)
and [confirmation](https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rq8-sampler-confirm-epoch8-20260913).

The live preview selects `original_t09`. Existing `generation/fid_4096` and
`generation/fid_50000` metrics retain the original sampler. Additional
`generation/sampler_fid_4096` and `generation/sampler_fid_50000` metrics use the
selected sampler at the same evaluation points. This adds sampling work while
preserving an interpretable training-progress baseline. Every saved FID JSON
and grid manifest records the actual sampler settings.

The tokenizer, coefficient levels, cached tokens, and training-target
temperature 0.0625 remain fixed; reconstruction rFID is still **4.407920**.
The generation-only source revision is retained under
`source-revisions/sampler-20260913`, with the prior executed source and
checkpoint preserved. Twelve focused tests passed, and an AST comparison
verified unchanged training, optimizer, scheduler, and resume logic. Selection
evidence is saved in `sampler-selection.json`. Future resumes inherit the
selected sampler from the launch receipt unless explicitly overridden.

## Fidelity evidence

All full results use the same 50,000 original ImageNet validation images and
the same original-validation Inception feature statistics. The source result
reproduces the checkpoint's recorded 4.210914 within 0.0043.

| Mapping | Matched rFID-50k | Difference from source | Full model parameters |
| --- | ---: | ---: | ---: |
| Frozen source OMP4 | 4.215117 | — | — |
| Initial eight-level RQ | 4.492764 | +0.277647 | 657,198,081 |
| **Refitted eight-level RQ** | **4.407900** | **+0.192783** | **657,198,081** |
| Sixteen-level RQ, superseded | 4.362995 | +0.147878 | 858,655,745 |

The initial reported 6.726 used the released training-reference statistics and
was incorrectly compared with the source's validation-reference rFID. That
comparison was invalid. The initial workflow stopped while caching, before any
stage-2 optimizer updates. The sixteen-level workflow was tested but never
launched; the user superseded it with the eight-level size constraint.

The new values minimize the actual four-step residual reconstruction error on
2,048 training images. Each iteration assigns greedy RQ tokens, then solves a
four-variable least-squares problem for the shared positive magnitudes. Signs
remain symmetric. No encoder, dictionary, or decoder weights are updated.
The selected positive magnitudes are approximately **1.668196, 2.818533,
4.589749, and 7.571713**.

The screening study compared early and later fitting iterations, prefix-error
fitting, global rescaling, and fitting against the frozen original tokenizer's
decoded images. All level fitting used training images only. The 64-iteration
final-reconstruction fit had the best matched 4,096-image screening rFID and
was then evaluated on all 50,000 validation images. Longer fitting and the
decoded-image objective did not improve the screening score.

Exact candidate tensors and fitting logs are under
`outputs/imagenet-tokenizer-fidelity-20260913/rq8-refit-extended`.
The final full evaluation is under
`outputs/imagenet-tokenizer-fidelity-20260913/full-matched-rq8-refit`.
The production directory's `selection.json` records every screening result,
the selected tensor file, the full result, and the remaining quality difference.

## Recipe and representation

The mapping is `0 = zero`, otherwise
`1 + atom_id * 8 + coefficient_bin`, with **131,073 vocabulary entries**.
Each token decodes to its signed coefficient times the frozen dictionary atom.
Four greedy residual codewords describe each 8×8 latent position. The codebook
is shared across depths and compatible with the Church run's RQ training path.

The model keeps the smallest released ImageNet RQTransformer body: width 1536,
12 spatial layers, four depth layers, 24 heads, and 1,000 class labels. The
expanded output classifier raises the total to 657M parameters, about 201M fewer
than the sixteen-level alternative. The released optimization recipe uses 100
epochs, effective batch 2,048, AdamW LR 0.0005, betas (0.9, 0.95), weight decay
0.0001, clipping 1.0, cosine decay, and no warmup. Sampling uses temperature 1,
top-k 16,384, and top-p 0.92 for the baseline; the selected preview sampler uses
temperature 0.9 with the same filters.

The target temperature **0.0625** is calibrated against the released ImageNet
RQ tokenizer on 128 disjoint training images. Its sampled/hard latent-MSE ratio
is 1.00838, versus the original control's 1.00891. The exact result is saved in
`temperature-calibration.json`. Training regenerates full-vocabulary stochastic
soft targets and sampled-prefix conditioning at every visit.

Two fixed random crop/flip views of each training image are cached, selecting
one per image per epoch. The cache stores FP32 encoder latents and uint32 hard
tokens. Completed encoder-cache segments from the initial workflow are reused
after checking checkpoint, manifests, precision, seed, views, and rank layout.
Hard tokens are regenerated with the new coefficient values. Frozen tokenizer
weights and coefficients then remain fixed throughout stage-2 training.

## Verification and operation

`fidelity-gate.json` binds full matched evidence to exact checkpoint and codebook
hashes. The production driver repeats the matched reconstruction audit after
building the validation cache and requires the +0.20 limit before continuing.
The preferred +0.1 limit remains the default for other invocations.

The launch requires a four-GPU, three-update preflight with strict checkpoint
reload, finite states, class conditioning, sampler decoding, exact chunked soft
cross entropy, and source hashes, plus a separate cache reuse smoke check.
The production model starts from fresh random weights; preflight weights are
never loaded into production. The driver snapshots its source before launch.

All 13 focused tests passed. The new model's three-update preflight passed with
no AMP skips, losses 12.0734 → 10.7375 → 9.6907, strict checkpoint reload,
finite optimizer/model state, correct class conditioning and decoding, and
dense/chunked CE agreeing within 0.000001. The final two updates processed about 593 images/s
with peak allocated memory 61.18 GiB per GPU. This short benchmark does not
include production data-loading or evaluation overhead. Microbatch 128 per GPU
and four accumulation steps retain effective batch 2,048.

The cache smoke check confirmed identical reused encoder latents and class
labels, freshly regenerated valid uint32 tokens, and all four ranks' progress.
The actual fidelity report passes +0.20 and rejects both +0.10 and +0.15.

The live production cache audit reproduced **4.407920 rFID** over all 50,000
validation images, a difference of +0.192803 from the source and only 0.000020
from the independent study. It passed the gate and the workflow advanced to
reusing completed training encoder-cache segments before encoding the remainder.
Both training views have finished caching. The detached four-GPU process is
training stage 2 and reporting to W&B.

## Cache timing and batch-size check

The observed full pipeline processes about 860 image views/s across four H200s.
The two-view cache is projected to take approximately 49 minutes including
validation token conversion, the reconstruction audit, and reuse of earlier
encoder work. Progress and current ETA are recorded in `cache-speed-assessment.json`.

A bounded benchmark briefly paused one cache worker and resumed it in a
`finally` block, with a 45-second resume watchdog. The other three workers kept
running. At FP32 with TF32 disabled, encoding plus quantizing measured 244.8,
248.5, and 266.4 images/s per GPU for batch sizes 64, 128, and 256. Batch 128
matched cached latents exactly; batch 256's maximum latent difference was
0.0000031. Both had identical integer tokens on the 256-image probe.

Batch 256's roughly 9% encoder gain would save only about one minute of the
remaining work before restart, autotuning, and replay overhead. The live cache
therefore retains batch 64. The encoder computation dominates; adding image
workers would not address that work. One complete crop/flip view already exists,
but skipping the second would change the augmentation policy and was not done.

Inspect status and launch evidence:

```bash
cat outputs/imagenet-rfid421-rq8-refit-20260913/launch-outcome.json
cat outputs/imagenet-rfid421-rq8-refit-20260913/train/status.json
tail outputs/imagenet-rfid421-rq8-refit-20260913/production.log
```

Resume after the workflow has stopped:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_imagenet_scaled_stage2.py \
  --base outputs/imagenet-rfid421-rq8-refit-20260913 \
  --run-id imagenet-rfid421-rq8-refit-480m-20260913 \
  --max-rfid-drift 0.20 \
  --reuse-cache outputs/imagenet-scaled-rq-stage2-20260913/cache \
  --sample-grid-on-start \
  --sampler original_t09 \
  --resume
```
