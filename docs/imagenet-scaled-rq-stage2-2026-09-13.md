# ImageNet class-conditional scaled-atom RQ stage 2

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-scaled-rq8-480m-20260913

Superseded by the [refitted eight-level run](imagenet-rq8-refit-stage2-2026-09-13.md),
which improves matched rFID-50k from 4.492764 to 4.407900 at the same model size.

The initial workflow was stopped before stage-2 training after the user requested
a reconstruction-FID drift limit of +0.1. Its 6.726 measurement used the ImageNet
training reference and was incorrectly compared with the source run's 4.21
matched-validation rFID. These scores are not directly comparable. A corrected
full matched-validation study is in `outputs/imagenet-tokenizer-fidelity-20260913`.
Artifacts, live status, provenance, and source snapshots are under
`outputs/imagenet-scaled-rq-stage2-20260913`; training output is `train/`.

## Checkpoint and integer vocabulary

The source is `best_rfid_slot1_model.pt` from artifact
`helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints`,
whose epoch-10 version has digest `c1d2d042e6876471ff28556cf1221932`.
The artifact metadata identifies this slot's ImageNet reconstruction FID as
**4.210914134979248**. The downloaded checkpoint SHA256 is
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.

The tokenizer follows
`helloimlixin-rutgers/laser/church-scaled-rq8-scratch-20260913`: a shared expanded
codebook, four residual selections per 8×8 spatial location, fixed preceding
contributions, and a single categorical ID per atom/coefficient pair:

```text
0                         = the zero vector
1 + atom_id * 8 + bin      = levels[bin] * dictionary[:, atom_id]
```

There are 16,384 atoms and eight nonzero signed levels, giving 131,073 classes.
The cache stores these IDs as **uint32**, since they exceed int16/uint16 range.
Both released transformer conditioning paths consume the physical scaled atom
vectors. The source encoder, dictionary, and decoder remain frozen.

Coefficient levels were fitted on 2,048 randomly selected ImageNet training
images, using the same symmetric Lloyd procedure as Church. A disjoint set of
128 training images calibrated the target temperature against the released
ImageNet RQ tokenizer at temperature 0.5. The control's sampled/hard latent MSE
ratio was 1.00891. Temperature **0.125** gave 1.02553 for the expanded tokenizer,
within the control-plus-0.02 limit; 0.25 and 0.5 exceeded it.

The expanded codebook SHA256 is
`904fc3fd5c8f5d96666e8a7da41c959d81701809b7d11552d164a7a6234b800f`.
On 256 held-out validation images, expanded-RQ latent MSE was 0.013202 versus
OMP's 0.009755, a ratio of 1.3533. Expanded reconstruction PSNR was 20.3194 dB.
**The source checkpoint's 4.21 rFID must not be attributed to the converted
tokenizer.** Its full 50,000-image validation reconstruction FID is
**6.726421790752056** against the released ImageNet training reference. This is
**not** matched-validation rFID and does **not** establish the amount of drift
from 4.21. The original report's comparison was incorrect. The corrected
evaluation compares both tokenizers against statistics from the same 50,000
original validation images before any replacement launch is approved.

## Model and recipe

The smallest released ImageNet configuration is
[in256-rqtransformer-8x8x4-480M.yaml](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-480M.yaml).
It uses 12 spatial layers, four depth layers, width 1536, 24 heads, and 1,000
class labels. Expanding its classifier raises the actual parameter count to
**657,198,081**, while keeping the released body and head configuration.

- Fresh seed-zero stage-2 initialization; no pretrained or preflight prior weights.
- 100 epochs, effective batch 2,048, AdamW LR 0.0005, betas (0.9, 0.95), weight
  decay 0.0001, gradient clipping 1.0, cosine schedule, and no warmup.
- Microbatch 128 per GPU, four GPUs, four accumulation steps. Fused AdamW uses
  the released all-parameter grouping. PyTorch SDPA preserves causal attention.
- FP16 model autocast and GradScaler; FP32 tokenizer geometry and chunked exact
  full-vocabulary soft cross entropy. CUDA matmul and cuDNN TF32 are disabled.
- Stochastic RQ targets are recomputed at every visit, including all vocabulary
  probability mass and conditioning every depth on its sampled prefix.
- The released sampling settings are temperature 1.0, top-k 16,384, top-p 0.92.
  Generation labels follow sample index modulo 1,000.
- Validation after epoch one and every two epochs. Fixed-seed 4,096-image FID at
  these checks; additional 50,000-image FID every ten epochs. FID uses the
  released ImageNet training reference statistics.

The released repository did not publish its stage-2 training loop. This driver
uses the pinned released architecture, configuration, scheduler, and sampler,
with the tested local loop and the tokenizer adaptation above. It does not add
Church's FID-triggered LR reductions to the ImageNet cosine schedule.

## Cache and recovery

All 1,281,167 training images and 50,000 validation images retain their official
sorted-WNID class labels. Training uses two independently seeded instances of
the released resize/random-crop/horizontal-flip transform. Validation uses its
center-crop transform. Each epoch selects one stored view per image, alternating
by `(epoch + image_index) % 2`. This finite augmentation pool is the explicit
speed tradeoff; augmentation is not freshly encoded every epoch.

The cache stores FP32 encoder latents alongside uint32 hard pair IDs and int16
class labels. Hard IDs support deterministic evaluation; encoder latents preserve
the original stochastic soft-target objective during training. Total array
storage is about 162 GiB. Ordered rank segments have progress receipts and are
flushed before advancement; incompatible partial caches are rejected.

Production is detached from the terminal. Full model/optimizer/scheduler/scaler
and all-rank RNG checkpoints are saved at step 25, every 100 updates, and every
epoch; model archives are saved every two epochs. Training SIGTERM/SIGINT saves
state at the next completed update. Cache preparation can resume from its flushed
rank progress. Eight consecutive AMP skips stop training with a saved checkpoint.
Failures are recorded in `failure-rank*.json`.

To inspect progress:

```bash
cat outputs/imagenet-scaled-rq-stage2-20260913/train/status.json
tail outputs/imagenet-scaled-rq-stage2-20260913/production.log
```

To resume after the workflow has stopped:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_imagenet_scaled_stage2.py --resume
```

The launcher verifies 68 source files and runs the snapshotted local training
driver. Released dependencies use the already pinned Church upstream snapshot.
Credentials are read from the local `.netrc` file and are absent from source and
launch receipts. The isolated Python environment is locally Git-excluded.

## Verification

Twelve focused tests passed: integer geometry, full-vocabulary targets, exact CE
loss/gradients, released ImageNet settings, SDPA forward/backward parity, class
conditioning, causal masking, cached sampling, and view/label/uint32 handling.
The four-GPU cache smoke built both views, checked every rank segment and finite
values, and successfully reused the completed cache on restart.

The full 657M model completed three production-sized updates, strictly reloaded
its checkpoint, verified finite model and optimizer tensors, and decoded valid
class-conditional samples. Dense and chunked CE measured 8.784004 and 8.784002.
The final two updates processed about 589–591 images/second and peaked at
61.18 GiB allocated per GPU. This is a short repeated-cache benchmark, not a
full-epoch throughput guarantee. Preflight images are integration tests only.

Initial weights SHA256:
`3359ae5e2e37254217cd1e2fc404fe572f8fb9d9f1a4c259e9146b81ff635acb`.
Microbatch 256 encountered a CUDA backward kernel error and was not selected.
The installed NCCL initially failed during collective initialization; the
verified production environment uses `NCCL_NVLS_ENABLE=0`.
