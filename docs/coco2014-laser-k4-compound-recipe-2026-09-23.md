# COCO2014 LASER K4 → compound RQ-Transformer text-to-image

This recipe launches a fresh LASER tokenizer (stage 1), selects its best
validation reconstruction-FID checkpoint, freezes it, builds aligned image/text
token caches, and trains a fresh text-conditioned compound prior (stage 2).
The detached supervisor runs the handoff automatically on four A100 80 GB GPUs.

The compound reference is [FFHQ run ffhqcmp0804205803](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803),
whose recorded FID-50k against 70,000 real FFHQ images is **8.174392700195312**.
Its live W&B configuration and summary are preserved in this run's
`assets/ffhq-reference-run.json`. That value is provenance, not a COCO result.

## Runs and files

- [Stage 1 W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/coco2014-laser-k4-stage1-20260923)
- [Stage 2 W&B](https://wandb.ai/helloimlixin-rutgers/laser/runs/coco2014-laser-k4-compound-stage2-20260923)
- Stage 1 configuration: `configs/stage1/coco2014-laser-k4-4a100.yaml`.
- Stage 2 configuration: `configs/stage2/coco2014-ffhq-compound-k4-4a100.yaml`.
- Persistent directory: `outputs/coco2014-laser-k4-compound-20260923`.
- Runtime snapshot: that directory's `runtime/`, copied to `/mnt/laser-coco/runtime`.
- Environment: `/opt/laser-coco-venv`; PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128.
- Authentication: private `/root/.netrc`, excluded from recipes and source snapshots.

## Dataset and transforms

Use the official COCO2014 split, not COCO2017 or the Karpathy split.
Train2014 has 82,783 unique images and 414,113 caption annotations; val2014
has 40,504 images and 202,654 captions. The dataset's verification receipt
records archive SHA-256 checks, every JPEG's CRC and decoding, caption linkage,
and disjoint image IDs. Its copy is `assets/coco2014-READY.json`.
Persistent data: `/workspace/Projects/data/coco`; local ZIP-backed copy:
`/mnt/image-datasets/coco2014`.

All image paths use PIL bicubic resize of the shorter side to 256, center crop
256×256, RGB, and normalization to [-1,1]. No random crops or horizontal flips.
Stage 1 visits each training image once per epoch. Cache extraction encodes each
unique image once in FP32; stage 2 expands those codes to **all** human training
captions. Caption IDs and image IDs never cross the official split boundary.
Evaluation generates one image per validation image, conditioned on its lowest-ID
caption. The other validation captions do not enter training or evaluation.

## Stage 1: fresh LASER

100 epochs, seed 0, four GPUs, 32 images/GPU, global batch 128, BF16 mixed
precision. The RQ-VAE encoder/decoder has width 128, channel multipliers
[1,1,2,2,4,4], two residual blocks, attention at resolution 16 plus mid attention,
256 latent channels, and learned 1×1 quant/post-quant projections. Its sparse
bottleneck has 16,384 normalized dictionary atoms, embedding dimension 256,
four OMP supports per spatial site, and an 8×8 latent grid. Coefficients are
unclipped. Data initialization seeds the dictionary from the first batch.

Losses: MSE 1, LPIPS 1, encoder commitment 0.25, dictionary fitting 0.25,
coefficient L1 penalty 0. PatchGAN has two layers, adaptive weight 0.75, and
starts after 5,000 generator updates. Adam uses learning rate 4e-5 for
autoencoder, dictionary, and discriminator; betas (0.5,0.9). LR warmup is 500
updates followed by a constant LR; gradient norm is clipped at 1.

Validation reconstruction FID uses all 40,504 val2014 images every epoch and
the existing TorchMetrics/torch-fidelity 2048-feature backend. The count is
divisible by four, so distributed evaluation does not pad images. The selected
checkpoint minimizes `val/rfid`. Stage-2 generation FID uses the original
RQ-VAE backend below; do not equate their numerical values.

## Stage 2: FFHQ compound recipe with text conditioning

Retain the reference's 350M backbone geometry: width 1024, 24 spatial layers,
four depth layers, 16 attention heads. The added text and compound heads raise
the actual parameter count; the exact count is in `preflight/stage2.json` and
W&B. Both stages start from random weights; integration-check weights are discarded.

At each of 8×8×4 events, predict the atom, then its coefficient conditioned on
that atom's frozen dictionary vector. Both spatial and depth histories consume
complete earlier atom/coefficient pairs, including the learned pair adapter.
Use two causal coefficient micro-transformer layers and independent 2,048-bin
coefficient heads at each of four depths. Training atom logits are unmasked;
sampling forbids repeated OMP support within a site.

For each depth, divide physical coefficients by the full unique training-set
maximum absolute value / 3. Use those same four scales for validation; never
clip coefficients. Bins span [-3,3]. Normalized soft coefficient targets have
temperature 0.5, with stochastic coefficient context tokens. As in the reference,
this target noise differs from deterministic nearest-bin reconstruction.
Atom loss weight is 1.5. Distribution geometry uses top-4 atoms, weight 0.05,
starts at epoch 2, and ramps over three epochs. This preserves the historical
distribution-geometry formulation.

Text adaptation uses the original RQ-Transformer 16,384-entry character BPE,
32-token prefix, and 0.1 BPE dropout during training. Raw captions and
deterministic text tokens are cached. The image/text loss weights are 0.9/0.1.
These text settings come from the vendored CC3M RQ-Transformer recipe; the
transformer backbone and optimization schedule retain the FFHQ settings.

Train for 200 epochs, seed 0, global batch 128 (16/GPU × four GPUs × two
accumulations). AdamW: LR 5e-4, betas (0.9,0.95), weight decay 1e-4, gradient
norm clipping 1, no warmup, cosine decay to zero. Each epoch has 3,235 complete
updates and drops the last 33 caption pairs from a freshly shuffled order.
BF16, SDPA attention, fused AdamW, and DDP synchronization at update boundaries.

Sampling: atom temperature 1, top-k 2,048, top-p 1; coefficient temperature 1,
top-k disabled, top-p 0.85. Atom top-k scales the FFHQ vocabulary fraction
(250/2,048) to the 16,384-atom dictionary, rounded to 2,048. This is a documented
COCO adaptation, not the CC3M top-p-0.7 sampler. Captioned previews every 500 updates.

## Evaluation and online checkpoint retention

Every five stage-2 epochs, generate 40,504 images using the fixed validation
caption list. Ranks partition it without repeats or omissions. FID uses the
vendored original RQ-VAE Inception implementation against the same 40,504 real
validation images with the fixed transform. Generated pixels are converted to
uint8 before FID/CLIP, matching the existing text-to-image evaluator. CLIP is
the mean raw cosine similarity from OpenAI CLIP ViT-B/32; longer captions are
truncated to its supported context. This is **FID-40,504 on official val2014**,+not a claimed reproduction of a published COCO FID-30k/Karpathy benchmark.

W&B receives metrics, previews, and independently selected best FID/highest CLIP.
Stage 1 online slots are `last.ckpt` and `best-01.ckpt` (best reconstruction FID).
It saves/uploads latest after the first batch, every 200 Lightning optimizer
steps, and validation; the newly ranked best uploads at the next epoch start.
Stage 2 slots are `last.pt`, `best-fid-01.pt`, and `best-clip-01.pt`. Latest is
saved after update 1, every 200 updates, and every epoch. Full recovery state
contains model, optimizer, scheduler, epoch/cursor and per-rank Torch RNG.
Best stage-2 slots contain model/config/metrics and tokenizer-cache provenance.
Best checkpoints exist only after their first completed evaluation.
Uploads are asynchronous and may trail local saves. Fixed online slots replace
older versions. Stage-2 BPE dropout has a separate RNG stream that restarts on resume.

## Later original RQVAE + RQTransformer comparison

Reference this file and the resolved source snapshot. Match COCO split, image
transform, all training captions, fixed validation caption selection, seed,
stage-1 architecture/compute budget, 8×8×4 latent geometry, text tokenizer and
loss weights, stage-2 backbone, global batch, LR schedule, epoch budgets,
sampling seed, FID reference/backend, CLIP model, evaluation cadence, and
independent checkpoint selection. Train the original residual VQ bottleneck and
native RQ token predictor for that run. LASER's atom/coefficient heads, scaling,
and geometry loss are method-specific and should be listed as differences.
Report actual parameter counts and time as well as image/caption exposures.

## Operation

```bash
cat outputs/coco2014-laser-k4-compound-20260923/status.json
tail -f outputs/coco2014-laser-k4-compound-20260923/stage1.log
tail -f outputs/coco2014-laser-k4-compound-20260923/stage2.log
```

The supervisor checks its source manifest, resumes its own latest stage-1 or
stage-2 state, and refuses a duplicate pipeline lock. The selected tokenizer is
exported with a SHA-256 identity; encoder/decoder output equality is checked
against the native stage-1 model. A full-size four-GPU stage-2 update, checkpoint
reload, conditioned sampling and metric-network check gate stage 2 again after
the production tokenizer/cache are ready. The machine must remain running.
