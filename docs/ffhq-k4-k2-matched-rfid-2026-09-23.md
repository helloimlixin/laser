# FFHQ K=4 evaluated with the K=2 reconstruction-FID protocol

The K=4 epoch-100 checkpoint scores **6.703592 rFID** on the original
RQ-VAE validation protocol, versus K=2's logged epoch-100 **7.027678**.
K=4 is lower by **0.324086 (4.61%)** under this protocol. K=2 was not rerun;
its epoch-100 checkpoint was not retained in its accessible W&B run files.

| Epoch-100 result | rFID |
| --- | ---: |
| K=2, historical `valid/rfid` | 7.027678 |
| K=4, historical `valid/loss/rfid` | 8.740578 |
| K=4, newly evaluated with K=2's protocol | **6.703592** |

## What changed between the historical evaluations

The input resize differs, in addition to the FID implementation. All 16
original-image tiles from K=4's final validation preview match Pillow
**BICUBIC** resizing of the official 1024px images exactly. K=2's checked
original tile (`01275.png`) matches **BILINEAR** exactly after reproducing
its float32 normalization and uint8 preview conversion. Bicubic and Lanczos
do not match that K=2 tile.

Changing the metric backend alone is a small effect in the controlled
comparison below. The earlier suspicion that the evaluator backend itself
explained the reported gap was not supported by this experiment.

| K=4 input preparation, same 10,000 IDs | Original RQ-VAE | TorchMetrics |
| --- | ---: | ---: |
| Official 1024px RGB → 256px bilinear | **6.703592** | 6.682177 |
| Official 1024px RGB → 256px Lanczos | 10.810473 | 10.727467 |

Each row scores the same reconstructions through both feature pipelines.
The Lanczos row is a sensitivity experiment, not a claim that K=4 trained
on Lanczos. Its saved originals instead establish bicubic preparation.
A full bicubic reevaluation was not performed; 8.740578 remains the historical
logged measurement, not a newly reproduced score.

## Evaluation contract and checks

- Checkpoint: `ffhq-a2048-k4-stage1-100ep-20260910-stage1-checkpoints:v99`,
  `last_model.pt`; checkpoint metadata confirms epoch 100.
- Checkpoint SHA256:
  `6abdb445b573d013bc6047d809891942194f0d0b1a0ca239d13c2fb19cbe1c70`.
- Exactly the 10,000 unique IDs in the vendored RQ-VAE
  `rqvae/img_datasets/assets/ffhqvalidation.txt`; no duplicates or omissions.
- Official FFHQ source files downloaded from their metadata URLs; original
  file MD5 and decoded RGB pixel MD5 checked for every image.
- Primary input: RGB, Pillow bilinear 1024→256, identity center crop,
  normalize to [-1,1]. Original images are square, so this matches
  `Resize(256) → CenterCrop(256)`.
- Frozen encoder, learned projections, continuous K=4 OMP coefficients,
  dictionary, and decoder. No coefficient tokenization or clipping.
  All encoder/decoder/projection checkpoint keys were checked on load.
- One image reconstructed at a time, matching the historical RQ-VAE
  `compute_statistics_dataset` path. Four GPUs process disjoint image IDs;
  FID is calculated once from all 10,000 feature vectors, never averaged
  over shard FIDs.
- Original vendored RQ-VAE/pytorch-fid Inception, feature dimension 2048,
  continuous clamped [0,1] pixels; Inception batch 32. FP32 inference,
  TF32 disabled; NumPy covariance and SciPy matrix square root.
- The real-image mean/std are **0.44474949 / 0.27036434**, matching K=2's
  log to its printed precision (**0.4447 / 0.2704**).
- The paired TorchMetrics path receives byte-converted versions of those
  same floating-point images. Its FID implementation source matches the
  downloaded training-version 1.8.2 wheel byte-for-byte.

This reproduces the evaluation recipe and checked preprocessing, not the
entire historical software/hardware environment. Current package versions,
feature-weight hashes, source hashes, checkpoint identity, per-image data
hashes, and worker logs are saved with the results. The sparse inference
adapter is the repository's `LaserAux`, with attention at resolution 16,
FP32 OMP, and coefficient clamping disabled.

## Evidence

- [Primary result](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/result-bilinear.json)
- [Lanczos sensitivity result](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/result-lanczos.json)
- [K=4 preview pixel comparisons](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/preview-resize-verification.json)
- [K=2 preview pixel comparison](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/k2-preview-resize-verification.json)
- [Evaluation provenance](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/evaluation-provenance.json)
- [Data manifest](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/validation-manifest.json)
- [Evaluator](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/evaluate_k4.py)
- [Aggregation](../outputs/ffhq-k4-k2-protocol-evaluation-20260923/aggregate.py)
- [K=2 W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k2-rqvae-strict-20260720-145706)
- [K=4 W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-stage1-100ep-20260910)

The reproduction scripts use `/tmp/laser_wandb_comparison` for downloaded
checkpoints, prepared images, and per-image feature arrays. Training weights
and the original W&B histories were not modified.
