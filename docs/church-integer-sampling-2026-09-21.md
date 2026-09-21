This experiment compares sampling settings for the integer-token run [church-laser-ft3best-integer-scratch90-h200x8-20260921](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-integer-scratch90-h200x8-20260921), using its best-FID checkpoint at epoch 50, optimizer step 3,100. The full checkpoint is pinned with a hardlink so continued training cannot replace it. Its MD5 and size match `best-fid-01.pt` in the parent's verified online checkpoint artifact v9.

Evaluation run: [church-laser-ft3best-integer-sampling-20260921](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-integer-sampling-20260921).

Every setting generates exactly **50,000 images** and compares them with the existing original RQ-VAE Inception statistics for **all 126,227 Church training images**. These are not validation-set or 4,096-image reference statistics. The reference SHA-256 is `ad3b5a341831d877110afdff37d6fff4be855612053e3eb9e2e7119f5593d364`; its receipt confirms zero dropped or padded real images.

| Setting | Temperature | Top-k | Top-p |
| --- | ---: | ---: | ---: |
| Baseline | 1.00 | 1,400 | 1.00 |
| Cooler | 0.95 | 1,400 | 1.00 |
| Cooler | 0.90 | 1,400 | 1.00 |
| Warmer | 1.05 | 1,400 | 1.00 |
| Narrower top-k | 1.00 | 700 | 1.00 |
| Wider top-k | 1.00 | 2,800 | 1.00 |
| Nucleus | 1.00 | unrestricted | 0.95 |
| Nucleus | 1.00 | unrestricted | 0.98 |

The sweep uses rank seeds 71,000–71,007, reset for each setting. This matches the recorded baseline evaluation, with the same eight ranks and generation batch 512/GPU. The baseline must reproduce its recorded FID 11.282049 within 0.02 before alternatives proceed. After all eight settings finish, the baseline and best nonbaseline setting each receive another 50,000-image evaluation with independent rank seeds 81,000–81,007. The selected sampler is determined by that confirmation comparison. The seed ranges are disjoint.

The evaluator uses the exact frozen tokenizer, including its stored dictionary rounding, and checks its complete state hash. Sampling uses the original FP16 autoregressive sampler; the frozen decoder and Inception run in FP32, moments accumulate in FP64, and TF32 is disabled. Generated pixels remain continuous RGB in [0,1] with the original Inception preprocessing. Convolution autotuning is disabled to bound transient memory during concurrent training.

Evaluation runs alongside Lloyd–Max compound training on eight H200s with a 15% per-process GPU memory cap. Decode and Inception batches are 64. No training weights, learning rates, or training samplers are changed. The integer run remains in its previously arranged pause/resume queue. Sampling temporarily shares GPU compute with the active training run.

Experiment directory: `outputs/church-integer-ft3best-sampling-20260921`.

- `frozen-best.pt` and `frozen-checkpoint.json`: pinned weights and online provenance; SHA-256 `681c70cb408b3a18e01ac7c6b7cd4cb434ec45475023af250cead0945e162d7a`.
- `request.json`, `protocol.json`, and `source-manifest.json`: exact settings, metric protocol, and evaluator hashes.
- `results.json`: completed settings, with generated/real counts and FID.
- `sweep/*` and `confirm/*`: all 50,000 generated token grids, feature moments, token usage, 64-image previews, and per-setting receipts. Full PNG sets are not retained; all images are decoded for the FID calculation.
- Per-setting checks include distinct rank RNG streams and the number of unique generated token grids; no generated samples are filtered or deduplicated before FID.
- `report.json`, `selected-sampling.json`, and `upload.json`: final paired results, selected sampler, and verified W&B artifact manifest, written after the sweep completes.

Completed 50,000-image sweep (seed base 71,000), sorted by FID:

| Setting | Temperature | Top-k | Top-p | FID vs all training images |
| --- | ---: | ---: | ---: | ---: |
| nucleus_095 | 1.00 | unrestricted | 0.95 | 10.839705 |
| cooler_090 | 0.90 | 1400 | 1.00 | 10.848488 |
| cooler_095 | 0.95 | 1400 | 1.00 | 10.936506 |
| nucleus_098 | 1.00 | unrestricted | 0.98 | 11.184754 |
| baseline | 1.00 | 1400 | 1.00 | 11.282052 |
| narrower_700 | 1.00 | 700 | 1.00 | 11.316776 |
| wider_2800 | 1.00 | 2800 | 1.00 | 11.387529 |
| warmer_105 | 1.05 | 1400 | 1.00 | 11.777486 |

All eight settings completed with exactly 50,000 unique generated token grids and distinct rank RNG streams. The baseline reproduced the historical training evaluation within 0.000003 FID. Independent-seed confirmation is recorded below.

Independent confirmation (seed base 81,000), again with 50,000 generated images per setting against all 126,227 training images:

| Setting | FID |
| --- | ---: |
| Baseline: T=1, top-k=1,400, top-p=1 | 11.248990 |
| Selected: T=1, top-k disabled, top-p=0.95 | 10.735621 |

The selected sampler improves confirmation FID by 0.513369 (4.56%). This agrees with the initial sweep's improvement; the T=0.90/top-k=1,400 alternative was nearly tied with the selected sampler in the initial sweep but was not independently repeated.

All ten evaluations completed: 500,000 generated images total, with 50,000 unique token grids in each evaluation. Tokenizer state remained unchanged. The report, exact configuration, all generated token grids, feature statistics, and preview grids were committed and remotely verified in `helloimlixin-rutgers/laser/church-laser-ft3best-integer-sampling-20260921-evaluation:v0`.

[Selected confirmation preview](../outputs/church-integer-ft3best-sampling-20260921/confirm/nucleus_095/samples.png) · [Selected sampling configuration](../outputs/church-integer-ft3best-sampling-20260921/selected-sampling.json) · [Full result report](../outputs/church-integer-ft3best-sampling-20260921/report.json).
