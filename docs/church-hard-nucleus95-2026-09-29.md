# Church: conditional nucleus sampling fix, 2026-09-29

On the same epoch-40 / step-2480 checkpoint, conditional top-p0.95 at temperature1 improved official FID50k by 6.862833. Training continued during this checkpoint-only evaluation.

| Metric | Full vocabulary, T1 | Conditional nucleus0.95, T1 |
|---|---:|---:|
| FID50k | 39.023496743 | 32.160663566 |
| Mean term | 19.806671143 | 15.706308365 |
| Covariance term | 19.216825600 | 16.454355202 |

The sampler draws an atom from a boundary-inclusive 95% nucleus, then draws its coefficient from that atom's conditional 95% nucleus. Both fields are committed together before the next autoregressive event. Repeated atoms remain allowed. There is no fixed top-k, refitting, temperature change, objective change, or physical coefficient rescaling. This is conditional per-factor filtering, not a global joint ordering of compound categories.

| Sparsity level | Mean retained atoms /16384 | Mean retained coefficients /2048 | Minimum atom mass | Minimum conditional coefficient mass |
|---|---:|---:|---:|---:|
| 1 | 4572.6 | 213.8 | 0.9500000477 | 0.9500000477 |
| 2 | 10733.8 | 119.0 | 0.9500000477 | 0.9500000477 |
| 3 | 13257.6 | 89.9 | 0.9500000477 | 0.9500000477 |
| 4 | 13939.3 | 65.8 | 0.9500000477 | 0.9500000477 |

Matched protocol: eight logical RNG streams with seeds2026093901..2026093908; 6,250 images per stream, batch sequence1024×6+106; decoder and Inception batch32, FP32 continuous pixels/features with TF32 off; BF16 model with FP32 probabilities; same official Church reference SHA809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6. All50,000 images and12,800,000 complete pairs passed count/coverage checks.

Completed checkpoint SHA256 `bdbbea3226097da9732c6dbb16a3c69ec22b5c26278937bf559278d8960cde88`; model state digest `d09b7c4592fc7dc930aacf35f69a7edb9cfe97ebac990acec509ba854e85cdd4`. A native replay of the first1,024 control images produced bitwise-identical Inception features. This connects the completed checkpoint to the pending checkpoint used by the recorded full evaluation; their serialized-file hashes differ because checkpoint metadata changed after evaluation. Both samplers used identical weights, verified unchanged afterward.

Eight sampler CPU tests and the capacity probe passed. The candidate used at most10.326GiB allocated /14.863GiB reserved per sidecar. Its175.87s maximum stream time includes generation, decoding, Inception extraction and contention with ongoing training; it is not an isolated generation-speed measurement.

[W&B comparison, grids and coverage](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-hard-nucleus95-step002480-20260929). Verified artifact `helloimlixin-rutgers/laser/church-hard-nucleus95-step002480-20260929-evaluation:v0` contains 87 files with matching remote digest/size entries. Sources, exact feature arrays, statistics and receipts are also under `outputs/church-hard-nucleus95-step002480-20260929/`.

Adoption is authorized by this measured improvement. The continuation must retain the latest parent weights, all Adam states, scheduler, cursor and eight RNG streams; preserve the old full-sampler BEST separately; start nucleus-sampler BEST from its own evaluations; retain300epochs/18600updates and previews every200updates. Deployment details are recorded separately after launch.

This result establishes a sampling-policy improvement on one checkpoint and fixed evaluation seed protocol. It does not establish that the training convergence problem is solved.
