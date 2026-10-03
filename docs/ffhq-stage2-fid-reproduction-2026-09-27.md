Update: the [new recovery bundle and PyTorch 2.4.1 rerun](ffhq-recovered-code-fid-2026-09-27.md)
scored **8.4334**. The historical **8.1744** remains unreproduced. The
earlier two-seed evaluation is documented below.

The recovered epoch-200 checkpoint produces **FID50k 8.4384 and 8.5303**
on two fresh generation seeds. The historical value is **8.1744**. The
mean of the two new FID50k measurements is **8.4843**, a difference of
**+0.3100 (+3.79%)**. The exact historical value has not been reproduced.
The remaining difference has not been attributed to a specific cause.

| Evaluation | FID50k | Difference from historical |
| --- | ---: | ---: |
| [Historical epoch 200](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803) | 8.1743927 | — |
| [Fresh seed 2026092700](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803-fid50k-reproduction-20260927) | 8.4383535 | +0.2639608 |
| [Fresh seed 2026092800](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803-fid50k-reproduction-seed2-20260927) | 8.5303440 | +0.3559513 |

Each new evaluation uses exactly 50,000 generated images and the full
70,000-image FFHQ reference. The second evaluation reuses the first run's
real-image features. Its real mean, covariance, and count match bit for bit.
Each seed base has eight disjoint rank streams; the second evaluation changes
the generated samples while preserving the checkpoint, reference, and policy.
The mean above is an average of two FID50k measurements, not a FID100k score.

The run's original output log explicitly records a reference change before
epoch 135: `fake=50000->50000, real=50000->70000`. The epoch-200 checkpoint
configuration also records the full-70k setting. Its uploaded trainer
snapshot still contains the earlier metric helper that caps real images
at 50,000. This reproduction therefore implements the final recorded
70,000-image protocol rather than invoking that older helper unchanged.

All 70,000 original 1024px RGB images were recovered from pinned revision
`d74f1f1f59e3bbe975bee29872b9bef827314577` of
[gaunernst/ffhq-1024-wds](https://huggingface.co/datasets/gaunernst/ffhq-1024-wds/tree/d74f1f1f59e3bbe975bee29872b9bef827314577).
Every decoded image matched the official pixel MD5 in
[NVIDIA's FFHQ metadata](https://github.com/NVlabs/ffhq-dataset).
Shard SHA256 values and exactly-once coverage of IDs 0–69999 were verified.
Preprocessing uses RGB and Pillow bilinear resizing to 256px, followed by
the archived CPU ToTensor/Normalize path and its inverse on the GPU.
The normalized transform was checked for exact equality at every uint8
intensity. This is a reconstruction of the documented reference; the
original run did not preserve its real feature statistics for direct comparison.

The evaluator uses TorchMetrics 1.9.0 `FrechetInceptionDistance(feature=2048,
normalize=True)` and torch-fidelity 0.4.0, matching the recorded package
versions. Inception weight SHA256 is
`6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.
The metric's float-to-uint8 truncation is preserved for both real and fake
pixels. These reference statistics should not be silently substituted for
an evaluator that uses continuous floating-point pixels.

Sampling preserves atom temperature 1, top-k 250, top-p 1; coefficient
temperature 1 and top-p 0.85; eight GPUs; and batch size 32 per GPU,
including the final partial batch. The cached sampler uses FP16 autocast;
decoder and Inception inference use FP32 with TF32 enabled and cuDNN
benchmarking enabled, as in the saved trainer. The physical coefficient
scales and attention resolutions are restored from the checkpoint config.

Every worker passed strict Stage 2 loading, Stage 1 non-quantizer key
checks, exact sample-count checks, finite pixel/feature checks, and agreement
between saved features and the actual TorchMetrics float64 accumulators.
An independent symmetric positive-semidefinite FID calculation agrees with
the TorchMetrics eigenvalue formula within **7e-12** on both evaluations.
This rules out a discrepancy between these two numerical calculations;
it does not identify the cause of the historical gap.

The checkpoint did not retain the original generation RNG state. The exact
uploaded trainer is recovered, but all imported historical revisions and
the final post-resume evaluator source were not recorded. Current PyTorch
is 2.8.0/CUDA 12.8; the recorded historical version was 2.4.1/CUDA 12.1.
These are reproducibility limits, not established explanations for the gap.
Two fresh evaluations are insufficient to attribute the difference to seed
variation or to a particular environment change.

The recovered checkpoint hashes, original sampling sweep, and model recovery
details are in [the recovery report](ffhq-stage2-recovery-2026-09-27.md).
Results, all generated token IDs, features, statistics, image previews,
reference manifests, metric source hashes, and evaluator scripts are retained
under `outputs/ffhqcmp0804205803-recovery-20260927/fid-reproduction/` and
`fid-reproduction-seed2/`, and in the two W&B evaluation artifacts.
The combined result is
[fid-reproduction-summary.json](../outputs/ffhqcmp0804205803-recovery-20260927/fid-reproduction-summary.json).
