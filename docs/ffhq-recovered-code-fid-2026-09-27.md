The updated recovery bundle produces **FID50k 8.4334** using the recorded
Python 3.11.9 / PyTorch 2.4.1+cu121 framework versions. The historical value
is **8.1744**, leaving a difference of **+0.2590**. The exact historical
score remains unreproduced.

| Evaluation | FID50k |
| --- | ---: |
| [Historical epoch 200](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803) | 8.1743927 |
| [Previous code, PyTorch 2.8.0, same seed base](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803-fid50k-reproduction-20260927) | 8.4383535 |
| [Updated recovery, PyTorch 2.4.1](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803-fid50k-recovered-torch241-20260927) | 8.4333811 |

The new result is **0.0050 lower** than the previous evaluation with the
same seed base, 2026092700. Each evaluation generates 50,000 images against
the full 70,000-image FFHQ reference. This comparison changes both the
reconstructed dependency bundle and framework environment; it does not
isolate either change as a cause.

The new bundle is `recovered/ffhqcmp0804205803`, supplied by repository
commit `202cc47`. Its trainer is byte-identical to the earlier W&B recovery.
The model imports are reconstructed from July 31 commit
`449ed552a9de7ac4cdcd155690eea02c933ca57e`. All 36 entries in its provenance
manifest were verified. The two log files omitted from the unpacked Git
directory were recovered from the supplied tarball with matching hashes.
The evaluation freezes a separate copy; it does not modify the recovery
bundle or either checkpoint.

A 32-image control using the updated bundle on PyTorch 2.8.0 produced
identical token IDs and preview image pixels to the previous implementation.
Inception features differed slightly despite identical preview pixels.
This limited control does not establish equivalence of all outputs.

Generation and feature extraction use Python 3.11.9, PyTorch 2.4.1+cu121,
torchvision 0.19.1+cu121, cuDNN 9.1, Pillow 10.2.0, TorchMetrics 1.9.0, and
torch-fidelity 0.4.0. All installed worker package versions are retained.
Both real and generated Inception features were recomputed in that
environment. The W&B coordinator and float64 CPU aggregation use the host
PyTorch 2.8.0 environment. The TorchMetrics FID formula and independent
symmetric PSD formula agree within **1.1e-11**.

The original policy is preserved: atom temperature 1, top-k 250, top-p 1;
coefficient temperature 1 and top-p 0.85; FP16 sampler autocast, FP32
decoder/Inception, TF32 enabled, cuDNN benchmarking enabled, eight workers,
and batch size 32 per worker. The final partial batches are included.
Every worker passed strict Stage 2 loading, Stage 1 key validation, sample
counts, finite pixel/feature checks, and agreement between saved features
and TorchMetrics accumulators.

Both checkpoint SHA256 values and all 70 prepared reference shard hashes
were reverified. A check of 100 original FFHQ images across the ten locally
retained raw shards confirmed that Pillow 10.2.0 resizing and the full
normalized image transform match the prepared pixels bit for bit. The
prior recovery verified official pixel MD5s for all 70,000 source images.

The final checkpoint and log specify 50,000 fake versus 70,000 real images.
The uploaded trainer still contains an older helper that uses equal real
and fake counts. This rerun explicitly adapts evaluation to the final
recorded 70k reference protocol, preserving the trainer source unchanged.

Remaining limits include the missing historical generation RNG state and
final post-resume evaluator source, reconstructed rather than verified
historical imports, and H100 hardware versus the original A100 hardware.
The remaining FID gap has not been attributed to a specific cause.

Results, generated token IDs, sample grids, real/fake features and statistics,
source snapshots, manifests, environment records, and checks are retained
under `outputs/ffhqcmp0804205803-recovery-20260927/fid-recovered-code-torch241/`.
The linked W&B run stores the score and corresponding evaluation artifact.
See the [previous evaluation report](ffhq-stage2-fid-reproduction-2026-09-27.md)
for the earlier two-seed results and reference reconstruction.
