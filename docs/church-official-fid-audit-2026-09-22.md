The active Church compound-history run uses the original RQ-VAE FID computation.
An independent audit on September 22 verified its source, weights, image
preprocessing, statistics, and distributed feature collection. No training or
evaluator changes were required.

The reference release is Kakao Brain's
[`rq-vae-transformer` commit 341395e562ac347f5eb62db9f5f08b9f2cc42a60](https://github.com/kakaobrain/rq-vae-transformer/tree/341395e562ac347f5eb62db9f5f08b9f2cc42a60).
Fresh downloads of `rqvae/metrics/fid.py` and `inception.py` match the active
files byte for byte. The cached FID Inception weights pass the upstream SHA-256
prefix check and have full hash
`6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`.

The active wrapper calls the released `get_inception_model`,
`mean_covar_numpy`, and `frechet_distance` directly. Inception extracts 2,048
FP32 features with autocast and TF32 disabled. Image inputs are float32 RGB
values in [0,1], retaining the original floating-point sampling pipeline.
Generated images receive `(decoded + 1) * 0.5`, followed by image-range clamping.
This matches the released sampling script; it does not clip sparse coefficients.
Inception applies its released bilinear resize to 299 by 299 with
`align_corners=False` and its own normalization. Means remain float32 and
covariances use NumPy's sample covariance, as in the release. The only evaluator
compatibility bridge handles SciPy's changing `sqrtm(..., disp=False)` return
signature without modifying the matrix or result.

Each epoch compares exactly 50,000 generated images against the fixed reference
for all 126,227 unique Church training images. The active token cache and real
reference hashes match the existing population/transform audit. All 36 audited
real-image probes were freshly checked against transforms downloaded from the
release and matched bit for bit: RGB conversion, bilinear short-side resize to
256, center crop to 256, tensor conversion, and normalization by 0.5.

The numerical audit fed the same 36 FP32 image probes through the active wrapper
and the original pickle-based evaluator, using their actual Inception models.
Weights, features, mean, covariance, and the final full 2,048-dimensional FID
matched exactly, with zero absolute difference. The original default activation
batch size also produced identical features in this test. The probe images are
for evaluator equivalence; their score is not a model-quality measurement.

A separate five-rank NCCL check partitioned those features into unequal groups
of 1, 4, 7, 10, and 14 rows. It verified exact reconstruction of all 36 rows,
exclusion of collective padding, and identical final FID on all five ranks.
CPU Gloo initially encountered PyTorch's inference-tensor collective limitation;
the completed distributed check used the actual production NCCL backend instead.
Feature equivalence was tested on CPU, so this does not assert bitwise equality
between different GPU hardware or convolution batch sizes.

Audit script, downloaded sources, complete results, and rank receipts are in
`outputs/church-compound-history-scratch-20260922/fid-audit-20260922`.
The machine-readable result is `result.json`; `publication.json` records the
verified W&B artifact after publication.
