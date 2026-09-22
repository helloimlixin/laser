# Physical sparse coefficients: prepared and validated

The alternative cache stores FP32 coefficients in physical latent units and
uses one shared 2048-center Lloyd–Max vocabulary across all four sparse
depths. Its scale vector is `[1, 1, 1, 1]`. No new training was launched.

- [Prepared cache](../outputs/church-physical-coefficients-20260921/compound-cache-physical.pt)
- [Codec configuration overrides](../outputs/church-physical-coefficients-20260921/codec-overrides.json)
- [Full validation report](../outputs/church-physical-coefficients-20260921/report.json)
- [Decoder comparison](../outputs/church-physical-coefficients-20260921/codec-comparison.png)
- [Preparation tool](../scripts/tools/prepare_physical_compound_cache.py)

The cache contains all 126227 training images, an 8×8 grid per image, and
16 alternatives of four atom/coefficient pairs per spatial location. The
original supports, alternative ordering, image labels, and tokenizer are
preserved. Coefficients are recovered by multiplying the source cache by its
existing depth scales. This exactly preserves the physical coefficients used
by the current codec; it cannot undo rounding that occurred when the original
FP32 normalized cache was created.

The coefficient bins were fitted on all 500248576 coefficients from training
images 4096–126226 and all their cached variants. The first 4096 training
images and all validation images were excluded. Lloyd fitting used 250 exact
empirical centroid-update iterations; global optimality is not claimed.

## Quantization comparison

Values below are added latent MSE relative to continuous sparse reconstruction,
not total reconstruction error or generated-image FID. Each row compares the
same atom supports and continuous coefficients.

| Evaluation | Current normalized bins | Normalized bins refitted on the same stochastic bank | New physical bins |
|---|---:|---:|---:|
| 4096 held-out training images, all 16 variants | 4.6514e-7 | **7.0238e-8** | 1.8007e-7 |
| 300 validation images, 16 fresh stochastic variants | 6.2378e-7 | **8.3445e-8** | 1.8551e-7 |

The raw-coefficient version improves on the *existing* normalized codebook,
which was fitted on greedy OMP codes. However, refitting the normalized
codebook on the same stochastic bank improves quantization further. Against
that matched-data control, physical bins have approximately 2.56× the added
latent MSE on held-out training images and 2.22× on validation. The normalized
control optimizes normalized coefficient MSE; the physical codebook optimizes
physical coefficient MSE. These are empirical quantizer comparisons, not proof
of an optimal quantizer under either parameterization.

The original greedy-OMP validation codes were also checked: current normalized
bins add 8.0773e-8 latent MSE, versus 1.8072e-7 for the physical bins. Their
different support distribution is why the fresh stochastic validation bank
was also constructed. It uses the same atom temperature 0.0625, 16 variants,
and validation-only seed 20260921. No validation coefficients enter either fit.

Physical token meanings are consistent across depth, but removing normalization
is not established as a reconstruction improvement. These small quantization
errors do not establish the direction of a future generated-FID change.

## Codec and implementation checks

Six focused tests pass. CPU integration against the active experiment's
frozen codec also verified:

- Full-cache supports and labels are unchanged, and serialization preserves
  the converted coefficient tensors exactly.
- Eight sampled images have bitwise-identical continuous physical latents.
- Hard coefficient IDs match nearest physical-bin centers.
- Soft targets match `exp(-(physical_coefficient - physical_bin)^2 / 0.125)`.
- Equal physical coefficients have identical soft targets at all four depths.
- Physical pair embeddings and the actual decoder API match explicit latent
  assembly; the tokenizer remains frozen.
- The existing stochastic-cache dataset loader accepts the new cache.
- A new cache identity prevents silently resuming a model with the old
  coefficient vocabulary.

The decoder comparison has three rows: continuous coefficients, current
normalized bins, and new physical bins. On those eight images, pixel MSE
relative to the continuous-coefficient decode is 4.0697e-7 for current bins
and 2.8836e-7 for physical bins. This small probe is not a perceptual or FID
evaluation and does not include the refitted normalized control.

The physical soft-target temperature remains 0.125. The changed bin spacing
changes target entropy; raw cross-entropy values from a future run therefore
should not be compared directly with the current run as a quality measure.

The coefficient vocabulary has changed. The current trained checkpoint cannot
be used unchanged with these overrides. A later experiment needs explicit
fresh training or a separately designed vocabulary adaptation. The current
continuation and its assets were not modified.

Cache SHA256:
`437bf76107dd3da5661db6c766b474be43304f37705d6c8431f2e34ae3afa9fa`.
The output directory retains the executed preparation source, fit traces,
independent validation bank, codec checks, and matched-control reports.

The subsequent user-authorized training trial is documented in
[the raw-coefficient experiment note](church-stochastic-rawcoeff90-2026-09-21.md).
The preparation receipts' `training_launched: false` fields describe the
completed preparation phase; the new trial has a separate output directory.
