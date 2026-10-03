The dark brush-like regions and clipped red/yellow spire tips are caused by
spurious mass at coefficient bin 0 or 2047 in sparse depth 0. They are visible
before the H100 continuation. The investigation uses a frozen step-24,500
checkpoint of the existing Church run; the diagnostic checkpoint, scripts,
raw measurements and image comparisons are preserved under
`outputs/church-ffhq-compound-350m-4a100-20260923/artifact-debug/`.

The distribution geometry loss contains a conditional-expectation error. The
model factors each event as `p(atom | history) p(coeff | history, atom)`, but
the original geometry loss multiplies an average of several atom vectors by
the coefficient mean conditioned on the teacher atom alone. Signed dictionary
alternatives need different coefficient conditionals. The corrected prediction
is `sum_a p(a | history) D[a] E[coeff | history, a]`, using the same top-four
candidate atoms plus the teacher atom, with duplicates masked. It calls the
existing atom-vector-conditioned coefficient head for each candidate and
retains gradients through both distributions.

Evidence from the frozen checkpoint:

- Cached continuous codes and nearest-bin codes reconstruct cleanly. Their
  pixel MAE is 0.000177 on [0,1], ruling out a material cache/quantization unit
  error in these samples. Decoder FP32 versus TF32 MAE is 0.000081.
- Exact full versus cached autoregressive passes agree in FP32: maximum atom
  logit difference 0.000038 and coefficient difference 0.000071 over all 256
  events. The completed atom/coefficient history is used consistently.
- Standard sampling generated 25 endpoint events among 2,048 depth-zero
  sites across 32 images. The mean predicted depth-zero endpoint mass is
  0.010534; mean soft-target endpoint mass on 3,945 cached images is only
  0.00000553. Other depths have no generated endpoint events in this batch.
- In a matched-image diagnostic, replacing only those 25 coefficients with
  their interior conditional mean removed the observed black/neon blotches.
  Fully clipped black pixels fell from 0.6031% to 0.1451% (75.9% relative).
  `endpoint-first8.png` shows original images on the left and diagnostic
  replacements on the right. This is a post-hoc causal diagnostic, not the
  production sampling procedure or evidence of post-training FID improvement.
- Lower coefficient temperatures made the problem worse. On the same seed,
  black pixels increased from 0.603% at temperature 1 to 1.387% at 0.1.
- On 128 real cached images, 213 depth-zero endpoint probabilities exceeded
  0.001. The original geometry gradient increases 70.4% of those logits;
  for 54 it even overcomes the opposing classification gradient. Correct
  candidate conditioning reduces that fraction to 16.0%, with zero cases
  overcoming classification. This supports the loss mismatch as a training
  mechanism for the observed endpoint spikes.

The correction is implemented in `src/training/compound_geometry.py` and the
existing trainer's `--compound-conditional-geometry` option. The legacy default
remains available for exact archived FFHQ reproduction. The new Church recipe
enables the correction explicitly. No parameters or checkpoint keys are added;
the model's classification logits and autoregressive sampler are unchanged.
The full pair chain, dictionary vector conditioning, coefficient targets,
sampling settings, optimizer, learning-rate schedule and global batch remain
the same. Endpoint probability is now logged per depth to W&B.

Validation includes explicit joint-probability enumeration and gradients,
signed atom alternatives, teacher-candidate deduplication, candidate head
gradients, strict checkpoint loading, full-pair causality, unchanged sampling,
and exact legacy FFHQ loss/logit/gradient equivalence when the option is off.
The focused regression suite passes 27 tests. One broader archived schedule
test could not import the optional FoundationVision VAR dependency; it was
excluded from the focused rerun. A four-H100 test resumed step 24,500, optimizer
state and all four RNG streams, completed two updates at global batch 256, and
verified every model/optimizer tensor is finite. It used microbatch 16 per GPU
while the main run was active. These test weights are not used in production.

The original run stopped cleanly at step 26,203, epoch index 53, batch cursor
74, preserving its cosine schedule at learning rate 0.00046226583587667046.
Its best FID remains 15.30345208056923 at epoch 45. The old runtime and source
manifest remain intact. The corrected source is separately archived as
`source-conditional-geometry.tar.gz`, with
`runtime-conditional-geometry-manifest.json` and a reviewable
`conditional-geometry.patch`. The selected continuation is recorded in
`resume-runtime.json` after its full-batch preflight passes, so the existing
restart helper restores the corrected runtime on subsequent restarts.

After the user chose a fresh start, the brief corrected continuation was
cancelled and superseded by the separate online run
`church-ffhq-compound-350m-4h100-jointgeom-scratch-20260923`. The existing run's
step-26,203 recovery and epoch-45 best-FID checkpoint are preserved, with both
online files verified against local MD5 checksums. The new run starts the
stage-2 model, optimizer and cosine schedule from scratch, with seed 0 and no
stage-2 checkpoint input. The clean tokenizer and prebuilt token cache are
reused. Its first two epochs retain the original geometry delay, followed by
the original three-epoch ramp. Candidate conditional heads are skipped while
the geometry weight is zero, avoiding unnecessary warmup computation.

The fresh source snapshot and configuration live in the new output directory.
A fresh four-H100 preflight exercised the corrected loss at batch 64 per GPU,
global batch 256, for two updates; all parameters and optimizer states were
finite, and peak allocated memory was 36.15 GiB per GPU. Those preflight weights
are discarded before the actual scratch launch.

Fresh autoregressive previews, endpoint metrics, and the unchanged 50,000-image
FID evaluation must establish the new run's image quality. No endpoint mask,
coefficient clipping, temperature change, or image postprocessing is added to
production.
