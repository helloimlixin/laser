# No-refit Church teacher direction

**Superseded by the user's clarification:** reuse the original sparse codes
unchanged. Do not replace them with the residual teacher described below. The
active experiment is documented in `church-fixed-code-sequence-2026-09-26.md`:
stage2 reads the original cache, with no re-encoding, refitting, bank selection,
or coefficient resampling. The stage1 method that originally produced the
cache is preserved.

The requested next representation fixes each sampled atom/coefficient pair before
constructing the next residual. The existing implementation is
`src/training/residual_pair_targets.py`. It does not perform a least-squares
refit, either between selections or after the final selection.

For a latent feature `z`, initialize `r = z`. At depth `d`, the teacher samples
from `q(a,c | r) ∝ exp(-||r - c D[:,a]||² / temperature)`, first using the exact
atom marginal, then the coefficient conditional for the selected atom. It
subtracts the actual sampled, quantized physical contribution from `r`. Earlier
pairs stay fixed, including when the requested sequence depth is extended.

This is a residual pair coder. Final OMP codes are also valid autoregressive
training sequences: computing a coefficient using the full selected support
does not itself expose future tokens to the predictor. The motivation for this
change is the incremental residual construction and its possible learning
advantages, not a demonstrated violation of causal masking by the current model.

Validation on this decision: 19 tests passed across
`test_residual_pair_targets.py`, `test_omp_joint_targets.py`, and
`test_compound_sequence.py`. These cover exact expanded-codebook target/loss
agreement, sampled-prefix residual updates, unchanged pairs when depth is
extended, replay, and causal sequence predictions. They do not establish better
generation quality or production throughput.

Production integration remains necessary. The existing teacher is experimental
and is not wired into the active frozen run. In particular:

- Use prequantization encoder latents or images, not the existing OMP trajectory
  bank. Both supports and coefficients must be regenerated sequentially.
- Align training and generation support masks: this teacher permits repeated
  atoms, whereas the current OMP sampler masks previously selected atoms.
- Keep the pair-history decoders and joint support-plus-coefficient likelihood.
- Calibrate the physical-distance temperature; the old normalized coefficient
  temperature is not the same noise parameter.
- Validate reconstruction and all-GPU throughput before a fresh stage2 run.
  The exhaustive teacher sums 16,384 × 2,048 pairs per residual step and must not
  be assumed to retain the current cached teacher's throughput.

The current production process and its checkpoints were not changed by this
decision or CPU validation.
