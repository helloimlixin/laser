# Church integer coefficient audit

The running 32,769-token model does not clip or normalize sparse coefficients.
Its integer lookup is correct, but its representation is a lossy four-step
scaled-atom residual quantizer, not a lossless encoding of the original OMP
supports and continuous coefficients.

For atom `a` and coefficient choice `b`, the nonzero token is `1 + 2*a + b`.
Token zero contributes a zero vector. Decoding a nonzero token returns
`dictionary[:, a] * levels[a, b]`; the four contributions are summed before
the frozen image decoder. Each atom has only two fitted coefficient values,
one negative and one positive. Selecting these entries can change both
coefficient magnitudes and atom choices relative to native OMP, and does not
perform OMP's least-squares coefficient refit. Raw units do not imply lossless
coefficient preservation.

The current saved levels range from -13.903696 to 14.989018. These values are
not clipped at embedding or decoding. Dictionary atoms are unit normalized,
matching the native stage-1 convention. The zero-token index guard, optimizer
gradient clipping, and decoded-image range clamp are separate operations; none
clips sparse coefficients.

## Fresh numerical checks

The audit ran on CPU without changing or interrupting training:

- Verified the tokenizer checkpoint, codebook, and complete frozen tokenizer
  state against their recorded SHA-256 hashes.
- Checked all 32,769 integer IDs against an independently constructed embedding
  table, including the zero token. Every entry matched exactly.
- Compared all four hard residual-quantization assignments for eight cached
  latent vectors with exhaustive FP64 Euclidean-distance search over the full
  codebook. Every chosen ID matched.
- Verified deterministic training targets choose the same IDs, with maximum
  probability error of 0.00003092 against the FP64 distance reference.

Reproduction: `outputs/church-integer-raw-rqrecipe300-20260922/coefficient-audit-20260922.py`.
Results: `outputs/church-integer-raw-rqrecipe300-20260922/coefficient-audit-20260922.json`.

## Evidence relevant to generation quality

An earlier paired 128-image reconstruction audit using this same frozen
representation measured native OMP latent MSE 0.00956866 and hard integer MSE
0.01779817, an 86.0% increase. Pixel MSE increased from 0.01313515 to 0.01398435,
or 6.47%. This demonstrates representation distortion; it does not measure
generated FID or establish how much of the FID is caused by quantization.

The fixed 300-image train and validation probes also show a growing
generalization gap: from epoch 17 to epoch 52, train KL fell from 7.0095 to
3.7279 while validation KL rose from 7.6310 to 9.7970. This pattern is consistent
with overfitting and began before the single-GPU continuation. More epochs do
not guarantee better generation quality.

Epoch 52 generated FID is 12.3933; the best remains 11.8709 at epoch 36.
Separately, replaying the original epoch-37 checkpoint with the continuation's
evaluator reproduced its original FID within 0.00000128, supporting evaluation
consistency after recovery.

Preserving the native continuous sparse code requires representing its
coefficient values as well as its atom indices. Two fixed values per atom
cannot do that. A change of representation would require a separate compatible
prior rather than modifying this run's codebook during training. The requested
300-epoch continuation and checkpoint monitoring remain active.
