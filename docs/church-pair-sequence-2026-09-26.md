# Church: fresh joint support/coefficient sequence model

Run: [church-omp-joint-sequence-scratch-b2048-h200x3-20260926](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-omp-joint-sequence-scratch-b2048-h200x3-20260926).

Runtime and receipts: `/mnt/laser-church/pair-sequence-fresh-20260926`.
This is a fresh **stage-2** experiment. The selected LASER tokenizer stays frozen.

## Implemented change

Keep standard full-refit OMP and the complete history of earlier atom/coefficient
pairs. At event t, model

`p(atom_t | earlier pairs) * p(coefficient_t | atom_t, earlier pairs)`.

The existing 24-layer spatial / 4-layer depth RQ backbone remains. It now feeds
two independently parameterized causal sequence decoders, each with two layers,
width 512 and eight heads, over the complete 256-event raster/depth sequence:

1. The support decoder receives the backbone state, strictly previous completed
   pair embedding, and strictly earlier physical reconstruction at the current
   site. It has no projection for the current atom or coefficient.
2. The coefficient decoder additionally receives the selected current atom and
   the refined support state. Its output augments the existing local coefficient
   conditioner. It cannot see the current coefficient or future pairs.

The new model has **420,365,312 parameters**, all initialized afresh using the
released RQ initialization helper. The new decoder outputs are active at
initialization, so both stacks receive gradients on their first update.

This follows the stacked conditional sequence-decoder idea in
[DCTransformer, section 3.2](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf).
It adapts that idea to LASER's support/value pairs; it does not reproduce DCT
channel/position/value tokens or the separate partial-image cross-attention encoder.

The objective is **support soft CE + coefficient soft CE**, averaged over pairs.
The former 1.5 support multiplier and division by total head weight are removed.
Consequently, the total loss scale differs from preceding runs. Both heads use
exact conditional targets within the existing 16-trajectory OMP bank; the
coefficient target averages kernels over all trajectories compatible with the
observed pair prefix and current atom. This changes the coefficient estimator's
variance, not its expected CE or the sampled teacher distribution.

## Recipe and execution

| Setting | Value |
| --- | --- |
| GPUs | All three H200s |
| Global batch | 2048; exact per-image DDP weighting |
| Training microbatch | Up to 342 per GPU, two accumulation steps |
| Epoch coverage | All 126,227 training images; 62 updates, including final partial batch |
| Optimizer | Fused AdamW, betas (0.9, 0.95), weight decay 0.0001 |
| Learning rate | 0.0005, cosine to zero over 300 epochs, no warmup |
| Gradient clipping | 1.0 |
| Dropout | Residual 0.1; attention and embedding 0 |
| Noise | Existing normalized coefficient temperature 0.0625; no further reduction |
| Precision | BF16 training, native sampler precision, FP32 targets/loss/FID |
| Generation batch | 8192 per GPU; decoding/Inception in chunks of 64 |
| Sampling | Atom k700/T1/p1; coefficient T0.9/p0.85 |
| Evaluation | Official released RQ FID50k every epoch, same Church real reference |

Transferable RQ optimization and dropout settings are matched. BF16 training and
the finite OMP teacher remain explicit differences from the local original RQ
FP16 training and online full-vocabulary Gibbs teacher. This is not a claim of
exact original-RQ training reproduction.

Measured training throughput: microbatch 342 **1714 images/s**, 448 **1710
images/s**. The 512 layout was slower (about 1389 images/s) and failed the
checkpoint RNG gather with an NCCL CUDA error near full GPU memory; it was
rejected. Larger memory occupancy did not improve throughput.

Generation benchmarks covered 1024 through 10,240 images/GPU. Batch 8192 was
fastest at about **178 images/s/GPU** for sampling and decoding. The largest
three candidates included an extra 8 GiB allocation for evaluation headroom;
10,240 was slower. Inception extraction is additional work.

## Verification and recoverability

- 27 relevant maintained-code tests passed; 13 tests against the frozen runtime
  checked decoder behavior, joint targets, and unequal-batch gradient equivalence.
- Current/future coefficient privacy, current-atom visibility only in the
  coefficient stage, influence of earlier pairs, first-step gradients in both
  stacks, and distinct generated supports were checked.
- Full 420M model: dense versus cached predictions across all 256 events, using
  future placeholders. Maximum absolute error was 8.11e-6 for support logits and
  7.15e-6 for coefficients.
- Actual three-GPU full-state resume restored all 582 Adam parameter states and
  three rank RNG records at step 8, then advanced optimizer and scheduler to
  step 10. These benchmark weights are excluded from production initialization.
- The fresh production run and resumed benchmark produced identical logged
  support CE, coefficient CE, coefficient KL/entropy, and total loss at step 10;
  their gradient norms differed by less than 2e-8.
- Predecessor runs were preserved at completed epochs 211 / step 13,082
  (T0.0625) and 105 / step 6510 (T0.25), including last, best, optimizer,
  scheduler, and rank RNG state. An independent rollback process keeps the old
  launches recoverable until production verification succeeds.
- The new run logs FID mean/covariance terms, generated/real covariance traces,
  variance ratio, held-out losses, and sample grids. Full last/best checkpoints
  are uploaded with remote digest/size verification.

The first production checkpoint completed at epoch 1 / step 62. All model and
optimizer tensors were finite; all 582 optimizer states and the scheduler were
at step 62, with three rank RNG records. First-epoch FID50k was 147.3116, an early
pipeline check rather than a mature quality comparison.

`production-local-accepted.json` records the validated launch and local recovery
checkpoint. The active source archive is committed online with independently
verified digests. Large checkpoint uploads continue asynchronously.
`production-accepted.json` initially marks that upload verification as pending;
the running `verify_production.py` replaces it with the committed checkpoint
artifact and verified remote digests after the transfers complete. The completed
local launch validation releases the predecessor rollback guard.

The source archive and manifest are uploaded as training provenance. To rebuild
or resume, use the archived `train.py` and runtime: the ordinary base model
factory alone does not attach the two new sequence decoders.

Startup initially waited for predecessor uploads. It was restarted before any
optimizer update to move those uploads into a background thread. Fresh weights,
seed, data, optimizer, schedule, and the W&B run identity were retained. The
committed provenance version after this restart contains the active wrapper.

No claim of improved FID is warranted from startup tests or the first epoch.
Compare mature selected checkpoints using the established common-seed FID50k
protocol, including both the local original RQ run and the released RQ model.
