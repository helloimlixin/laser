# LSUN Church with a 65,537-token joint sparse-vector vocabulary

**Launched from scratch on eight H100 GPUs on September 29 at 13:34 UTC.**
The eight-GPU preflight passed at 1.076 seconds per global update (about
1,903 images/second), with 67.60 GiB peak allocated memory. No generation FID
has been measured yet. Live run: [church-joint65k-20260929](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-joint65k-20260929).


The user requested a return to the earlier representation in which one integer
identifies an atom and its physical coefficient, with a modestly larger
codebook. This experiment prepares fresh stage-two training on LSUN Church
256 using that representation and the current frozen stage-one tokenizer.
The runtime is `/tmp/laser-church-joint65k-20260929`; durable output
is `outputs/church-joint65k-20260929` and the W&B run ID is
`church-joint65k-20260929`.

## One integer, one physical sparse-vector contribution

There are **65,537 classes**: one zero vector and 16,384 dictionary atoms with
four coefficient levels each. The integer encoding is
`0`, or `1 + 4 * atom_id + coefficient_level_id`. For each atom, the levels are
`[old_negative, old_negative / 2, old_positive / 2, old_positive]`. Its original
two signed levels are retained exactly; the two additional levels are obtained
by halving their magnitudes. Every old codeword remains available.

These are physical coefficients that directly multiply the corresponding
dictionary atom. The same four values for a given atom are used at every
residual depth, with no depth-specific normalization or scale. Different atoms
retain their previously fitted levels; this is not a common scalar grid shared
by every atom. No coefficient fitting, dictionary learning, OMP support fitting
or least-squares coefficient refitting is performed in preparing this book or
constructing its training targets.

Each 8×8 spatial grid contains four residual tokens per site. Decoding sums the
four selected physical vectors and applies the frozen stage-one projection and
decoder. Repeated atoms are allowed, as in the earlier integer residual
quantizer. Thus four residual terms need not contain four distinct atom IDs.
The maximum ID is 65,536, so serialized IDs require uint32 or a wider integer;
uint16 cannot represent this vocabulary.

The new codebook SHA256 is
`f0b74b14018ae6ad098806b9b63d0de64f50ccf41f086619ebb9ef3d06aeadcb`.
Its source two-level book SHA256 is
`812c10ba93167663cb56c19da0c1673f6aa88feb0bc264cf9d1f2e2f13d4b1cb`.
The frozen tokenizer SHA256 is
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
Preparation verifies all 65,537 embeddings, exact preservation of every old
codeword, exact uint32 round trips and a bitwise match to the source dictionary.
Its comparison to the current tokenizer's freshly normalized dictionary differs
by at most 2.98e-8, within the recorded floating-point tolerance.

## Model and training objective

The production model is native RQTransformer: 24 spatial transformer layers,
four depth layers, width 1,024 and 16 attention heads. It has **420,469,761
trainable parameters** across 460 parameter tensors and one shared
65,537-class output head. Stage-two weights, optimizer and schedule start fresh.
Physical codeword embeddings supply the cumulative residual-depth context.

Training uses the authenticated frozen encoder cache, with 126,227 arrays of
shape 8×8×256 in FP32. The teacher starts from each encoder vector and forms
probabilities over all 65,537 physical codewords:

`q(token | residual) ∝ exp(-||residual - codeword(token)||² / 0.125)`.

It samples one complete token from that distribution, subtracts exactly that
token's physical vector, and constructs the next depth's distribution from
the resulting residual. Fresh histories and soft targets are generated on
each visit. The model minimizes a single full-vocabulary soft cross-entropy;
there is no atom/coefficient loss-weight balancing. The teacher geometry and
loss calculations use FP32; target geometry disables TF32. Chunking controls
memory without restricting the target vocabulary. Temperature 0.125 is reused
from the earlier integer recipe; it has not been calibrated anew for this
four-level book.

The run uses eight H100 GPUs, global batch 2,048 and 300 epochs
(62 updates per epoch, 18,600 total). The selected layout is batch
256 per GPU with one optimizer update per local batch, without accumulation.
Exact epoch coverage retains the final 1,299-image update with sample-weighted
gradients. AdamW uses LR 5e-4, betas (0.9, 0.95), epsilon 1e-8, weight decay
1e-4, gradient clipping 1 and fresh cosine decay to zero without warmup.
Residual dropout is 0.1; attention and embedding dropout are zero. No geometry,
reward or contrastive objective is included.

## Sampling and publication plan

Native cached RQ ancestral sampling predicts one complete joint token at each
event. **Top-k 1,400 ranks whole atom/coefficient classes**, with temperature 1
and no nucleus filter. The zero class participates in the same vocabulary.
The full sampled vector enters the subsequent context.

The plan produces 64 preview images every 200 updates and evaluates 50,000
images every ten epochs (620 updates). Evaluation uses eight fixed logical
random streams, generation batch 1,024, decoder/feature batches 32, continuous
pixels, FP32 decoder/Inception and TF32 disabled. The reference SHA256 is
`809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`.
Mean and covariance contributions are retained separately. BEST selection uses
this fixed joint-token sampling policy.

Full LAST checkpoints are saved locally every 200 updates and around FID
boundaries; remote and durable publication starts at step 200, then every FID
boundary and completion. BEST is published after a measured FID selects it.
Checkpoint identity includes the codebook, tokenizer, source and plan, together
with model, optimizer, schedule, data cursor and per-rank RNG states. The source is verified in `church-joint65k-20260929-source:v0` (48 files),
and the exact new/old books, frozen tokenizer, official reference and preparation
evidence are verified in `church-joint65k-20260929-recovery:v0` (7 files).
The training cache is identified by checksum and can be regenerated with the
included frozen encoder; it is not duplicated in that recovery bundle.

The first64-image preview at step200 is present on W&B with matching image
bytes. The full LAST200 checkpoint (5,046,293,154 bytes) is committed and
verified in `church-joint65k-20260929-selected-checkpoints:v0`; SHA256 is
`7a998a64224d6a3157513986c3a0f65a947064f87d11b68feacd2f7f8945e23b`.
Its 460 Adam states are all at step200, and all eight model/teacher RNG states
are present. BEST is intentionally empty until the first measured FID at
step620. Production is continuing under the detached controller.

## Preparation evidence and limits

On the fixed 128-image validation reconstruction probe:

| Representation | Latent MSE | Pixel MSE |
| --- | ---: | ---: |
| Continuous OMP reference | 0.00951425 | 0.01499938 |
| Earlier 32,769-token book | 0.01774222 | 0.01592311 |
| New 65,537-token book | 0.01347106 | 0.01545984 |

The continuous OMP solve is used only to construct the diagnostic reference.
The new book lowers reconstruction error relative to the earlier compact book
on this probe, while remaining less accurate than that continuous reference.
These are reconstruction measurements, not generation FID. No stage-two
checkpoint was loaded and no optimizer updates were taken for this probe.

The stochastic teacher probe covers 1,024 training sites. At temperature
0.125, relative residual distortion is 0.04386477 for the new book versus
0.05614039 for the old book. New target entropies by depth are approximately
0.0823, 0.2442, 0.5454 and 1.2465 nats. These observations support the specific
preparation check; they do not establish a better generative training outcome.

`diagnostics/math-validation.json` verifies full-vector teacher probabilities
against a dense construction (maximum absolute error 7.90e-7), exact teacher
RNG replay, correct prefix residual updates, soft-CE values and gradients,
cached/full-forward logits (maximum absolute error 5.96e-7), causality and
sampler shape/range. The production head shape is verified as 65,537×1,024.
Strict resume also passed: four uninterrupted updates match three updates
plus a separate-process resumed update bit for bit across every weight, Adam
state, scheduler, cursor and model/teacher RNG. The eight-rank test verified
identical final weights on all ranks and finite gradients for all 460 parameter
tensors. Its three smoke updates are discarded; production starts fresh.

Generation BF16 was verified at the classifier output. A sampling/decode/feature
benchmark measured 89.3 images/second per GPU with batch 1,024, versus 48.3 with
batch 128. This new generation batching differs from the prior factorized runs.

Source receipts are `plan.json`, `book-preparation.json`,
`reconstruction-preflight.json` and `diagnostics/math-validation.json` in the
isolated runtime.

The [earlier integer run](church-integer-raw-recipe-2026-09-22.md) reached
best FID50k **11.8709209** at epoch 36. It establishes historical experience
with the integer-token RQ structure; it is not a matched control for this
larger vocabulary, current runtime and GPU/evaluation layout. The immediately
preceding [shared-grid factorized run](church-shared-physical-scratch-2026-09-29.md)
completed all 300 epochs with final FID **12.3279588** and archived BEST
**11.5018081** at epoch 120. Its full final LAST and BEST are verified in
`church-shared-physical-scratch-20260929-selected-checkpoints:v30`. Its
untruncated sampling policy and different prediction/target formulation also
prevent treating those scores as a controlled test of this book expansion.
