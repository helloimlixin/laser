# Review of the recovered 6 kbps benchmark

The user identified [oyg7smih](https://wandb.ai/helloimlixin-rutgers/laser/runs/oyg7smih),
a completed benchmark report from September 8, rather than a training run. Its
artifact contains 22 rate-defined LASER variants and eight baseline entries,
evaluated on 200 full VCTK recordings from eight held-out speakers. It includes
both 6 and 12 kbps results. Its six-kilobit audio-mode ViSQOL results include:

| System | Payload kbps | Container kbps | ViSQOL audio 48 kHz |
| --- | ---: | ---: | ---: |
| Released MDCTCodec | 6.0078 | 6.0602 | 4.277668 |
| LASER q8, 10k quantization warmup | 6.0078 | 6.0726 | 4.169022 |
| LASER four atoms, dictionary 64, q4 | 6.0078 | 6.0726 | 4.164528 |
| Separately trained MDCTCodec RVQ control | 6.0078 | 6.0602 | 4.100055 |

The report's paired speaker bootstrap gives LASER q8 minus the trained RVQ
control **+0.06897**, 95% interval **[0.05066, 0.09196]**. Against the released
MDCTCodec checkpoint it gives **−0.10865**, interval **[−0.14879, −0.07429]**.
These support different conclusions: a positive comparison against this trained
control, and a remaining quality gap against the released weights. The report
labels the trained RVQ control as trained simultaneously; exact equality of all
training conditions has not been independently audited here. The four-atom q4
variant is a valid 6 kbps alternative, but its difference from q8 is inconclusive
in the archived speaker bootstrap. The highest LASER score, 4.267743, is a
**12 kbps** variant.

The strongest archived 6 kbps recipe is training run
[ds6thjoj](https://wandb.ai/helloimlixin-rutgers/laser/runs/ds6thjoj). Its selected
checkpoint is epoch 224, step 383850. We recovered it from the run's selected
checkpoint artifact and verified SHA-256
`079e1e1c83e1d21bf6b520bc26eb28ccddf5faf0edd3f529726fe088bbfbfdee`
against the benchmark inventory. It loads strictly into the current model.

Relative to the current recovered K2 model, its substantive saved model settings
are 4,096 rather than 8,192 atoms, 8 rather than 7 coefficient bits, bound
42.32958984375 rather than 18.45128059387207, and a 10,000-update quantization
warmup rather than immediate quantization. Both use two atoms at 150 Hz and thus
40 raw bits per frame. Archived source uses **uniform256** coefficients; the
current model uses **signed127**. Directly resuming the old checkpoint with the
current default coefficient quantizer would change its representation and is
not an exact continuation.

Archived tracked model source was reconstructed from the report's diff and base
commit. Encoder, decoder, sparse encoding/reconstruction and OMP functions match
the current implementation. The evaluation helper disables the internal
coefficient quantizer and applies the appropriate exact quantizer externally,
then decodes tokens parsed from an actual five-byte-per-frame raw payload. No
running training process or training configuration is modified.

Only **16 of the archived 200 test files** occur in the current 200-file test.
Their overall scores therefore cannot establish a current-versus-old model
improvement. Twelve archived test files also occur in our recently prepared
128-file validation manifest. The new comparison removes those twelve and
excludes both test sets, leaving **116 validation recordings** across eight
speakers. This does not alter existing training checkpoint selection or either
locked test manifest.

`scripts/compare_mdctcodec_recovered_recipe.py --diagnostics` evaluates both
selected checkpoints on that shared validation set and computes a paired
speaker bootstrap. It also evaluates current floating coefficients and a dense
bottleneck bypass as diagnostics. Neither diagnostic is a 6 kbps codec; bypass
changes the decoder input distribution and is not a guaranteed achievable upper
bound. The current quantized evaluation reproduced all retained per-recording
ViSQOL values from the previous comparison exactly. The q8 payload helper passed
2,000 random/boundary round trips, exact five-byte size checks, and out-of-range
token rejection checks.

Recovered report, source, manifest, checkpoint provenance and comparison outputs
are under `outputs/mdctcodec_recovery/oyg7smih/`; the checkpoint itself is under
`outputs/mdctcodec_recovery/ds6thjoj/checkpoints/`.

## Completed shared-validation comparison

The [new online comparison](https://wandb.ai/helloimlixin-rutgers/laser/runs/0n5df495)
contains the frozen 116-file manifest, checkpoint hashes, per-utterance scores,
source verification, evaluation helper and aggregate results.

| Selected checkpoint / diagnostic | ViSQOL audio 48 kHz | STOI | Payload kbps |
| --- | ---: | ---: | ---: |
| Current q7, epoch 254 | 4.214836 | 0.898965 | 6.006953 |
| Recovered q8, epoch 224 | 4.175241 | 0.887929 | 6.006953 |
| Current, floating coefficients (diagnostic) | 4.219657 | 0.899143 | Not a 6 kbps codec |
| Current, dense bypass (diagnostic) | 4.227501 | 0.919435 | Not a 6 kbps codec |

Current minus recovered ViSQOL is **+0.039594**, with paired speaker-bootstrap
95% interval **[0.026779, 0.053705]**. This favors continuing the current recipe
over replacing it with this archived checkpoint. It is a comparison of selected
models, not a controlled estimate of the effect of dictionary size or warmup.

Floating coefficients improve ViSQOL by 0.004821, interval
[−0.000090, 0.012335]. Dense bypass changes it by 0.012666, interval
[−0.018497, 0.052582]. These diagnostics provide no clear evidence that removing
either bottleneck operation alone produces a large ViSQOL gain with this frozen
decoder. They do not establish a ceiling on improvements after retraining.
The active 300-epoch schedules and five-epoch checkpoint uploads remain in place.

## Trained RVQ control on the current locked test set

Recovered training run [48sl14ex](https://wandb.ai/helloimlixin-rutgers/laser/runs/48sl14ex)
contains the exact RVQ checkpoint used in oyg7smih: epoch 193, generator update
165,482. SHA-256 is
`d23f5e1fd13dc5bedb6146becfea5f5f15c7cca419e20e6a98043ae859b04af6`.
`scripts/benchmark_mdctcodec_trained_rvq.py` strictly loads all encoder, decoder
and four-codebook quantizer weights into the verified authors' architecture.
It serializes four 10-bit indices per frame and synthesizes from parsed codes.
No continuous latent is passed around the payload. The payload passed random,
boundary, rate and invalid-index checks; parsed-code latents matched the forward
quantizer output within numerical tolerance on the first real test utterance.

The [new RVQ evaluation](https://wandb.ai/helloimlixin-rutgers/laser/runs/tu44anv2)
scores **4.109210 ViSQOL**, **0.889778 STOI**, and **6.007956 kbps** on precisely
the current locked 200 recordings. The archived 4.100055 result used a different
manifest. All model selection remains based on validation.

The [updated interim report](https://wandb.ai/helloimlixin-rutgers/laser/runs/4xzrqlym)
uses the already evaluated LASER epoch-241 checkpoint, scoring 4.192316. Its
ViSQOL difference from trained RVQ is **+0.083106**, paired speaker-bootstrap
95% interval **[0.049052, 0.125593]**. Its difference from released MDCTCodec is
**−0.090100**, interval **[−0.119388, −0.062452]**. This is an interim comparison
of checkpoints with different training histories and update counts, not a
controlled estimate of the bottleneck architecture's effect.

Both final evaluation watchers now include this completed baseline via
`--baseline-dir outputs/mdctcodec_benchmark_trained_rvq`. The training processes
continue unchanged. The original run's final evaluator uses GPU 0 after that
training process ends; the lower-LR evaluator uses GPU 1. New logs are
`finalizer_with_trained_rvq.log` in each training directory. Final reports now
use 5,000 paired speaker-bootstrap draws and explain the difference between
released baselines and the recovered trained RVQ checkpoint.
