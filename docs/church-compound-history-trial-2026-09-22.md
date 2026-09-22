This continuation trial was superseded by the user's clarification to train
from scratch. It stopped after completed epoch 59; the control never started.
The active replacement is the [fresh run](church-compound-history-scratch-2026-09-22.md).
The remainder records the cancelled experiment's design and verification.

Option 3 is implemented as a causal coefficient decoder with direct pair
history and a physical prefix reconstruction. It is an additive, zero-initialized
residual around the existing local coefficient conditioner, allowing a faithful
warm start from the best compound epoch-57 checkpoint. The old conditioner,
atom predictor, coefficient bins, and raw unclipped coefficient units remain.
This is a DCTransformer-inspired LASER experiment, not a reproduction of its
DCT tokenizer or three-decoder architecture.

Implementation:
[decoder](../src/models/compound_coefficient_decoder.py),
[compound integration](../src/training/compound_history.py),
[tests](../tests/test_compound_coefficient_history.py).

The maintained trainer exposes `--coefficient-history-layers 2` and
`--coefficient-history-width 512`; the default layer count is zero. Existing
checkpoints require the explicit state expansion used by this experiment before
resuming with the added parameters. New experiment checkpoints record these
settings and load strictly through the maintained model factory.

The new two-layer, width-512 decoder has 7,879,168 parameters. It uses eight
attention heads, residual dropout 0.1, and SDPA causal attention over all 256
compound events. Each position receives:

- the existing backbone state used to predict its atom;
- the current selected atom vector;
- the previous completed compound-pair embedding, shifted across spatial sites;
- the sum of physical contributions from strictly earlier depths at the current
  site, using the actual coefficient history available during generation.

The sequence decoder can attend directly to those earlier events. Its output
projection starts at zero and adds to the existing coefficient hidden state
before the unchanged depth-specific classifier. Neither the current coefficient
nor future pairs enter the current prediction. Prefix sums shift before
accumulating; subtracting the current contribution from a cumulative sum would
introduce avoidable numerical dependence on that current value.

Generation appends one event to the decoder KV cache after selecting its atom,
then samples its coefficient. The cache persists across all 8×8 spatial sites
and resets between image batches. Duplicate-support masking remains unchanged.
The retained 2,048 coefficient centers mean values remain quantized; no new
normalization or clipping is introduced.

Verification completed before production launch:

- 14 focused tests passed, including the existing full-pair tests and factory
  validation for incompatible modes.
- Full epoch-57 predictions match the original exactly at initialization in
  CPU FP32 and GPU BF16.
- With a nonzero output branch, current/future perturbations leave earlier
  logits exactly unchanged at events 0, 1, 3, 4, 127, and 255.
- The active branch's cached and dense coefficient predictions agree across
  all 256 events, maximum FP32 logit difference 1.90735e-5.
- Eight full five-GPU optimizer updates succeeded. All new parameters changed;
  original optimizer steps advanced from 3,534 to 3,542 and new parameter states
  advanced from zero to eight.
- The training preflight fit at approximately 113 GiB reserved per H200.
  End-to-end sampling at 2,048 images per GPU used 25.75 GiB peak allocated
  memory and took 8.93 seconds on the single-GPU preflight, including decoding
  the first 64 images. This is a throughput check, not a FID evaluation.

Evidence:
[initial checks](../outputs/church-compound-history-ab-20260922/initial-verification.json),
[GPU checks](../outputs/church-compound-history-ab-20260922/gpu-verification.json).

The matched experiment runs the history branch first, followed by the unchanged
compound control. Both begin from the preserved epoch-57 state and run epochs
58–63, six complete passes over all 126,227 Church training images. Shared
weights, all existing AdamW moments, the cosine scheduler, and all five rank RNG
states are restored. Only the new decoder parameters start with empty optimizer
state. The old parameter order is checked before expanding the optimizer group.

Both arms use global batch 2,048 on five H200 GPUs, no accumulation, and the
existing AdamW settings and 300-epoch cosine trajectory (57-epoch LR approximately
4.5677e-4). There is no LR restart, dropout change in the existing backbone, or
tokenizer change. New branch dropout necessarily consumes additional randomness,
so subsequent stochastic draws need not be identical across arms. This is a
one-seed controlled continuation, not a bitwise trajectory comparison.

Every epoch computes official RQ FID on 50,000 generated images against the
same full training reference, preserving the stage-2 cache's resize/center-crop
transforms. Sampling stays at atom temperature 1, top-k 250, top-p 1;
coefficient temperature 1, all 2,048 bins, top-p 1. FID batch size remains 2,048
per GPU to preserve the source evaluation layout. The separate coefficient-
temperature-1.2 sampling result is not substituted into this comparison.

Training/validation atom NLL and coefficient KL are monitored at every epoch
on the existing fixed probes, without removing any training images. The first
64 rank-zero FID samples are logged without selection or extra sampling.
Full last and best-FID checkpoints include optimizer, scheduler, and five-rank
RNG states, with online manifest digest and size verification.

The primary comparison is mean FID50k over epochs 61–63; best post-continuation
FID and validation losses are secondary. The source FID 9.9552 is preserved as
the initial best. The supervisor releases GPU state before final uploads drain,
allowing the control to start while the history arm finishes uploading. It then
publishes a comparison automatically. It does not restart the integer run or
automatically promote an unconfirmed winner.

Runs:
[history decoder](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-compound-history-e57to63-20260922),
[control](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-compound-control-e57to63-20260922),
[comparison](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-compound-history-comparison-20260922).
Current state and eventual numeric results are in
`outputs/church-compound-history-ab-20260922/status.json` and `comparison.json`.
