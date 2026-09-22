# Matched Church stage-2 target-temperature trial

The user approved trying stronger stage-2 regularization after the distortion
audit found early overfitting and nearly hard integer-code targets. This trial
changes training target temperature from 0.125 to 0.25. It retains all raw
coefficient levels without clipping or normalization.

The temperature widens the geometric target probabilities and changes the
stochastically sampled code histories during training. It does not alter
sampling temperature, code ordering, or the coefficient values in the book.

Both arms start from the same preserved integer-model checkpoint: epoch 22,
step 1364, FID50k 12.352177. Model, AdamW, cosine schedule, and all five rank RNG
states were compared bit-for-bit against that source and passed. The trainer
is identical between arms; only the temperature parameter and run/output
identifiers differ. The common source is deliberately earlier than the latest
main-run state so the ablation starts before further overfitting.

Each arm runs six epochs, ending at epoch 28. The τ=0.25 arm runs first, then
the τ=0.125 control. All five H200 GPUs are used with global batch 2048 and the
same accumulation, optimizer schedule, dropout, tokenizer, dictionary, raw
coefficient book, full 126,227-image data set, and preprocessing as the original
integer run. Training targets are sampled afresh on every visit.

The primary comparison is mean official FID50k across the last three epochs;
best achieved FID and common-target validation KL are also reported. Both
arms are monitored with τ=0.125 and identical fixed RNG streams, so monitoring
losses are comparable despite differing training target distributions. FID
uses all training images as the reference under the matching cache-input
transforms. Sampling remains temperature 1, top-k 1400, top-p 1. FID and a
64-image preview are logged every epoch. Full best and latest states are
uploaded and their online checksums verified.

The main run stops at a completed epoch and drains its final checkpoint uploads
before the trial starts. Its latest and best states are preserved by immutable
local hard links and verified W&B artifacts. The trial supervisor resumes that
main run after both arms finish their GPU work; trial artifact uploads can
finish concurrently. Trial results do not automatically replace the main run.

Files: `outputs/church-integer-targettau-ab-20260922`. Initial-state proof is
`initial-state-verification.json`; original-state preservation is recorded in
`main-handoff.json`. Each arm has `train/status.json`, `evaluations.jsonl`,
`heldout.jsonl`, samples, full checkpoints and upload verification receipts.
On completion the supervisor writes `comparison.json`, publishes `report.md`
and a W&B comparison table, and records `original-resumed.json`.

Online runs:

- [Comparison and distortion evidence](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-integer-targettau-comparison-20260922)
- [τ=0.25 trial](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-integer-tau025-trial-e22to28-20260922)
- [τ=0.125 control](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-integer-tau0125-control-e22to28-20260922)

The comparison is a bounded continuation experiment, not a completed
from-scratch recipe comparison. A lower FID does not by itself prove that
structural distortions have been resolved; inspect the fixed previews too.

## Autoregressive dependence check

The model generates 256 integer tokens in `(row, column, residual depth)`
order. Each token jointly selects an atom and one of its two fitted raw
coefficient levels; zero is an additional token. This is finite coefficient
quantization, not arbitrary continuous coefficient prediction.

The released spatial transformer receives summed code embeddings from earlier
locations, shifted by one location. The depth transformer receives cumulative
embeddings from earlier depths at the current location, shifted by one depth.
Both attention stacks are causal.

`verify_causality.py` checked the actual epoch-22 trained weights in FP32 on
CPU. Replacing the current and all later tokens at six boundaries (0, 1, 3,
4, 127, 255) changed no prediction through the current token: maximum absolute
logit difference was exactly zero for every case. Changing a preceding token
did affect a later prediction (maximum difference 0.975774), serving as a
positive control. The receipt is `causality-verification.json`. This checks
causal dependence; the separate cached/full-forward audit checks sampling
implementation consistency.
