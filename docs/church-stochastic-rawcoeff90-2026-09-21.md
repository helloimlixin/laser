# Church stochastic compound training with raw coefficients

This experiment trains a fresh transformer using the validated physical
coefficient cache. Coefficients stay in OMP least-squares units, all four
depth scales are one, and one shared 2,048-bin Lloyd–Max codebook gives each
coefficient token the same physical meaning at every depth. The selected
stage-one tokenizer remains frozen. The previous transformer's coefficient
vocabulary is incompatible, so no stage-two weights are resumed.

Run: [church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921).

Output: `outputs/church-stochastic-rawcoeff90-20260921/`.

| Setting | Value |
|---|---|
| GPUs | All five available H200s, DDP |
| Batch | 192 per GPU, 960 global, no accumulation |
| Schedule | 90 epochs, 131 updates/epoch, 11,790 updates |
| Learning rate | 5e-4 cosine to zero, no warmup |
| Supports | Same 16 stochastic OMP alternatives per site, temperature 0.0625 |
| Training visits | Independent uniform choice of a complete trajectory per site |
| Coefficient targets | Soft physical-distance targets, temperature 0.125 |
| Objective | (1.5 × atom CE + coefficient soft CE) / 2.5 |
| Generation | Atom top-k 250; coefficient full vocabulary; temperatures and top-p 1 |
| FID | 50,000 samples at epochs 1, 10, 20, …, 90 |
| FID RNG | 20261921 + rank; training RNG restored afterward |
| Checkpoints | Full last and best FID, saved each epoch and every 500 steps |
| Previews | Every 500 optimizer updates |

The cache is
`outputs/church-physical-coefficients-20260921/compound-cache-physical.pt`,
SHA256 `437bf76107dd3da5661db6c766b474be43304f37705d6c8431f2e34ae3afa9fa`.
Preparation and quantization comparisons are documented in
[the physical-cache validation note](church-physical-coefficients-2026-09-21.md).
This trial changes both normalization and fitted coefficient bins. Relative
to the original eight-GPU trial, global batch also changes from 1,536 to 960.
Generated FID, rather than cross-entropy across different codebooks, determines
whether the new representation helps.

The run uses an isolated copy of the original training runtime, preserving
other work in the repository. Eleven focused model/cache tests pass, cache
and tokenizer SHA256 values match, the complete LR curve is monotonic with
the correct endpoint, and scheduler restoration preserves LR. A snapshot
test verifies that background checkpoint uploads remain valid after a later
checkpoint replaces the source file.

The detached supervisor first performs 12 real updates across five GPUs,
generates three previews, and validates the saved full checkpoint. Each
rank verifies unit scales, exact physical centers, frozen tokenizer state,
unclipped FP32 coefficients, and changing stochastic support selections.
Production starts from scratch after that isolated preflight succeeds.
Preflight checkpoint files are then discarded, preserving receipts/previews.

The preflight passed all 12 updates, three previews, five codec assertions,
and ten stochastic-selection checks. Approximately 48% of sites changed
their atom trajectory between independently sampled visits. All 259 Python
files in the frozen runtime match the completed continuation's runtime.
Production subsequently passed step 130 with finite metrics, unit coefficient
scales on all ranks, and LRs matching the intended cosine curve. Steady
training throughput is approximately 3,030 images/second across five GPUs.

The physical cache is published as its own verified W&B dataset artifact.
The frozen runtime and launch provenance are uploaded separately. Full best
and last checkpoint uploads use bounded copies on local `/tmp` storage;
durable checkpoint files remain under `/workspace`.

Verified cache artifact:
`helloimlixin-rutgers/laser/church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921-physical-compound-token-cache:v0`.
The published cache manifest matches the local 3,102,431,193-byte file and
MD5 `wcuyRst1byN2Or7br83s5A==`. The separate `training-provenance:v0` artifact
contains the frozen runtime and launch files. Local verification receipts
are `preflight-verification.json`, `online-launch-verification.json`, and
`production-verification.json`.

For current progress, inspect `status.json`, `train/metrics.jsonl`, and
`train/checkpoint-upload.json` under the output directory. `complete.json`
is written only after epoch 90 and the final verified checkpoint upload.

Epoch 1 completed with FID50k **159.01273567**. The first full `last.pt`
checkpoint saved successfully at step 131, but copying its best-FID snapshot
hit the workspace account quota and stopped the initial launch. The failed
partial best copy was discarded. Two redundant setup states from completed
experiments (the continuation's branch initialization and the original run's
preflight checkpoint, approximately 9.7 GB total) were copied to
`/tmp/laser-archived-intermediates/`, hash-verified, and replaced with symlinks.
All production best/last checkpoints remain under `/workspace`.

The epoch-1 best file was restored as an exact hash-verified copy of the valid
last checkpoint. Recovery checks validate all model tensors, optimizer step
131, cosine scheduler position 131, coefficient scales, and all five rank
RNG states. See `quota-recovery.json` and
`production-checkpoint-verification.json`. Resumption uses this state without
repeating epoch 1 or restarting the LR schedule.

Recovery succeeded: the run saved a new full checkpoint at epoch 2 / step
262, with matching optimizer/scheduler step counts and five RNG states.
`resume-verification.json` records this check. Training continued beyond step
320 at approximately 3,030 images/second under supervisor PID 13194. The
initial checkpoint upload was queued on restart and continues asynchronously.
