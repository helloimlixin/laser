# Lower-LR continuation from the stochastic compound model's best checkpoint

The parent run finished 90 epochs. Its best FID50k was **10.88615195 at
epoch 70 / step 5740**; epoch 80 scored 11.27253790 and epoch 90 scored
11.38630654. The LR at the selected checkpoint was 5.848888922e-5 and
subsequently decayed to zero. These results justify testing a smaller update
size, but do not establish that excessive LR caused the regression.

Parent: [stochastic compound 90 epochs](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-stochastic-compound90-h200x8-20260921).
Continuation: [best epoch 70, LR 2e-5, 30 epochs](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-stochastic-best70-lr2e5-30ep-h200x5-20260921).

The continuation restores the model and all AdamW moments and step counters
from the full epoch-70 checkpoint, then starts a new cosine schedule at 2e-5
and ends at 1e-6 after 30 additional epochs. The ending epoch is 100, not 120.
The source checkpoint's size and MD5 match `best-fid-01.pt` in the parent's
committed `selected-checkpoints:v7` artifact. Its SHA256 is
`f2d44f793953d3200ce6aeb98439f0e515724bd8774183cbc0944b5dc9eec170`.
An immutable local hardlink preserves that complete original state.

Following the user's instruction to prioritize throughput over preserving
the original effective batch, training uses **all five available H200s**,
192 images per GPU, global batch **960**, and **no gradient accumulation**.
Batch 192 follows the preceding H200 throughput benchmark. There are 131
optimizer updates per epoch, 3930 additional updates, and a final global
step of 9670. Consequently this experiment changes batch size as well as LR;
it is not an isolated LR ablation.

The eight-rank checkpoint is explicitly branched at an epoch boundary. The
first five saved rank RNG states initialize the new layout. This preserves
the model and optimizer state, but cannot reproduce the eight-rank random
trajectory. Subsequent checkpoints retain all five current rank RNG states.

The tokenizer, sixteen-variant stochastic support bank, Lloyd–Max centers,
loss, and sampling parameters follow the parent's frozen source and assets.
FID uses the same 50000 generated images and full Church training reference,
with seed 20261921 + rank and RNG restoration after evaluation. The starting
checkpoint is re-evaluated on the current five-GPU layout before training.
That matched baseline initializes this branch's best-checkpoint comparison;
the original parent's 10.88615195 score remains separate provenance. FID is
then measured at epochs 75, 80, 85, 90, 95, and 100.

The five-GPU preflight passed eight optimizer updates, two generated previews,
and full checkpoint validation. The matched starting FID50k is
**10.87463273**, close to the parent's recorded 10.88615195. All 259 Python
files copied from the parent's frozen runtime are unchanged; the new wrapper
and supervisor contain the continuation settings and provenance handling.

Experiment directory:
`outputs/church-stochastic-best70-lr2e5-30ep-20260921/`.

- `request.json`: source, online FID history, schedule, and GPU layout.
- `prepared.json`: source-artifact verification and checkpoint adaptation.
- `validation.json`: 11 passing focused tests, optimizer-state checks, and
  scheduler endpoint/reload checks.
- `preflight/complete.json`: eight real GPU updates and preview/checkpoint checks.
- `baseline/complete.json`: starting-checkpoint FID under the current layout.
- `status.json`: detached supervisor phase and process IDs.
- `train/metrics.jsonl`: training losses, LR, and throughput.
- `train/checkpoint-upload.json`: verified full latest/best online artifact.
- `complete.json`: written only after all 30 epochs and final uploads finish.

The supervisor runs preflight, baseline evaluation, and training in sequence.
Preflight updates are isolated from production. Checkpoints are saved each
epoch and every 500 steps, previews every 500 steps, and full latest/best
states are uploaded through the existing verified background uploader.

The first production launch hit the workspace quota while staging a hardlink
for upload, before taking any training updates. Disposable preflight/baseline
checkpoint links were removed after their successful checks. The branch's
uploader now makes bounded immutable upload copies under
`/tmp/laser-checkpoint-transfers/`; durable last/best checkpoints and the
original source remain under `/workspace`. An independent snapshot check
passed, and production was relaunched from the unchanged starting state.

Production ran online at approximately 3020 images/second across all
five GPUs. The first completed continuation epoch saved a full checkpoint at
epoch 71 / step 5871, with scheduler position 131 and five rank RNG states.
Logged LRs match the requested cosine curve. `production-verification.json`
and `online-launch-verification.json` record these checks.

The continuation completed all 30 additional epochs at epoch 100 / step
9670. Its final FID50k was **11.24451789**, and the best remained the matched
epoch-70 starting score **10.87463273**. No evaluated continuation checkpoint
improved that baseline. Full final and best checkpoint uploads were verified
in artifact
`helloimlixin-rutgers/laser/church-laser-stochastic-best70-lr2e5-30ep-h200x5-20260921-selected-checkpoints:v7`.
