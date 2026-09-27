# Church LASER three-epoch continuation on Amarel

Submitted job **61690160** for four A100 GPUs (two nodes with two GPUs each),
partition `gpu`, 72-hour limit. Check the scheduler for current state; submission
and CPU validation alone do not establish healthy GPU training.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3ep-scratch-adaptive-lr-20260916

Run directory:
`/scratch/xl598/runs/laser/church-laser-ft3ep-scratch-adaptive-lr-20260916-amarel`

The recovered `model-church-laser-ft3ep-scratch-adaptive-lr-20260916-selected-checkpoints:v93`
contains epoch 91, optimizer step 5,642, at an epoch boundary. The allocation
resolves the run's latest artifact once and stages that exact version on each
node. The resumed run retains its original W&B identity.

The complete three-epoch tokenizer was recovered from
`church-laser-consistent-rq32k-latest:v0`, together with the compact codebook,
temperature calibration, frozen source, and original real FID statistics. The
tokenizer export records epoch 3 and 2,961 updates from source checkpoint SHA256
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.
Its archived stage-one driver identifies the starting ImageNet reconstruction
FID as 4.210914134979248. Stage one is already complete.

## Training and hardware

- Strictly restore all 386,882,561 model parameters and 460 AdamW state entries,
  cosine scheduler, AMP scaler, FID LR multiplier, and available per-rank RNG.
- Preserve global batch 2,048, 62 updates per epoch, and the 300-epoch schedule.
  Four GPUs use local batch 32 and 16 accumulation steps. Local batch 16 and
  world sizes 8 or 16 are supported; changing batching requires an epoch boundary.
- Resume LR is 0.00019741014654283464; saved multiplier is 0.5. The next cosine
  update was verified as 0.00019739293583602796.
- Rebuild the missing FP32 encoder-latent cache on allocated GPUs. The exact
  tokenizer state is frozen; stochastic soft targets remain recomputed per visit
  at temperature 0.125. A rebuilt cache is recorded as such, not claimed bitwise
  identical to the former H200 cache.
- Use two loader workers per rank, node-local LMDB and training latents, and
  node-local W&B staging and full checkpoint files.
- Checkpoint and upload every epoch; keep the best three checkpoints under the
  current FID protocol. Save and upload before the allocation time limit.

## Evaluation and interpretation

Validation covers all 300 held-out images every five epochs. FID covers exactly
50,000 generated images every ten epochs against the byte-identical original
126,227-image real reference. Sampling remains top-k 1,400, top-p 1, temperature
1. Decode and Inception use batches of eight. Previews contain 100 samples every
200 optimizer steps.

Generation is distributed in fixed global batches of 100, each seeded with
71,000 plus its global batch index. Assignment to ranks does not change this
sampling protocol. This changes the original two continuous RNG streams. The
first new FID therefore establishes a fresh plateau-comparison baseline while
retaining the saved LR multiplier. Historical best FID 11.891982117351915 at
epoch 80 and the previous top-three checkpoints remain in pinned artifact v93;
their provenance is written to `train/historical-fid-selection.json`.

The user suspects exposure bias. This continuation preserves the saved training
objective. Validation prediction loss and generation FID can help characterize a
recurring plateau; an LR reduction does not establish that exposure bias was
resolved. A new exposure-bias objective would be a separate experiment.

## Validation and operation

`preflight.json` records strict restoration, optimizer shape/step checks,
analytic next-LR agreement, plateau control behavior, exact tokenizer state,
exact reference statistics, supported batching, and FID partition counts. Both
LSUN splits match archived key order and 36 transformed-pixel probes each.
Container Python compilation and launcher syntax checks passed.

Launcher: `scripts/submit_church_laser_ft3ep_resume.sh`. Runtime uses isolated
copies in `RUN_BASE/runtime` and frozen dependencies in `RUN_BASE/stage2-source`.
Other live experiments are independent.

A detached Codex supervisor records its session in `monitor/session.json`,
current findings in `monitor/status.json`, and fixes in `monitor/actions.jsonl`.
It watches queueing and startup, repairs scoped launch failures, and stops only
after recording allocated GPU identity and increasing finite-loss training
steps over at least 60 seconds, or after documenting an explicit blocker.
