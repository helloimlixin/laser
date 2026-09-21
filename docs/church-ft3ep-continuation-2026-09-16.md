# September 16 Church continuation

The user subsequently authorized a fresh transformer run. Its settings and
launch are documented in [church-ft3ep-scratch-adaptive-2026-09-16.md](church-ft3ep-scratch-adaptive-2026-09-16.md).
The recovery findings below describe the unavailable previous run.

The requested run is [church-laser-ft3ep-scratch-20260916](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3ep-scratch-20260916).
**Training has not been relaunched: its transformer checkpoint is missing.**
W&B recorded a preview at epoch 167, optimizer step 10,354. The previous host
was `24c8069f0ff5`, and its checkpoint was saved at
`/workspace/Projects/laser/outputs/church-stage2-scratch-20260916/train/last.pt`.
That directory is absent from this workspace. The run's W&B files and artifacts
contain no model checkpoint. Searches of local checkpoints, Church model
artifact collections, and backup runs found other experiments' checkpoints,
which cannot continue this run's weights and optimizer history.

The exact three-epoch tokenizer and compact codebook are present locally.
Their SHA-256 hashes match this run's W&B configuration:

- Tokenizer: `1dbf4519a5c6248ec7f0ab3865511371a0370d247e3c5fc1bffb15492f476624`.
- Codebook: `b7a4bac5f80da85a1ff341e6782d66439a2fff0bcd88b35aca529919fe4843dd`.
- Restored frozen tokenizer state: `a4887de4981cb626cd645b21170181dce90e6ddd576e96524ba38b6ca0d400cd`.

The original RQVAE LSUN loader was exercised on the restored Church LMDBs.
All 126,227 training keys, 300 validation keys, and 72 saved pixel probes match.
The pipeline uses RGB conversion, bilinear short-side resize to 256, center
crop, tensor conversion, and normalization to [-1,1]. Its FP32 latent cache
hash also matches. Stochastic codes/soft targets are recomputed on each visit.
All 93 frozen stage-2 source files match their manifest. CPU normalization
rounding is handled by restoring the codebook's stored dictionary values and
requiring an exact hash match across the entire frozen tokenizer state.

The new driver, `scripts/tools/continue_church_stage2.py`, was derived from the
training source uploaded by this exact run. It keeps the 300-epoch horizon,
global batch 2,048 (256 per GPU, two GPUs, accumulation four), tokenizer,
target temperature 0.125, and sampling temperature 1/top-k 1,400/top-p 1.
Its continuation settings are:

- Halve the resumed cosine learning rate once, approximately 1.03e-4 at epoch
  167. Restore this multiplier unchanged on subsequent resumes. Evaluate FID50k
  every 10 epochs against the same full training reference. After two checks
  without at least 0.1 FID improvement, halve the multiplier again, with one
  check of cooldown and a 1e-6 floor. Improved quality is not established.
- Save full training state each epoch and commit a W&B model artifact containing
  `last.pt`, `best-fid-01.pt`, `best-fid-02.pt`, and `best-fid-03.pt` as available.
  New ranked checkpoints contain optimizer, scheduler, AMP, RNG, data cursor,
  and LR-controller state. A legacy best weights checkpoint is carried forward
  when supplied alongside the resume checkpoint and its protocol matches.
  Missing historical weights cannot be reconstructed from logged FID values.
- Pin immutable checkpoint files until the artifact commits, and prune
  superseded local FID states only afterward. Upload on initial restore,
  completed epochs, and graceful shutdown.
- Generate a fixed-seed 100-image 10 × 10 preview every 200 optimizer steps, preserving
  training RNG. FID diagnostic grids also use 10 × 10 images.

Once this run's checkpoint and any retained best checkpoints are restored,
launch with:

```bash
.venv-imagenet-stage2/bin/python scripts/tools/launch_church_ft3ep_continuation.py \
  --checkpoint /path/to/this-runs/last.pt
```

The launcher checks resume metadata before starting two GPU workers and uses
W&B `resume='must'`. It reads the credential from `WANDB_API_KEY` or the private
credential file outside this repository. It rejects missing checkpoints.
An actual checkpoint may require path metadata reconciliation after transfer;
the launcher deliberately requires matching cache and calibration identities.

Nineteen continuation/recovery tests passed, covering noncompounding LR
scaling, LR state recovery, FID plateau decisions, top-three selection,
immutable upload files, failed upload receipts, exact stochastic training
recovery, and a CUDA preview test for grid dimensions and RNG preservation.
The CUDA preview test uses a test model; a production resume preflight remains
pending the missing checkpoint. No production checkpoint upload has occurred.
The broader legacy FID test module could not collect because this environment
lacks its unrelated matplotlib dependency.

Search results, W&B snapshots, and verification receipts are under
`outputs/church-ft3ep-resume-20260916/`.
