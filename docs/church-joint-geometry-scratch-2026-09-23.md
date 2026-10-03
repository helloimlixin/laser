The new [online Church run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-ffhq-compound-350m-4h100-jointgeom-scratch-20260923)
starts stage 2 from random initialization, seed 0. No stage-2 model, optimizer,
scheduler, or RNG checkpoint is loaded. It supersedes the continuation of
`church-ffhq-compound-350m-4a100-20260923` after the user's request to restart
because of the coefficient geometry bug. The original latest checkpoint at
step 26,203 and best checkpoint at epoch 45, FID 15.30345208056923, remain in the
original output directory and are verified online.

The [artifact investigation](church-coefficient-artifacts-2026-09-23.md)
documents the loss error and its correction. Full autoregression over both
atom IDs and coefficients, dictionary atom-vector conditioning, the two-layer
coefficient micro-transformer, depth-specific heads, coefficient soft targets,
and the original sampling settings remain. The 404,738,048-parameter model
uses corrected atom-conditional expectations in its geometry loss. The old
two-epoch geometry delay and three-epoch ramp are retained, as are the 300-epoch
cosine schedule and global batch 256.

Four H100 80GB GPUs run BF16 DDP, batch 64 each, SDPA and fused AdamW. The
validated 126,227-image 8×8×4 token cache is staged locally and loaded into RAM.
The frozen tokenizer is unchanged; cached reconstructions were clean. The
assets in the new output directory link to the preserved original assets.

The fresh initialization passed a two-update four-GPU preflight with geometry
enabled immediately for validation. Every parameter and optimizer tensor was
finite, and peak allocated memory was 36.15 GiB per GPU. These preflight weights
are not used in production. The 13 focused math/gradient/causality/FFHQ-parity
tests pass after the warmup optimization; an earlier checkpoint/regression
suite passed 27 focused tests.

The recipe is
`configs/stage2/lsun-church-ffhq-compound-350m-4h100-jointgeom-scratch.yaml`.
Resolved configuration, frozen source, checksummed manifest, preflight report,
launch record and logs are under
`outputs/church-ffhq-compound-350m-4h100-jointgeom-scratch-20260923/`.
The active runtime is `/mnt/laser-church/runtime-jointgeom-scratch`.

Full latest checkpoints are saved and queued for online upload every 500
updates and every five epochs. Best-FID checkpoints are full recovery states
and are uploaded when selected by the unchanged 50,000-image evaluation every
five epochs. The new run has no best-FID checkpoint until its first evaluation.
Previews are generated every 500 updates. Endpoint probability is logged at
each sparse depth to track the diagnosed failure mode.

Initial online verification reached step 360 with finite loss 8.4449 and
gradient norm 0.1196, at approximately 1,264 images/s during the original
geometry warmup delay. The first production recovery checkpoint at step 500
subsequently committed to persistent storage and was queued as online
`last.pt`; its preview was also generated. Fresh FID results are pending.

To recover this run after its first production checkpoint exists:

```bash
python scripts/tools/resume_church_ffhq_h100.py \
  --base /workspace/Projects/laser/outputs/church-ffhq-compound-350m-4h100-jointgeom-scratch-20260923
```

The helper selects the new run's frozen runtime and newest complete checkpoint,
restores all four RNG streams, and retains the online run identity. It refuses
to launch over a live supervisor or to silently start fresh without a recovery
checkpoint. Credentials remain outside the repository.

At step 8,037, this same run resumed with an equivalent fused two-token
coefficient attention kernel. The selected runtime is now
`/mnt/laser-church/runtime-fast-pair`; the recovery command above follows that
selection automatically. With the corrected geometry fully active, measured
training throughput improved from 488.54 to 1,096.54 images/s (2.2445×).
See [the kernel validation and deployment report](church-fast-pair-attention-2026-09-23.md).
