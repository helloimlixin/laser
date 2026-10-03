CC3M text-to-image stage 2 is launched through a detached supervisor on six H100
80 GB GPUs. It completes and verifies the full compound token cache before
starting a fresh 100-epoch prior. Preparation progress and training share this
[online W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/cc3m-rfid421-compound-650m-6h100-20260923).

The frozen ImageNet LASER checkpoint is epoch 10, reconstruction FID 4.210914,
from artifact `helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9`,
file `best_rfid_slot1_model.pt`. Its SHA-256 is
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.
This reconstruction score is provenance, not a CC3M generation result.

The [official CC3M recipe](https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/cc3m/cc3m-rqtransformer-8x8x4-650M.yaml)
provides width 1280, 26 spatial layers, four depth layers, 20 heads, and a
32-token prefix with the released 16,384-token BPE vocabulary. Text BPE dropout
is 0.1 during training. Its deterministic token IDs and raw captions are cached;
fast CPU BPE dropout is applied to captions at training time. Text/image loss
weights are 0.1/0.9. AdamW uses betas (0.9, 0.95), weight decay 1e-4, gradient
clipping at 1, no warmup, and a 100-epoch cosine schedule to zero.

The six-GPU effective batch is 2,040: 34 images per GPU and ten accumulation
steps. This is 0.39% below the official batch 2,048.
Peak learning rate is proportionally scaled to 0.000498046875. Each epoch uses
1,424 complete updates; its remaining 994 examples are dropped from a newly
shuffled order. The training seed is zero. Computation uses BF16, SDPA attention,
fused AdamW, and DDP with synchronization on the last accumulated microbatch.

The compound representation and objective match the archived
[FFHQ FID 8.1743927 run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803).
The 8 × 8 × 4 latent holds 256 complete atom/coefficient events. Each event
predicts `p(atom | text, earlier pairs)` followed by
`p(coefficient | text, earlier pairs, current atom)`. Both spatial and depth
history consume the complete earlier pairs. Coefficient prediction uses the
selected atom's frozen dictionary vector in the same two-layer causal
micro-transformer, with four independent 2,048-bin coefficient heads. Pair
embeddings include physical contributions and the learned FFHQ adapter.
Atom loss weight is 1.5; normalized soft coefficient targets use temperature
0.5 and stochastic context tokens. The distribution geometry weight is 0.05,
with top-4 atoms, an epoch-2 start, and a three-epoch ramp. Training leaves atom
logits unmasked; generation prevents repeated OMP supports. The resulting model
has 705,039,360 parameters, including the extra coefficient-conditioning layers.

The verified `pixparse/cc3m-wds` mirror at revision
`46f3d69f840e59d77d52e8decfe5baec97e94c7f` contains 2,905,954 training pairs in
576 shards and 13,443 validation pairs in 16 shards. This is the available
downloaded subset, not every original CC3M URL. The builder stages shards to
local disk with bounded shared-storage reads and verifies each SHA-256 before
decoding. Direct concurrent tar reads exposed shared-mount allocation failures;
the local staging path resolved them. Cache shards are individually resumable.

The cache uses one deterministic resize/center-crop 256 view per image, following
the successful cached FFHQ formulation. This intentionally differs from online
random image crops in the released CC3M training pipeline. Encoder and OMP
computation, and stored coefficients, are FP32. The full training maximum at
each sparse depth maps to normalized magnitude 3. Coefficients are never clipped;
validation uses the training scales. Atom IDs and coefficient values stay paired
with captions throughout extraction and merging. The complete compact cache is
copied to local disk and materialized in RAM once per training rank.

Every epoch, evaluation generates one image per available held-out caption,
without distributed-sampler padding or duplicates. FID uses the original
RQ-VAE Inception implementation against statistics computed from the same
13,443-image validation subset and fixed resize/center-crop transform. These
scores should not be presented as the released paper's full-split FID protocol.
CLIP score is the mean raw cosine similarity from OpenAI CLIP ViT-B/32, as in the
upstream evaluation; long captions are truncated to its supported context.
Sampling uses temperature 1, official CC3M atom top-k 16,384/top-p 0.7, and the
FFHQ coefficient top-p 0.85. Text-captioned previews are logged every 500 updates.

The complete latest recovery checkpoint is saved and queued online after the
first update, every 100 updates, and every epoch. It includes model, optimizer,
scheduler, epoch, data cursor, and all six Torch RNG states. BPE dropout uses an
independent tokenizer RNG whose stream restarts on resume. Serialization first
writes an immutable local file; a bounded worker persists it to shared storage
and queues upload while training continues. Copy failures propagate to training.
SIGTERM/SIGINT request a save after the next completed optimizer update.

Online checkpoint files are `last.pt`, `best-fid-01.pt`, and `best-clip-01.pt`.
Best FID and highest CLIP are retained independently, as model/config snapshots,
when a completed evaluation improves them. No production best checkpoint exists
before the first evaluation. Fixed filenames replace previous online versions.
Large uploads are asynchronous and can trail the newest local save.

The launch passed focused tests for archived FFHQ parity, text and pair
causality, cached generation, validation partitioning, cache merging, and
checkpoint ranking. Full-size CUDA validation completed finite forward/backward
updates, a model/optimizer save and reload, conditioned generation, and both
metric networks. A separate full six-GPU preflight completed one 2,040-image
update with equal post-update parameter probes, finite gradients on every rank,
and six RNG states. Preflight weights are never used for production. The actual
8,461,179,687-byte recovery-state persistence path was also verified. A separate,
explicitly labeled W&B transport-test run verified all three upload filenames
by size and MD5; those tiny transport files contain no production metrics.

Recipe: `configs/stage2/cc3m-rfid421-compound-650m-6h100.yaml`.
Persistent directory: `outputs/cc3m-rfid421-compound-650m-6h100-20260923`.
Local execution directory: `/mnt/laser-cc3m/runtime`.
Source snapshots, a SHA-256 manifest, dependency versions, checkpoint provenance,
preflight reports, and the launch command are retained in the persistent
directory. Authentication is stored privately in `/root/.netrc`, outside those
files. The machine must remain running for preparation and training to continue.

Inspect progress:

```bash
cat outputs/cc3m-rfid421-compound-650m-6h100-20260923/status.json
tail -f outputs/cc3m-rfid421-compound-650m-6h100-20260923/cache.log
tail -f outputs/cc3m-rfid421-compound-650m-6h100-20260923/training.log
```

Restart a stopped supervisor with the persistent `resume.py`. It restores the
snapshotted source and cached assets if local files are missing, reuses completed
cache shards, and resumes only this run's own latest stage-2 checkpoint.
