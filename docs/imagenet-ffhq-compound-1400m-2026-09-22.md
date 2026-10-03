ImageNet class-conditional stage 2 uses the original rFID-4.210914 tokenizer and
the successful FFHQ-v4 compound formulation, scaled to the released 1.4B
RQ-Transformer. The supervisor waits for verified ImageNet archives and extraction,
builds the full unclipped FP32 token cache, and starts a fresh prior automatically.

The stalled torrent download was replaced on September 22. Validation and devkit
archives came from official HTTPS endpoints. The training archive came from the
public `HeyEffCV/ILSVRC2012` mirror at revision
`ef12fdbfdf9cef006c08d47e3efa8a1dcdb6a385`: all four parts downloaded in 153 seconds,
and their assembled archive matched the official MD5
`1d675b47d978889d74fa0da5fadfb00e`. Provenance is recorded in
`assets/imagenet-data-source.json`. All three official archive MD5 checks passed
before extraction. Completed verification receipts bind each checksum to the
local file's size, modification/change timestamps, and inode.

Archives and extracted images use local disk at `/mnt/laser-imagenet`, avoiding
shared-storage latency. Resumable extraction uses 16 separate CPU processes.
Inspect `/mnt/laser-imagenet/logs/setup.log` for preparation progress. W&B records
extraction progress, and the supervisor reports a failed dataset process instead
of waiting indefinitely after it exits.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-ffhq-compound-1400m-20260922

Recipe: `configs/stage2/imagenet-ffhq-compound-1400m.yaml`.
Output root: `outputs/imagenet-rfid421-ffhq-compound-1400m-20260922`.

The September 23 performance revision computes detailed diagnostics only on
the final microbatch of each logged update, avoiding 159 unused diagnostic
computations per logging interval. The optimizer, losses, batch sizes, token
formulation, and sampling protocol are unchanged. Checkpoint serialization also
retains a local immutable inode under `/mnt/laser-checkpoint-upload-cache`.
The persistent recovery checkpoint is still committed before uploads are queued;
W&B uploads use local hard links rather than an additional shared-storage copy.
File identity receipts reject stale local copies during upload or resume.
The revision's tests and measurements are in `performance-20260923/`.
The run resumed at update 1,100 with optimizer, schedule, data cursor and RNG
restored. Subsequent steady windows measured 257–258 images/second versus about
255 before the revision. The larger expected wall-time benefit comes from
eliminating the extra 17,496,884,165-byte copy on every checkpoint save; it is
not a large change in GPU compute throughput. The online `last.pt` for update
1,100 was verified against the complete local file's MD5.

The frozen stage-1 source is `best_rfid_slot1_model.pt`, epoch 10, from W&B artifact
`helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9`.
Its SHA-256 is
`dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab`.
The 4.21 figure identifies that source checkpoint's reconstruction FID; it is not
a measured compound-token reconstruction FID or a generation result for this run.

The historical recipe comes from `ffhqcmp0804205803`, whose recorded FFHQ
FID-50k was 8.1743927. Its W&B configuration and archived source were checked.
The archive SHA-256 is
`9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f`.

The model has **1,457,980,928 parameters**: width 1536, 42 spatial layers,
six depth layers, 24 attention heads, two atom-conditioned coefficient blocks,
and four independent 2,048-bin coefficient classifiers. There are 16,384 atom
classes, 1,000 conditioning classes, and 256 complete pair events per image.
The coefficient blocks contain 56,663,040 parameters and their classifiers
contain 12,603,392. Both spatial and depth contexts consume shifted complete
atom/coefficient pairs. The atom predictor has the archived unmasked training
logits; generation enforces distinct OMP supports.

The full training maximum at each depth determines its coefficient scale,
mapping that maximum to normalized magnitude 3. Extraction uses FP32 encoder
and OMP computation, stores FP32 coefficients, and never clips coefficients.
The historical normalized soft-target temperature is 0.5, with stochastic
coefficient contexts. Atom loss weight is 1.5; distribution geometry weight
is 0.05 with a two-epoch delay and three-epoch ramp. These retain the historical
target noise, which is distinct from deterministic coefficient quantization.

Optimization follows the released ImageNet schedule: 100 epochs, effective batch
2,048 (microbatch 128 on one B200, accumulating 16 times), cosine learning rate
5e-4 to zero, no warmup, AdamW betas (0.9, 0.95),
weight decay 1e-4, and gradient norm clipping at 1. Gradient clipping is separate
from the disabled coefficient clipping. Stage 2 starts from random weights.
Preflight weights and synthetic data are never used for production training.

The cache contains all 1,281,167 training images with sorted official WNID labels
and one deterministic resize/center-crop view per image. This follows the cached
FFHQ data formulation. No latent flips or invented additional image views are used.
The supervisor requires the dataset setup's `READY.json` with exact train/val
counts, checks the stage-1 hash and class coverage, and verifies cache encoding
and pair decoding before training. The local preparation process completes
archive verification and extraction first. After cache verification, stage 2
loads the complete sparse cache into RAM once (`token_cache_in_ram: true`);
training batches do not reread source images or page tokens from shared storage.

FID-50k runs every five epochs against the released ImageNet-256 training
reference statistics, using the original RQ-VAE Inception implementation and
uniform class labels. Sampling uses temperature 1, atom top-k 2,048 / top-p 1,
and coefficient top-p 0.85. Atom top-k scales the historical FFHQ vocabulary
fraction (250/2,048) to the 16,384-atom ImageNet dictionary. W&B receives labeled
64-image previews every 500 optimizer updates.

Latest recovery state is saved and uploaded every 100 updates and every epoch.
The online run files are `last.pt` (full model, optimizer, scheduler, RNG, and data
cursor) and `best-fid-01.pt` (the best FID-50k model and configuration). Best files
are uploaded when an evaluated model improves. No best-FID checkpoint exists
before the first completed FID evaluation. Fixed online names avoid accumulating
large immutable checkpoint versions. SIGTERM/SIGINT request a checkpoint after
the next completed optimizer update. Single-GPU resume preserves the shuffled
epoch order and sample cursor.

The supervisor executes a source snapshot with a SHA-256 manifest. Its launch
configuration, source manifest, capacity report, preflight report, and checkpoint
provenance are also uploaded as a W&B run-config artifact. Credentials live in
the private user `.netrc`, outside the repository and source snapshot. The
environment must remain running for the supervisor and training to continue.

Validation includes exact archived/current logits, objective and gradient parity;
class conditioning; four-depth pair causality; checkpoint retention and upload;
and 32 passing launch-focused tests. An earlier broader CLI run had two existing
archive recipe assertions expecting `ddpm` while the unchanged YAMLs specify
`rqvae`; those stage-1 tests are unrelated to this stage-2 launch. The exact
full-model CUDA/reload/generation results are in `preflight/complete.json`.
The 128-image microbatch passed with finite loss and gradients and 105.35 GiB
peak allocated memory. The saved model and optimizer reloaded successfully,
all state tensors were finite, and class-conditioned generation decoded valid
images. These synthetic-data checks establish integration, not image quality.
Online upload transport was independently checked for both fixed checkpoint slots
in a separate, explicitly labeled transport-test W&B run.

Inspect the pipeline and current training:

```bash
cat outputs/imagenet-rfid421-ffhq-compound-1400m-20260922/status.json
tail -f outputs/imagenet-rfid421-ffhq-compound-1400m-20260922/production.log
tail -f outputs/imagenet-rfid421-ffhq-compound-1400m-20260922/training.log
```

To restart a stopped supervisor, use the snapshotted script and recipe:

```bash
python outputs/imagenet-rfid421-ffhq-compound-1400m-20260922/runtime/scripts/tools/run_imagenet_ffhq_compound.py \
  --base outputs/imagenet-rfid421-ffhq-compound-1400m-20260922 \
  --config outputs/imagenet-rfid421-ffhq-compound-1400m-20260922/runtime/configs/stage2/imagenet-ffhq-compound-1400m.yaml
```

The supervisor lock rejects duplicate instances, and the recipe resumes only
its own production checkpoint if one exists.
