ImageNet stage 2 relaunch, September 24, 2026

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-ffhq-compound-1400m-7h100-20260923

This resumes the matching interrupted run from epoch 5, update 3,175, including
AdamW, cosine scheduler, and all seven RNG streams. Its generation FID is
56.9432624 from the first 50,000-image evaluation; stage-1 reconstruction rFID
is 4.2109141. The checkpoint and full ImageNet cache were checksum verified.
The cache contains 1,281,167 images, 1,000 labels, and shape 8 × 8 × 4.

The FFHQ FID 8.1743927 recipe retains full atom/coefficient pair autoregression,
compound embeddings, selected dictionary atom-vector conditioning, two
coefficient micro-transformer layers, and four depth-specific coefficient
heads. The 1,457,980,928-parameter model, objective, coefficient settings,
sampling settings, total batch, and 100-epoch schedule remain unchanged.

Seven NVLink-connected H100 80 GB GPUs run DDP with per-GPU batch 48 and
six accumulation steps (global batch 2,016). Peak LR remains 0.0004921875 with
the checkpointed cosine position. A batch-72 capacity probe exceeded memory;
48 is the largest fitting divisor of the per-rank 288-image effective batch.
The validated cache and frozen runtime run from local storage, with the cache
loaded into RAM. Persistent checkpoint copies overlap training.

The former run failed when training resumed after its epoch-5 evaluation.
A minimal reproduction showed PyTorch serialization retaining CUDA storage
through Python pickler cycles even after tensors were deleted; explicit garbage
collection released it. Checkpoint saves now collect those cycles, and the
generation boundary collects them before moving optimizer state to CPU.
This changes storage lifetime without changing tensors or the objective.

The isolated runtime also uses the repository's tested immutable checkpoint
payload writer. It validates the new persistent file and atomically replaces a
small symlink, avoiding overwritten multi-gigabyte payloads on shared storage.
It retains local upload copies and propagates background persistence failures.
The latest is saved every 100 updates and each epoch; best FID and best IS are
selected independently every five epochs from 50,000 class-balanced samples.
W&B remains online with fixed last.pt, best-fid-01.pt, and best-is-01.pt slots.

The existing online latest and best-FID files have matching local MD5 hashes
and were checked with authenticated HTTP range reads. New latest saves are
queued asynchronously, so online files may lag local saves during transfer.

Validation artifacts are under
outputs/imagenet-rfid421-ffhq-compound-1400m-7h100-20260923/relaunch-20260924/.
They contain the runtime diff, previous source/manifest/launcher, checksums,
20 passing repository tests, and 23 passing isolated-runtime tests. The
recovery probe resumes the exact checkpoint, saves at step 3,180, generates
images, then continues training through step 3,190 in a separate output folder.
Probe updates are not part of production training.

Inspect training.log and training.pid in the main run directory. resume.py
restores the verified runtime and assets, rejects duplicate/occupied-GPU
launches, and starts detached training. The host must remain running.
Credentials are kept in the private /root/.netrc, outside repository snapshots.

Full-scale recovery verification passed. The step-3,180 checkpoint reloaded
with all model and optimizer tensors finite, scheduler step 3,180, all seven
RNG states, and epoch-5 batch cursor 30. It persisted atomically in 93.42
seconds while generation and training continued through update 3,190.

Production is running as detached PID 5572. At update 3200, loss and gradient norm were finite and the latest measured compute window processed 929.1 images/second. All 324 active runtime files match the updated manifest; W&B reports the run as running. Latest online checkpoint transfers remain asynchronous.

Precision correction: the later [throughput audit](imagenet-throughput-2026-09-24.md)
verified that this runtime's inner `amp=False` disabled the outer BF16
context. Its transformer forward therefore used FP32. The later explicit
BF16 runtime, batch changes and measured throughput are documented in that
audit; the recovery and storage fixes described above are retained.
