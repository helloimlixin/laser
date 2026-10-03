The Church run stopped making optimizer progress after its step-20,210 log.
Epoch 40's 50,000-image FID had completed at 13.330741; the subsequent wait
was checkpoint I/O, not FID generation. Rank 0 was idle while the other three
H100s reported 100% utilization at approximately 120–130 W, consistent with
distributed synchronization waits rather than useful model computation.

The checkpoint uploader rejected valid local serializations because the
shared filesystem finalized each immutable payload's ctime after its receipt
was written. Device, inode, size, and mtime were unchanged. The strict identity
comparison therefore fell back to shared storage and copied the 4.86 GB best
checkpoint into a local W&B slot again. Open file descriptors and a growing
temporary upload file confirmed that redundant copy. Rank 0's shutdown trace
located the main-thread wait in `checkpoint_writer.wait()` at the epoch
boundary, waiting for the background callback to finish.

The fix accepts a ctime-only change for explicitly enabled immutable
checkpoint storage, only when the receipt identifies the same unique payload
under `.checkpoint-data` and device, inode, size, mtime, and local file size
still match. Mutable checkpoints retain strict comparison. The local
serialization had already been checked against the committed persistent
payload before its receipt was written. This avoids rereading checkpoint
bytes from shared storage for uploads or best-checkpoint snapshots.

Fifteen checkpoint tests passed, including the ctime regression, rejection of
changed inode/device/size/mtime, preservation of strict mutable-file behavior,
immutable upload contents, interrupted persistence, and background transfers.
Only `_checkpoint_upload_source` changed in the frozen training source.
Model, optimizer, coefficient targets, geometry loss, batch size, sampling,
and data are unchanged. No new model-training preflight was necessary for
this isolated filesystem change; the previous four-GPU model preflight and
the new storage tests are recorded separately.

Recovery preserved full step-20,000 and best-FID epoch-40 checkpoints,
including optimizer, scheduler, and four RNG streams. The old torchrun group
was stopped, and the same online run resumed step 20,000. Approximately 213
updates after that checkpoint are replayed. This was a continuation, not a
restart from random weights. The new runtime is
`/mnt/laser-church/runtime-local-checkpoints`, selected by `resume-runtime.json`.
The original runtime and source archive remain available.

The source snapshot, manifest, and I/O preflight are in the active run's root
directory. Detailed diagnosis, recovery receipts, tests, online checkpoint
verification, and live utilization measurements are retained under
`checkpoint-io-fix/`. A temporary receipt-refresh guard used during diagnosis
was stopped during recovery; the resumed runtime contains the permanent fix.

Live verification passed step 20,760, beyond the new step-20,500
checkpoint and the next epoch boundary at step 20,706. The checkpoint
committed in 33.06 seconds in the background and queued both online slots
without the redundant shared-storage read. Across 28 samples showing all
four GPUs computing, median utilization was 99%, 99%, 99%, 98% and
median power was 587 W, 578 W, 604 W, 593 W. The measured
training window at verification was 1099 images/s. Full online bytes for the
step-20,000 latest and epoch-40 best checkpoints matched local MD5 values
at handoff; subsequent latest checkpoints continue through the same upload slots.
