# Church: larger effective batch for GPU throughput

Updated at epoch 15: [kernel and memory tuning](church-kernel-throughput-2026-09-26.md)
raises measured throughput to 2,092.53 images/sec with 409–410 images per GPU and
no accumulation. Global batch 2,048 and 62 updates per epoch remain unchanged.
The rest of this document records the earlier effective-batch resize.

The user requested increased batch size after observing the batch256 run's low
throughput. The first five epochs are preserved at update 2,470 with official
FID50k 28.2037, including full model, AdamW, scheduler and all five rank RNG states.

The selected continuation uses **effective batch 2,048**, split across five H100
NVL GPUs with a maximum microbatch of 205 and two accumulation steps. Exact
per-image DDP weighting preserves the global count, including the final partial
batch of each epoch. There are now **62 updates per epoch**, matching the named
source run; this intentionally replaces the earlier batch256/494-update choice.

The complete-update benchmarks used fresh, isolated model weights:

| Effective batch | Maximum microbatch/GPU | Accumulation | Images/sec |
| --- | ---: | ---: | ---: |
| 256, prior production | 52 | 1 | ~510 |
| 512 | 103 | 1 | 905.26 |
| 1,024 | 205 | 1 | 1,376.27 |
| 1,280 | 256 | 1 | 1,523.50 |
| 1,536 | 308 | 1 | 1,620.68 |
| **2,048** | **205** | **2** | **1,655.76** |
| 2,048 | 256, then remainder | 2 | 1,624.25 |
| 2,048 | 308, then remainder | 2 | 1,564.62 |

Every candidate completed a full checkpoint write. The larger physical
microbatches were slower at batch2,048. The selected configuration improves
steady-state training throughput by approximately 3.2 times and reduces expected
training time per epoch from about249 seconds to76 seconds. FID and checkpoint
work are additional; the unchanged FID50k evaluation took about134 seconds.

The continuation restores epoch5 weights rather than benchmark weights. All582
Adam parameter states and all five RNG records are retained. Learning rate stays
at its exact boundary value, `0.000499657383688644`, with cosine decay tied to
epoch progress and reaching zero at epoch300. There is no optimizer reset,
warmup restart or automatic LR increase. AdamW betas0.9/0.95, weight decay1e-4,
gradient clipping1, dropout0.1, fixed-code labels and the model are unchanged.

With 2,470 updates already completed and295 epochs remaining at62 updates each,
the planned final **true optimizer count is20,760**. The scheduler's logical
counter is310 at migration, with an explicitly recorded offset of2,160 to the
optimizer counter. This preserves the epoch-based cosine phase without
mislabeling past optimizer work. Larger batch changes gradient noise and update
frequency; this is a mixed-batch continuation rather than a fresh batch2,048 run.

Six tests check state preservation, continuous LR and correct cosine phase for
all five candidate batches, and rejection of mid-epoch changes. Serialized model
and optimizer tensors are compared exactly against the preserved source state.
A separate check verifies immutable hardlink staging for asynchronous checkpoint
uploads; this avoids copying the full checkpoint again before each upload.

Production launched under PID 11458 and sustained approximately1,650 images/sec.
The first continued epoch completed at epoch6/update2,532. Its full checkpoint
passed finiteness checks, contained all582 Adam states at update2,532 and all five
rank RNG records, and had scheduler position372 plus the recorded2,160 offset.
Its LR `0.0004995066821070681` matches the epoch6 cosine value. Model and optimizer
state were continued from epoch5; none of the benchmark updates entered production.
Checkpoint uploads run asynchronously; `production-accepted.json` records the
remote digest verification once it completes.

Runtime: `/mnt/laser-church/batch-throughput-20260926`.
W&B: [larger-batch continuation](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fixed-codes-joint-sequence-b2048-h100x5-resize-20260926).
The original batch256 checkpoint is retained under `previous-b256/`.
Resume command after stopping the active process:

```bash
python /mnt/laser-church/batch-throughput-20260926/launch_continuation.py --resume
```
