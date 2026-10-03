# ImageNet joint continuation on eight H200s

Continue `helloimlixin-rutgers/laser/imagenet-rfid421-joint-scratch-20260926`
from its complete optimizer checkpoint at step 26,312 (epoch 41.436), using
`imagenet-rfid421-joint-best-8h200-20260926`. The parent reached FID 30.29766
at epoch 40 and improved at every five-epoch evaluation. Its epoch-five FID
was 51.06507; the original orthogonal run's was 50.55738. Those early results
do not establish an architectural winner. This continuation prioritizes
reusing completed training to improve quality per additional GPU-hour.

The 870 model tensors, 870 Adam states, and exact warmup/cosine scheduler
restore successfully, and every model and optimizer tensor is finite. The
initial learning rate is 0.00032385736114016116, continuing the parent's
schedule rather than resetting it. Full provenance is in the output folder.

## Changes

- Four GPUs, microbatch 252 and accumulation two become eight GPUs,
  microbatch 252 and accumulation one. Effective batch stays 2,016, with
  635 optimizer updates per epoch and 63,500 across the full 100 epochs.
  The checkpoint's within-epoch cursor maps from 554 to 277 microbatches.
- Two fixed image views become fresh `Resize(256)`, `RandomCrop(256)` and
  horizontal flip with probability 0.5. Views are reproducible per image and
  epoch, independent of data loader prefetch and recovery.
- Online dictionary encoding uses FP32 encoder/OMP precision and TF32
  convolutions as in the source cache. Encoder chunks bound memory, while
  a shared dictionary Gram matrix avoids repeated construction.
- Previews run every 200 optimizer updates; FID uses 50,000 generated images
  every five epochs. The original RQ-VAE Inception implementation, ImageNet
  training reference statistics, and parent's sampler remain unchanged:
  atom temperature 1, top-k 2,048, top-p 1; coefficient temperature 1,
  all 2,048 bins, top-p 0.85. No rejection sampling.

The dictionary-coordinate joint architecture, coefficient scales, physical
soft target temperature 0.03125, loss weights, AdamW settings, clipping,
and 100-epoch schedule remain those of the parent. The fresh orthogonal
run is saved separately before releasing its GPUs. No parent checkpoint
is modified or pruned. The new run owns its recovery and best checkpoints.

This is a continuation with changed augmentation, not a fresh paper
reproduction. The paper's batch 2,048 would give 625 complete updates per
epoch for this dataset. Retaining 2,016 preserves the trained parent's
optimizer trajectory. Doubling GPUs is intended to increase update
throughput; it does not imply twice the FID improvement. New ranks receive
independent deterministic RNG streams; world-size and augmentation changes
make this a statistically comparable continuation, not a bitwise replay.

## Validation and acceptance

Forty targeted tests passed for fresh augmentation, dictionary/orthogonal
encoding, coefficient history, optimizer schedules, and resume behavior.
Each rank audits its initial optimizer counter and learning rate, then
checks all weights and Adam moments after 20 retained training updates.
The first recovery checkpoint is checked independently after durable copy.

The first scheduled quality comparison is epoch 45, step 28,575: 2,263 new
updates after the source checkpoint. A win against the joint baseline
requires measured FID below 30.29766 under the unchanged evaluation protocol.
No such win is claimed before evaluation.

## Production confirmation

The run resumed at step 26,312, epoch 41, batch cursor 277. All eight ranks
passed the live 20-update audit with all 870 Adam counters advanced correctly
and finite weights/moments. At step 26,500 the global training loss was
6.66358 and median steady throughput was 2,035.6 fresh images/sec, or 1.0097
optimizer updates/sec. Loss on new crops is not directly comparable with the
parent's rank-zero loss on its two cached views.

The source checkpoint SHA256 is
`9af9215bad82aed249ac16ec7c052f9f5d9f5ef610c796110fe0ef2914eeac45`.
The fresh orthogonal run was preserved at step 1,781 with its complete Adam
state and eight RNG streams. GPU processes were released after durable save;
waiting for a redundant W&B checkpoint upload was not allowed to hold the GPUs.

The first continuation recovery checkpoint is step 26,400. Its independent
verification result is written to `checkpoint-verification.json` in the run
output directory. Startup and step-26,400 previews are saved under `samples/`.
New 50,000-image FID remains pending until epoch 45.
