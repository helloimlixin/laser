# FFHQ H200 precision and validation optimization, 2026-09-26
Subsequent [commitment and quality tuning](ffhq-k4-rqrecipe150-quality-2026-09-26.md) corrected the effective commitment weight to 0.25, selected model/dictionary LR 2e-5 and batch 32/GPU with accumulation 2, and added full-set LPIPS/MSE validation. The configuration below describes the preceding continuation.


The same [W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-rqrecipe150-20260924) now permits TF32 **convolutions during training**. Matrix multiplication TF32 stays disabled. Parameters, Adam states, sparse-code OMP solves, dictionary updates, and validation remain FP32. AMP/BF16 is not enabled in production.

The selected configuration measured **153.29 training images/s**, versus **66.55** for the previous FP32 configuration: **2.30×**. This is training-loop throughput, excluding validation/checkpoint uploads. The longer test used four warm-up updates and 100 timed updates with actual images and complete generator/discriminator/dictionary updates.

| Training mode | Images/s | Peak allocated GB/GPU |
|---|---:|---:|
| fp32 | 66.55 | 127.46 |
| tf32 | 153.11 | 127.46 |
| bf16 | 192.13 | 106.52 |
| bf16_cl | 152.68 | 106.52 |

The initial comparison used four warm-up and 20 timed updates per candidate, always loading the same step-33,600 checkpoint. Both BF16 and TF32 also completed 104-update finite-loss/parameter, optimizer/scheduler-counter and exact cross-rank dictionary checks. BF16 was faster, but its short-run held-out FLIP/MSE drift was larger, so TF32 was selected. These short tests do not prove identical long-term reconstruction quality.

Held-out comparison: 512 fixed, evenly spaced official validation images, using FP32 single-image reconstruction for every candidate. These subset FIDs must not be compared directly to the full 10,000-image FIDs shown in the main run.

| Mode after 24 updates | Subset rFID | FLIP | MSE |
|---|---:|---:|---:|
| source | 40.32015 | 0.215659 | 0.018600 |
| fp32 | 40.44449 | 0.208347 | 0.017239 |
| tf32 | 40.26755 | 0.217438 | 0.018516 |
| bf16 | 40.75680 | 0.233897 | 0.022127 |
| bf16_cl | 40.72944 | 0.216335 | 0.018629 |

Validation continues on **all 10,000 images every epoch**, using the same original RQ-VAE Inception/rFID implementation and NVIDIA FLIP. Reconstruction batches now contain 32 images per GPU. On the 512-image paired check, all 131,072 sparse-code entries matched single-image inference, mean absolute reconstruction difference was 4.50e-7, and maximum absolute difference was 3.24e-5 in normalized model output units. This is numerical agreement, not bitwise equality. Reconstruction time in that check fell from 6.71 to 3.95 seconds.

The fixed real-image Inception mean/covariance is computed once, then cached in FP64. An independent second FID accumulation using only reconstructed-image features and the cached reference reproduced the uncached subset FID exactly. Full-run cache files include the reference count, protocol and first-creation signature. Cache reuse does not skip reconstructed-image evaluation.

Training keeps batch 64/GPU, global batch 128, accumulation 1, discriminator BatchNorm groups of 32, the existing architecture, both Adam states, scheduler positions, LR 4e-5, and the 150-epoch target. Console summaries are refreshed every 10 batches; all metrics are still accumulated and W&B step logs retain their cadence.

Resume source: epoch index 71, batch cursor 301, optimizer step 33,600, checkpoint ID `c92e483e92944724bdf3d06141c18a88`. Training was stopped immediately after an atomic recovery checkpoint. Both restored optimizer counters equal 33,600 and each rank retains its 19,264 consumed-image cursor and saved RNG/partial metrics. Benchmark-trained weights were discarded. The source checkpoint is retained locally and uploaded as `precision-source-step33600.pt` in the original run's Files.

Runtime: `/tmp/laser-ffhq-k4-rqrecipe150-20260924`. Active supervisor: `precision_supervisor.py`; status: `precision-supervisor-status.json`; logs: `precision-training-attempt*.log`; restore audit: `precision-resume-audit.json`. Existing recovery checkpoints, versioned epoch artifacts, best-three retention, previews and CPU zoom watcher continue. The numerical change is recorded in checkpoint lineage.

Reproducible runtime, selected configuration, comparison/stability results, verification and launch receipts are stored alongside this report in `outputs/ffhq-k4-rqrecipe150-precision-20260926`. Credentials are excluded.

## First live verification

W&B step timestamps measured **153.19 images/s** between optimizer steps 33,700 and 33,750. Full epoch-72 validation completed on 10,000 images in **205.11 seconds** (previous epochs approximately 245 seconds), including the first real-statistics cache creation. Checkpoint `6720c4278513438db065d85200462352` saved at step **33,768** with both optimizer counters verified and the new precision signature.

Full validation rFID was **11.0241** and FLIP **0.27096**; the preceding epoch reported rFID **10.7014** and FLIP **0.22164**. FLIP therefore increased at the first post-change epoch. Previous validation FLIP fluctuated between approximately 0.211 and 0.264. One epoch cannot establish whether this increase is ordinary adversarial-training variation or a precision effect; equivalent long-term quality is not established. The faster configuration is running with full validation every epoch, making this observable.

Using the measured training rate, first validation time and previous checkpoint/preview overhead gives approximately **11 minutes per complete epoch**, compared with 20 minutes previously. At 72/150 epochs complete, about **14 hours** remain under that estimate. Later reuse of cached real-image FID statistics may reduce validation time further; that full-data cache-hit timing has not yet been measured.
