# FFHQ K4 RQ recipe: resume on two H200 GPUs, 2026-09-25
Subsequent [throughput tuning](ffhq-k4-rqrecipe150-throughput-2026-09-25.md) selected batch 64/GPU, accumulation 1, and disabled activation checkpointing. The details below describe the initial recovery.


The existing [W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-rqrecipe150-20260924) resumes its 150-epoch training budget from completed epoch 36, optimizer step 16,884. The recovered artifact is `ffhq-a2048-k4-rqrecipe150-20260924-stage1-checkpoints:v35`, checkpoint ID `d786a3cc911c4a9f90a52a7f81b62514`, SHA256 `01514ff9f8e7453e40c60b2fa3b98e91287eaadab3ada3450018dece36dab5c1`.

Model, dictionary, discriminator, Adam moments/counters, both scheduler states, and best-three checkpoint rankings are restored. Production starts from the untouched downloaded checkpoint. Disposable verification weights are separate. W&B uses `resume=must` with the original run ID.

The original four H100s are replaced by two H200s. Each GPU still processes 32 images per microbatch; accumulation increases from one to two, retaining global batch 128 and discriminator BatchNorm scope 32. Training remains FP32 with TF32 disabled, activation checkpointing, gradient dictionary updates, and LR 4e-5. Scheduler and logging clocks remain measured in actual optimizer updates. Rank 0/1 RNG states are restored at the epoch boundary. Changed topology, sample ordering, and accumulation mean this is not a bitwise continuation.

Every frozen runtime source hash matched the original checkpoint before adaptation. Only the launch driver and accumulation scheduler/logging clocks changed. The exact diff and source hashes are in the [resume receipt directory](../outputs/ffhq-k4-rqrecipe150-resume-20260925/).

The full official 60,000-training/10,000-validation split was recovered from the lossless FFHQ mirror pinned to revision `d74f1f1f59e3bbe975bee29872b9bef827314577`. All 70,000 decoded RGB images match official pixel MD5 records. After original BILINEAR 1024-to-256 preprocessing, all 60,000 training PNG SHA256 hashes match the original prepared-data manifest. Training augmentation and held-out validation are unchanged.

A two-GPU restored-state test passed eight generator, discriminator, dictionary Adam, and scheduler updates. It checked finite parameters/losses, dictionary synchronization, unchanged LR, and model/discriminator checkpoint roundtrips. Native rFID, NVIDIA FLIP, and preview generation were exercised on 128 validation images; this smoke score is not a full-dataset quality result.

Runtime: `/tmp/laser-ffhq-k4-rqrecipe150-20260924`.

- Supervisor: `resume_supervisor.py`; bounded to four attempts, with a process lock.
- Launch/PIDs: `resume-launch.json`; live status: `resume-supervisor-status.json`.
- Training logs: `resume-training-attempt*.log`; current epoch: `training-status.json`.
- Checkpoints: `train/last_model.pt` plus inherited best-three files. Recovery saves every 200 optimizer updates; versioned W&B uploads each epoch.
- Existing zoom gallery resumed; its watcher status is `zoom/watcher-status.json`.
- Full 10,000-image native rFID/FLIP, coefficient maps, and reconstruction previews continue each epoch.

The original and adapted runtime archives, configuration, migration diff, launch receipt, and verification reports are preserved under `outputs/ffhq-k4-rqrecipe150-resume-20260925`. The adapted archive and reports are also saved to the original W&B run. Credentials are inherited through process environments and are excluded from scripts and archives.
