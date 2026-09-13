# Persistent FID monitoring and learning-rate reductions

The user requested continued monitoring of the corrected Church run and lower learning rates if FID fluctuates again. The existing run `church-joint-geometry-20260912` now has a checkpointed FID controller; LR adjustments require no future relaunch. Both GPUs, the 256-image effective batch, microbatches of 32, model, tokenizer, relative-noise targets, geometry loss, and sampling thresholds are retained.

## Decision rule

Only the fixed 4,096-image screen series controls LR. The 50,000-image scores are never mixed into this series. A meaningful improvement is a reduction of at least 0.25 FID from the controller's best score. Two consecutive evaluations without that improvement halve the multiplier on the existing cosine schedule. Two subsequent evaluations are cooldown checks; a further two misses after cooldown can trigger another reduction. The effective LR is `max(1e-6, scheduled_lr * multiplier)`, so reductions persist and the existing decay continues.

The controller tracks sample count, seed, batch size, GPU count, sampling settings, and precision. A changed evaluation protocol establishes a new baseline rather than being compared with the old series. Invalid scores, mismatched checkpoint steps, and changed cached evaluations are rejected. Re-reading the same completed evaluation cannot trigger another reduction.

The policy was armed from the latest completed matching evaluation at epoch 15, FID4096 **22.60037**. Epoch 20 completed during installation and improved to **21.78087**, so the initial production multiplier remains **1.0**. The epoch-10 FID50k remains **22.13660**; it is a different sample-count series and does not control this scheduler.

## Persistence and observability

`src/church_fid_lr.py` owns the policy state. Rank zero makes each decision and broadcasts the complete state to the second rank. The trainer updates both optimizers, saves the controller with the model/AdamW/data-stream checkpoint, and records decisions under `lr_monitor/*` in the existing W&B run. Checkpoint state takes precedence over cached display files. A resume also restores the policy automatically from the saved execution metadata.

The installed policy is `outputs/church-joint-geometry-20260912/fid-monitor/policy.json`. Current controller state and the latest decision are in `joint/lr-monitor.json`; training logs include both `train/base_lr` and `train/lr_multiplier`. The original recipe config is retained, with execution and monitor settings recorded separately.

`scripts/tools/monitor_church_training.py` is a separate, read-only health process polling every 15 seconds. It reports missing/failed processes or a status heartbeat older than five minutes, and writes `fid-monitor/health.json` plus a change log. It follows the run's current PID and exits when training completes. LR decisions are part of the trainer and do not depend on the health process remaining alive. The health monitor does not send external messages or automatically restart a failed training process.

## Installation and verification

The original two-GPU process saved and paused at step **10,090**, epoch **20.46646**. Its pre-monitor checkpoint is preserved as a hard link under `fid-monitor/pre-monitor-last.pt`. Production resumes that checkpoint, retaining its optimizer moments and data stream. Verification weights are never transferred into production.

Six tests cover plateau patience, improvement/reset behavior, cooldown, the exact 0.25 threshold, the minimum LR, unchanged base decay before reductions, cached-evaluation idempotence, checkpoint resume, protocol changes, score validation, and synchronized decisions/optimizer rates across ranks.

The real 404,738,048-parameter checkpoint is also exercised on both GPUs with deliberately synthetic FID events confined to `/tmp/church-fid-monitor-verification`. Two missed improvements trigger a half-rate update, and a save/resume continues at that rate without repeating the reduction. Four actual optimizer updates verify preserved optimizer step counts and unchanged recipe configuration. The production checkpoint remains at its original saved step throughout this probe.

The authoritative installation evidence is under `outputs/church-joint-geometry-20260912/fid-monitor/`: `unit-tests.xml`, `full-model-verification.json`, `verification.json`, `pause.json`, `installation.json`, and `health.json`.
