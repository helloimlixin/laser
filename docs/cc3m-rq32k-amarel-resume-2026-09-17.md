# CC3M RQ32K continuation on Amarel

Target: [cc3m-imagenet421-rq32k-650m-20260915](https://wandb.ai/helloimlixin-rutgers/laser/runs/cc3m-imagenet421-rq32k-650m-20260915).

Run directory: `/scratch/xl598/runs/laser/cc3m-imagenet421-rq32k-650m-20260915-amarel`.

The requested continuation is currently blocked on recovery of the original full checkpoint, source snapshot, and fitted cache/codebook. No CC3M GPU allocation has been submitted. W&B reports training paused at optimizer step 14640, epoch 10.3185; its run state is finished, but the requested target remains 100 epochs. Logged progress is not a verified checkpoint.

The exact run's 55 W&B files total 66.6 MB and contain logs, media, metadata, and small entrypoint scripts. No checkpoint is present. The project artifact inventory had no matching checkpoint or source bundle on September 17. The recorded Git commit, also current origin/main, lacks the untracked `src/cc3m_rq_training.py` implementation. Shared Amarel searches did not find the target checkpoint or source.

The original machine was recorded as `node-0`; its run directory was `/workspace/Projects/laser/outputs/cc3m-imagenet421-rq32k-20260915`. Recovery needs `train/last.pt` and its receipt, `source-snapshot-prompt-grid`, the corresponding source manifest and production config, and the fitted compact codebook/cache. The latest receipt must establish the real resume step. A second checkpoint mirror was under `/tmp/laser-cc3m-imagenet421-cache/train-state/last.pt` on that machine. A reachable SSH host, shared bundle path, or new W&B artifact will unblock preparation.

The local pixparse CC3M dataset and exact ImageNet rFID 4.21 tokenizer are available. The architecture is a 675,050,753-parameter text-conditional RQTransformer with 32,769 image tokens and 16,384 text tokens, 8×8×4 image shape, soft-target temperature 0.125, global batch 2048, and the original 100-epoch cosine schedule. Its two cached DALL-E crops and 100 cached BPE dropout text views must be retained. Original local batch 128 used 48.38 GiB on A100 80GB; local batch 32 with accumulation preserves global batch on Amarel A100/L40S.

A separate detached Codex supervisor is running on `amarel3`, tmux server `laser-cc3m-monitor`, session `cc3m-monitor`. Its instructions authorize recovering inputs, preparing and validating the launcher, selecting 8 or 16 A100/L40S GPUs using real Slurm probes, submitting one allocation, and fixing/relaunching failures. It checks supplied input locations and Slurm every minute, with routine Codex passes every 30 minutes while waiting and every 5 minutes during training. The existing ImageNet monitor and other runs are independent.

Live evidence and operating instructions: `RUN_DIR/monitor/README.md`, `heartbeat.json`, `snapshot.json`, `latest-result.json`, `status.md`, and `logs/`. Set a reachable location in `RUN_DIR/recovery-inputs.json` or place a resume bundle in `RUN_DIR/incoming/`. Creating `RUN_DIR/monitor/STOP` stops monitoring without cancelling training. The supervisor must be restarted if the login host or tmux process stops.
