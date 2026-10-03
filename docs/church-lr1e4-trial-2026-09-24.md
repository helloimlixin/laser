# Church learning-rate reduction trial, September 24, 2026

The batch-2048 Church continuation reached FID 10.749856 at epoch 30, then
14.038440 at epoch 35 while its learning rate remained close to 0.0005.
Gradients and losses remained finite. This motivates a lower-learning-rate
experiment; it does not establish the learning rate as the cause of the FID
variation.

The user approved a trial from the epoch-30 best with base learning rate 0.0001,
retaining optimizer state, global batch 2048, the four-H100 layout, and sampler.
The trial covers ten additional epochs, ending at epoch 40. It evaluates 50,000
images after five and ten additional epochs, at global steps 6760 and 7065.

## Learning-rate handoff

The source checkpoint is epoch 30, global step 6455, of
`church-compound-rqopt-b2048-4h100-20260924`. Its saved cosine position is 1525
of 18,300 optimizer updates, with base learning rate 0.0005 and actual learning
rate 0.0004914814565722676.

Scale optimizer learning-rate fields and scheduler base/last learning rates by
0.2, retaining the scheduler's position, horizon, and step counter. The resulting
base learning rate is 0.0001 and actual resumed rate is
0.00009829629131445352. The cosine schedule is not restarted. Model weights,
AdamW moments and step counters, and all four RNG streams are retained.

The trainer and runtime are unchanged. Merely editing the YAML learning rate
would leave the checkpointed scheduler's old learning-rate scale in effect;
the checkpoint's scheduler and optimizer learning-rate metadata must agree.
The transformation is recorded in `audit/prepare.py` and `handoff.json`.

## Preserved settings

- 128 images per GPU, four GPUs, four accumulation steps: effective batch 2048.
- BF16, compiled pair attention, fused AdamW, and the in-memory prebuilt cache.
- Validated Church tokenizer and original-image token cache; original RQ FID
  evaluator and reference statistics.
- Full atom/coefficient autoregression and dictionary atom-vector conditioning.
- Physical coefficient targets at temperature 0.125, atom loss weight 1.5,
  and geometry loss disabled.
- Atom top-k 250, coefficient top-p 0.85, and existing preview/evaluation cadence.
- Full optimizer-capable last and best-FID checkpoint uploads to online W&B.

The previous run and its checkpoints remain preserved. The new run owns
independent best and last payloads, with best-checkpoint paths rebased to the
new directory so checkpoint rotation cannot remove the parent run's best.

## Files and run

Configuration: `configs/stage2/lsun-church-lr1e4-trial-4h100.yaml`.
Run directory: `outputs/church-lr1e4-trial-20260924`.
Recovery: `outputs/church-lr1e4-trial-20260924/resume.py`.

W&B: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-compound-lr1e4-b2048-4h100-20260924

Preflight and live verification receipts are stored in the run directory.
The source FID 10.749856 is the retained baseline, not a new trial result.
