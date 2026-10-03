The fixed-checkpoint sampling sweep found no supported improvement over joint-token top-k 1,400. Keep the existing [Church training run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-joint65k-20260929), its original sampler, and its 300-epoch target. The alternative continuation worker was prepared and checked but was never launched.

All four measurements use the same step 2,480 checkpoint (40 epochs), SHA-256 `5914d1f22fb429a6759b8b51a473def368ffb6cbc66c7e6859006a3a90d260ec`, with 65,537 complete atom–coefficient tokens and temperature 1.0. No training updates, refitting, model changes, or objective changes occurred during the comparison.

| Sampling policy | FID50k | Mean term | Covariance term | FID change |
|---|---:|---:|---:|---:|
| Top-k 1,400 (original) | 10.584468 | 3.419906 | 7.164562 | +0.000000 |
| Top-p 0.95, no top-k | 10.904869 | 3.802691 | 7.102178 | +0.320401 |
| Top-p 0.98, no top-k | 11.262406 | 4.090193 | 7.172214 | +0.677939 |
| Full vocabulary | 11.775774 | 4.431300 | 7.344474 | +1.191306 |

Top-p 0.95 reduced covariance by 0.062384, but increased the mean term by 0.382785 and worsened total FID by 0.320401. Top-p 0.98 and full-vocabulary sampling worsened both covariance and total FID. Inference truncation alone did not resolve covariance error at this checkpoint. This comparison does not identify the underlying training failure or establish which sampler would win at a later checkpoint.

The evaluation used exactly 50,000 generated images per policy, eight fixed logical RNG streams starting at seed 2026093901, generation batches of 1,024, and decoder/Inception batches of 32. The frozen decoder and official Church reference were identical; generation used BF16 cached logits with FP32 probabilities, while decoder/features used FP32, continuous pixels, and TF32 disabled. The original top-k result was reused after checkpoint/protocol verification and verification of all eight saved feature hashes. Old and new top-k implementations produced identical sampled codes in the eight-rank parity probe. No feature means or covariances were modified.

A separate free-running baseline probe at the same checkpoint measured the probability mass retained by top-k:

| Depth | Mean retained probability mass |
|---|---:|
| 1 | 97.39% |
| 2 | 87.60% |
| 3 | 71.64% |
| 4 | 55.92% |

These are 1,024 token observations per depth from the first two sampled rows at every spatial location across eight ranks; the probe generated 1,024 images and is not another FID evaluation. BF16 ties at the top-k boundary retain slightly more than 1,400 IDs. The earlier CPU teacher-context probe at step 1,860 is archived separately and is not this measurement.

The prespecified acceptance rule required lower total FID with no increase in covariance. None of the candidates passed. The original frozen worker resumed from its full step-2,480 checkpoint and was verified running at step 2,530 on all eight GPUs, with all 460 Adam states restored and the scheduler resumed at step 2,480. All 35 frozen source files matched their original hashes. Optimizer, schedule, data position, and all RNG streams were preserved; the durable `resume-verified.json` records the restart (controller PID 249473, torchrun PID 249540). Its sampling remains whole-token top-k 1,400, temperature 1.0, no top-p; samples remain every 200 updates. Parent LAST/BEST checkpoints remain associated with the original run and policy.

Results, sample grids, and verified artifacts are available in the [W&B sampling sweep](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-joint-sampling-sweep-20260929). Local evidence is archived in `outputs/church-joint-sampling-20260929`; `prepared-continuation` contains unlaunched code, not an active training run. Its CPU restoration preflight verified every model and Adam tensor and the scheduler against step 2,480 without initializing CUDA or performing an optimizer update.
