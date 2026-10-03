# Church fixed-code lower-LR continuation

The high-LR run reached FID50k 10.4321812 at epoch 35, then fluctuated without improving through epoch 105 (latest FID 12.7726733). The user requested a fix and relaunch after discussing a lower learning rate. This experiment restores that best checkpoint and changes the learning rate while retaining the model, data, optimizer moments, sampling and evaluation protocol.

Run: `helloimlixin-rutgers/laser/church-fixed-codes-joint-sequence-lr1e4-from-e35-b2048-h100x5-20260926`.

Runtime: `/mnt/laser-church/lr-repair-20260926`. Reproducible launch: `python launch_continuation.py` in that directory; subsequent resumes require `--resume`. The launch validates the checkpoint migration, scheduler tests and five-GPU preflight before starting.

The resumed LR is **0.0001**, down from 0.0004833951066242992 at epoch 35. The existing cosine curve is uniformly multiplied by 0.2068701123152272; its phase and 300-epoch horizon remain unchanged. The corresponding cosine base LR is 0.00010343505615761361. Thus the W&B base-LR configuration is slightly above 0.0001, while the actual LR at the first resumed update is exactly 0.0001. There is no scheduler restart or optimizer reset.

The initial comparison ends at epoch 45, with unchanged FID50k evaluations at epochs 40 and 45. The shorter execution limit does not compress the cosine schedule. It adds 620 optimizer updates, taking the true update count from 4330 to 4950. The scheduler count goes from 2170 to 2790; the offset of 2160 preserves the earlier batch-256 training history.

| Setting | Value |
|---|---|
| Hardware | Five H100 NVL GPUs |
| Effective batch | 2048 |
| Physical batch per GPU | 409–410; one microbatch per update |
| Images / updates per epoch | 126227 / 62 |
| Optimizer | Fused AdamW; betas 0.9, 0.95; weight decay 0.0001; clipping 1 |
| Targets | Same fixed atom and coefficient cache; hard targets |
| Precision and kernels | BF16 training; FP32 gradients; compiled blocks; fused hard cross-entropy |
| FID | Official RQ implementation, 50000 generated images, same real statistics and seed |
| Sampling | Atom top-k 250, temperature 1; coefficient temperature 0.9, top-p 0.85 |

Historical control FIDs are 10.8883547 at epoch 40 and 12.2331788 at epoch 45. The inherited best is 10.4321812 at epoch 35. Quality improvement must be evaluated from the new measurements; a successful resume alone does not establish that the LR caused the earlier fluctuations.

The complete epoch-105 predecessor checkpoint and epoch-35 best checkpoint are preserved under `preserved-predecessor/`. The epoch-105 checkpoint was verified uploaded as parent selected-checkpoints artifact `v22` before terminating its training process. The source epoch-35 best is also available in the parent selected-checkpoints artifact `v21`, which the new run records as input lineage.

Validation records live beside the runtime: `lr-state-verification.json` compares every model, Adam and RNG tensor after serialization; `lr-resume-tests.json` checks 620 subsequent scheduler updates and a second resume; `lr-gpu-preflight.json` validates three isolated updates on all five GPUs. Inherited batch and kernel benchmark records document previously accepted execution settings. The isolated preflight's initial step-limit argument was corrected from an absolute to an additional-update count before passing; it never advanced the production checkpoint.

## Ten-epoch comparison

| Epoch | Original LR FID50k | Lower LR FID50k |
|---|---:|---:|
| 40 | 10.8883547 | 11.0285933 |
| 45 | 12.2331788 | 11.0942322 |

At epoch 45 the lower-LR result is 1.13895 FID points (9.31%) better than the original at the same epoch. It is slightly worse at epoch 40 and does not beat the inherited epoch-35 best of 10.4321812. The two lower-LR scores are close, but two evaluations cannot establish long-term stability or prove that excessive LR was the sole cause of the fluctuations.

Recent training throughput was 2078 images/sec, versus 2074 at matching original updates. Median gradient norm was 0.827 versus 2.385. Held-out atom NLL was 10.4472 versus 10.4911 at epoch 45; the train/validation gap remains. The continuation therefore retains the lower LR as a promising stabilization attempt, with the original 300-epoch cosine horizon and unchanged FID cadence. Epoch-45 model, optimizer, RNG and scheduler state are preserved for the continuation; the epoch-35 best remains separately protected.

The epoch-45 checkpoint was verified uploaded as selected-checkpoints artifact `v3`. The same W&B run then resumed at update 4950 and LR 0.00009779818301191861 with an execution limit of 300 epochs. `continued-resume-verified.json` confirms all five ranks restored 582 Adam states each, the scheduler retained logical step 2790, online configuration retained the 300-epoch horizon, and training advanced past update 4960. The completed trial and its checkpoints are retained under `preserved-trial/`.
