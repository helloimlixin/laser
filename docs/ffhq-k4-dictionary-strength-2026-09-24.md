# K4 dictionary-strength pilot

User requested testing dictionary update strength. Controlled comparison against completed LR1e-5/relaxation0.25 pilot: reduce only dictionary_update_relaxation to 0.05. Start from the identical completed epoch132 source checkpoint (rFID6.131557583668723), train through142. Model/discriminator LR constant1e-5 with saved moments/counters and RNG restored. Source is the historical4e-5 checkpoint, so migration whitelist allows LR rebase plus relaxation; relative to the completed pilot only relaxation changes.

Unchanged batch44 per GPU/global176, four H100s, BF16 with autocast cache disabled, FP32 OMP/Adam, online alternating-residual updates, minusage2, six backtracks, all2048atoms. Dictionary Adam remains inactive. Relaxation0.05 is the initial update relaxation; backtracking may lower accepted relaxation further.

Same full10000-image bilinear native rFID and NVIDIA FLIP evaluation each epoch, fixed coefficient/FLIP heatmaps and source-selected zoom regions, latest plus best-three checkpoint uploads. Seed best retention with source epoch132.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-lr1e5-dict005-e132-142-20260924
Control: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-lr1e5-e132-142-20260924

Compare epoch133–142 mean, best, final rFID and average absolute adjacent-epoch change. Control values: mean6.9262792, best/final6.5189659, adjacent change0.1950480. K2 target5.943687. A single trajectory comparison does not establish multi-seed reliability.

Runtime /tmp/laser-ffhq-k4-dict005-20260924. Detached bounded-retry supervisor; CPU zoom watcher. Status and results mirrored alongside scripts in outputs/ffhq-k4-dict005-20260924.
