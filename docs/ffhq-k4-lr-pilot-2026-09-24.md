# FFHQ K4 learning-rate pilot — 2026-09-24

Running: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-lr1e5-e132-142-20260924

Branch from completed epoch 132 of the 100–150 continuation (rFID 6.131557583668723), train 10 epochs through 142. K2 target is 5.943687 at epoch 136. No claim of improvement yet.

Only active optimization change: model and discriminator LR 4e-5 → 1e-5, constant, with Adam moments, step counters, RNG, and scheduler counters preserved; no warmup restart. The dormant dictionary Adam group is also relabeled 1e-5 but has no gradients/state. Online alternating-residual dictionary updates retain relaxation 0.25, minimum usage 2, six backtracks, all 2048 atoms.

Unchanged: four H100 80GB GPUs, batch 44 per GPU/global 176, local discriminator BN batch 44, BF16 codec with autocast cache disabled, FP32 OMP/parameters/Adam, bicubic training and bilinear validation. Full 10,000-image native RQ-VAE rFID and NVIDIA FLIP every epoch; fixed coefficient and reconstruction FLIP maps; identical zoom crop regions in separate CPU gallery.

Verification: 16-update four-GPU smoke test passed (synchronized unit-norm dictionary, inactive dictionary Adam, checkpoint roundtrip); scheduler simulation verified constant model and discriminator LR for all 3410 planned updates. Production updates confirmed at 1e-5; all four GPUs sampled at 100% utilization and approximately 78,000 MiB memory each. Smoke-test rFID uses only 128 images and is not comparable to the full benchmark.

Best-three plus latest checkpoint policy retained, seeded with source epoch 132 best; epoch checkpoints upload to W&B. Detached supervisor permits four attempts and resumes recovery checkpoints. Compare epochs 133–142 against the previous 4e-5 branch using matched rFID mean, minimum and adjacent-epoch variation. Historical comparison is not an independent multi-seed experiment.

Zoom gallery: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-lr1e5-e132-142-20260924-zooms

Local runtime: /tmp/laser-ffhq-k4-lr1e5-20260924. Scripts/configuration and JSON status mirrored in outputs/ffhq-k4-lr1e5-20260924.
