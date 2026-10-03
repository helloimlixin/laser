# K4 LR1e-7 with gradient-trained dictionary

User requested1e-7, no dictionary relaxation, and explicitly not freezing dictionary. Switch dictionary_update_mode from alternating_residual to gradient. Model, discriminator and dictionary Adam LR all constant1e-7. No relaxed residual updates; dictionary still receives gradients, gradient projection and unit-norm normalization. Existing dead-atom revival remains enabled. Model/discriminator Adam moments and counters retained; dictionary moments start fresh because source has no dictionary Adam state, and are retained on subsequent resumes.

Start from original epoch132 checkpoint rFID6.131557583668723 and train10epochs through142. Same4H100, batch44/global176, BF16 codec with autocast cache disabled, FP32 OMP/parameters/Adam, full10000-image native rFID and NVIDIA FLIP, fixed heatmaps and zoom crops, best-three plus latest uploads. This changes both LR and dictionary update algorithm relative to earlier pilots; results cannot isolate the LR effect alone.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-lr1e7-e132-142-20260924

Previous3e-6 pilot stopped after epoch138 checkpoint/upload (rFID7.2256738473), zooms preserved through138. Four-GPU smoke test passed: dictionary changed through16Adam updates with nonzero moments, synchronized unit-norm atoms, zero additional alternating updates, and checkpoint roundtrip including dictionary moments. Both LR schedules simulated for3410updates at1e-7. Runtime /tmp/laser-ffhq-k4-lr1e7-20260924.
