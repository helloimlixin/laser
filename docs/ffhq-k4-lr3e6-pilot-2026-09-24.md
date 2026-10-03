# K4 LR3e-6 pilot

User-directed lower-LR test after weaker dictionary updates failed to improve rFID. Branch from the identical epoch132 checkpoint, source rFID6.131557583668723, train10epochs through142. Model and discriminator LR constant3e-6; saved Adam moments/counters and RNG retained, no warmup restart. Dictionary relaxation0.25, identical to completed1e-5 control. Dictionary Adam inactive. Batch44/GPU, global176, 4H100, BF16 codec/FP32 OMP and Adam, same data and evaluation.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-lr3e6-e132-142-20260924
Control: https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhq-a2048-k4-lr1e5-e132-142-20260924

Compare epochs133–142 mean, best and final rFID and mean absolute adjacent change. Control mean6.9262792, best/final6.5189659, adjacent change0.195048. K2 target5.943687. Scheduler simulation verified3410updates at3e-6 for both optimizers.

Weaker-dictionary run stopped after epoch139 evaluation/checkpoint/upload (rFID6.7892186); source/best and epoch139 zooms preserved. Four-GPU 16-update smoke test passed: dictionary synchronized, dictionary Adam inactive, checkpoint roundtrip verified, relaxation0.25. Full10k native rFID, NVIDIA FLIP, fixed coefficient/FLIP maps and zoom gallery retained; best-three plus latest checkpoint artifacts, bounded-retry supervisor. No claim of improvement before evaluation.
