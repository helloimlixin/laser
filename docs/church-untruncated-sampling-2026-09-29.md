# LSUN Church: matched untruncated sampling evaluation, 2026-09-29

Full-vocabulary sampling at temperature 1 worsens FID by 1.675952 on the frozen best checkpoint at update 10540. Each arm generated 50,000 images with the same eight logical RNG streams, batch sequence, tokenizer, official Church reference, continuous FP32 decoded pixels and FP32 Inception features. Four physical GPUs ran each arm concurrently.

| Sampler | FID 50k | Mean term | Covariance term | Generated covariance trace |
| --- | ---: | ---: | ---: | ---: |
| Historical | 9.493208036 | 3.044492006 | 6.448716030 | 86.835919455 |
| Untruncated T1 | 11.169160294 | 4.190478802 | 6.978681493 | 84.542503730 |

Historical sampling: atom top-k 700, top-p 1, temperature 1; coefficient top-k 2048, top-p 0.85, temperature 0.9. Untruncated sampling: atom top-k 16384 and coefficient top-k 2048, both temperatures 1, both nucleus filters disabled with None. Existing same-site atom uniqueness remains. Atom selection and conditional coefficient selection follow the original model factorization. There is no refitting, reward, retraining or checkpoint mutation.

Checkpoint SHA256: `8e9a209aec912909b33ec8fd92c18c42d41f09a5d57e431b05a39d3285d9609b`. Historical replay differs from the recorded 9.49320803649016 by 0. Real covariance trace: 102.022432621.

The independent replay audit found identical SHA256 hashes for all eight historical Inception feature arrays, bitwise-equal mean/covariance statistics, and exactly equal FID components.

The comparison changes atom truncation, coefficient nucleus filtering and coefficient temperature together. It tests that complete sampling setting; it does not isolate which change explains the result or establish a training-objective root cause. These are measurements from one fixed checkpoint and seed protocol.

Samples, metrics, source snapshots, statistics and verified artifacts are on W&B: [historical](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-untruncated-best10540-20260929-historical) and [untruncated](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-untruncated-best10540-20260929-untruncated). Complete feature arrays and receipts are preserved under `outputs/church-untruncated-20260929/arms/`.

Training finished all 300 epochs / update 18600 before this evaluation. Final FID was 9.648757077031945, best FID 9.49320803649016 at 10540. Full LAST and BEST checkpoints, including optimizer/scheduler/RNG state, are remotely verified in `helloimlixin-rutgers/laser/church-best114-freshpairs-20260929-fresh-selected-checkpoints:v96` and durably saved under `outputs/church-best114-freshpairs-20260929/arms/fresh/`.

The first evaluation launch failed before checkpoint loading/sampling at the initial NCCL barrier. The retry restored the existing training launcher's NCCL_NVLS_ENABLE=0 and CPU thread settings. Failed logs/source are retained, and the successful launch kept checkpoint, native samplers and evaluator unchanged.
