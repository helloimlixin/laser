The additional speed tests retained the existing LSUN Church implementation. Neither candidate justified deployment. The active run remains [church-vector-lrhalf-20260928](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-vector-lrhalf-20260928), with the same objective, global batch 256, precision, model, teacher and half-LR schedule.

Reusing depth-transformer computations for identical candidate prefixes passed the strict FP32 gradient comparison (relative L2 difference 2.51e-6), but exceeded the fixed production TF32 tolerance: 5.24e-4 versus 5e-4. A shared-GPU benchmark also showed no speed improvement. This candidate was rejected without deployment; shared-device timings are not production throughput estimates.

Asynchronous gradient communication preserved the 64 bucket boundaries, collective order and FP32 averaging. All elements of all 510 gradient tensors passed the eight-rank correctness check. The exclusive benchmark measured 15.67 ms for the existing implementation versus 13.74 ms for the candidate. The 1.93 ms saving was below the preselected 5 ms requirement, so this candidate was also rejected. It would account for less than 0.5% of an ordinary training update even if the entire isolated saving carried over.

The exclusive test took 21.7 seconds after the completed epoch-30 checkpoint. Training resumed from step 14,670 with all model, optimizer, scheduler, cursor and eight RNG states retained. All eight startup records passed strict source validation; no candidate source or migration ledger was installed. The canonical training log follows `speed3-baseline.log`.

The uncontended window before the test measured 0.4770 seconds/update, approximately 537 images/second. Samples every 200 updates and full last/best checkpoint publication continue. Epoch-30 FID, computed before the speed test, was 10.16806 (mean 3.53904, covariance 6.62902); the best remains 9.91859 at step 9,780. These metrics cannot be attributed to a speed change, since none was applied.

Benchmark, checkpoint, continuation and source-validation receipts are stored in [the speed audit outputs](/workspace/Projects/laser/outputs/church-vector-lrhalf-20260928/speed3/review.json).
