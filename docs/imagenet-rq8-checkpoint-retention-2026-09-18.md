# ImageNet RQ8 online checkpoint retention

The run `helloimlixin-rutgers/laser/imagenet-rfid421-rq8-refit-480m-20260913`
keeps the latest full checkpoint and the three distinct checkpoints with the
lowest **original_t09 FID-4000**, its regular selected-sampler evaluation.
FID-4000 and FID-50000 are not mixed in this ranking. Existing single winners
for the other sampler/evaluation combinations are retained as well.

The ranking was backfilled from committed checkpoints in the resumed training
sequence:

| Rank | Optimizer step | Epoch | FID-4000 | Initial source artifact |
| --- | ---: | ---: | ---: | --- |
| 1 | 15,024 | 24 | 46.2051458702 | checkpoints:v79 |
| 2 | 13,772 | 22 | 47.7306127633 | checkpoints:v67 |
| 3 | 12,520 | 20 | 49.4443251012 | checkpoints:v53 |

The full artifact collection name is
`imagenet-rfid421-rq8-refit-480m-20260913-checkpoints`. Its `latest`, `best-1`,
`best-2`, and `best-3` aliases are online in W&B. Initially the ranked aliases
pin their historical source artifacts, where the ranked file is
`best-fid-original_t09-4000.pt`. Subsequent bundles contain all three ranked
files plus `last.pt`; `best-fid-top3.json` identifies each file, FID, epoch,
step, and rank. Each checkpoint includes model, optimizer, scheduler, scaler,
RNG state, configuration, and resume cursor.

The new tracker considers every selected-metric evaluation, including results
that place second or third without setting a new overall record. Duplicate
steps do not fill multiple slots. Scores must be finite, metrics must match,
and ties prefer the earlier step. Best files are immutable hard links with
step-specific names. Retiring a local best file cannot remove bytes held by
an upload snapshot. The ranking is stored both in full training checkpoints
and an atomic JSON sidecar, so it survives restart.

Uploads remain asynchronous with durable snapshots, retries, and coalescing
to the latest saved state when upload bandwidth falls behind training.
The latest checkpoint is saved every 100 optimizer updates, at epoch
boundaries, and on graceful shutdown. Uploads are also requested after FID
evaluation and immediately on startup.

The isolated runtime, historical artifact inventory, digest checks, and online
alias verification are in
`outputs/imagenet-rfid421-rq8-refit-20260913/best3-20260918/`.
The training kernels, 162 GiB RAM cache, six-H200 batch partition, and optimizer
settings are unchanged. Six retention/upload tests and the seven existing
resume/data tests passed. The launcher remains:

The policy was activated with a graceful restart at step 16,663. Production
restored all six rank RNG states and advanced to step 16,666 at approximately
1,009 images/sec without additional AMP skips. The new immutable upload job
was checked to contain the latest full resume state, all three ranked
checkpoints, and the ranking manifest. Historical `best-1`/`best-2`/`best-3`
aliases were independently verified as committed online.

```bash
python scripts/tools/launch_imagenet_rq8_h200.py
```
