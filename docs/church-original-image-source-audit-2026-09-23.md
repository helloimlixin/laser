The active Church stage-2 cache contains sparse tokens extracted directly
from original LSUN Church training photographs. The cached reconstructions
shown in the watermark diagnostic are outputs of the frozen decoder for
inspection; they are not used as new images to build the training cache.

The source path is
`/workspace/Projects/data/lsun/lmdb/church_outdoor_train_lmdb`, containing
126,227 original images. The cache builder's LSUN branch reads RGB bytes from
that dataset, applies resize-to-256 and center-crop-to-256, and calls the
frozen encoder and OMP directly. It stores only `atoms`, `coeffs`, `labels`,
and `meta`. The stage-2 dataset returns atom IDs, FP32 coefficients, and labels;
the active training loop uses those tensors without decoding and re-encoding
images. Decoder calls in the cache builder are subsequent validation only.

The paths are:

```text
Original LSUN training RGB -> frozen encoder + OMP -> cached atom/coefficient
pairs -> autoregressive training

Cached atom/coefficient pairs -> frozen decoder -> diagnostic reconstruction

Generated atom/coefficient pairs -> frozen decoder -> Inception features
-> comparison with the fixed LSUN FID reference statistics
```

A fresh independent check loaded 24 original photographs from the persistent
LMDB: the eight images in the earlier reconstruction diagnostic and 16
additional images sampled with seed 20260923. It verified the sorted LMDB key
order, the 126,227-image population, and the active tokenizer checksum. All
6,144 atom IDs matched the existing cache exactly. Mean normalized coefficient
absolute error was 4.9637066e-7, maximum 5.9604645e-6, using FP32 encoding with
TF32 disabled. Small floating-point differences remain across execution and
batch sizes; this is a 24-image audit, not a full cache rebuild.

As a negative control, the eight cached reconstructions were encoded as if
they were source photographs. Only 27.3926% of their atom IDs matched, no
image matched all IDs, and coefficient mean absolute error was 0.8125409.
These reconstructions do not reproduce the training cache. The earlier
cache-build validation also independently encoded its first 256 source images
and reported exact atom and coefficient matches.

Cache SHA-256:
`4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586`.
Tokenizer SHA-256:
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.

The active FID reference is `/mnt/laser-church/assets/lsun_256_church.npz`,
SHA-256
`809489d8316b9e6eb9dc3bc021b6d602f4b6d816cc80621c6b9c189a9253a7f6`.
It supplies a 2,048-dimensional feature mean and 2,048×2,048 covariance.
When this file is configured, the evaluator loads those fixed real-reference
statistics and measures newly generated images against them. It does not
compute real-reference statistics from cached reconstructions. This audit
checks the configured file and evaluation code; it does not recompute the
full original-image FID reference statistics.

The watermark-bearing photograph at index 89,267 is among the independently
re-encoded originals. Its watermark is present in the original JPEG bytes.
Removing such originals would change the full training population. No data
filter, model change, cache replacement, or training restart was made.

Reproducible audit code, original-image key and byte-hash records, numeric
comparisons, and the log are retained in the active run's `source-audit/`
directory and uploaded to W&B.
