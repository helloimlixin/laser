# Church hard-target checkpoint: bounded sampler diagnostic, 2026-09-29

Frozen step 1800, SHA256 `0a5bafbd6ba2b140a73c7ec2b18891a65e8935e3cb26d4eb9b486696d0ae18a3`. Generated 256 images per policy; no FID calculation or training changes. Weights verified unchanged before/after evaluation. The full control matched the native sampler exactly for codes and generator state.

Policies: full atom/conditional coefficient vocabularies at T1; historical-like atom top-k700/T1 followed by the selected atom's coefficient at T0.9/top-p0.85. Repeated atoms allowed in both. Filtering is conditional per factor, not global joint top-k. Threshold ties can retain slightly more than 700 atoms.

| Level | Top700 mass, full-generated histories | Top700 mass, true histories | Actual retained atom mass, filtered histories | Retained coefficient mass at T1, filtered histories |
|---|---:|---:|---:|---:|
| 1 | 60.40% | 60.21% | 73.49% | 82.24% |
| 2 | 20.29% | 20.04% | 25.66% | 82.19% |
| 3 | 11.35% | 11.24% | 12.96% | 82.33% |
| 4 | 10.84% | 10.40% | 10.05% | 82.53% |

Full and filtered histories diverge; these columns are policy-specific descriptive estimates, not measurements at identical contexts. Coefficient mass is conditional on the selected atom, not exact joint retained probability. The true-history probe uses 64 training images, not a held-out generalization test.

Later-level atom distributions are broad even on true histories. This supports testing conservative truncation, while it also shows why copying top-k700 across every level removes substantial model probability. It does not establish a training root cause. Qualitative inspection of the first grids suggests smoother surfaces under filtering, but recognizable structural failures remain; 256 images do not establish improved FID or preserved diversity.

A sensible next matched evaluation is mild conditional nucleus filtering (for example atom/conditional coefficient top-p0.95 at T1), compared with full sampling on the same checkpoint. This is a candidate, not a validated setting or a production change. Keeping T1 would isolate filtering before tuning temperatures.

The older fixed-checkpoint experiment measured historical filtered FID9.493208 versus full T1 FID11.169160; that comparison jointly changed filtering and coefficient temperature and used an older architecture/codec. See [matched historical report](church-untruncated-sampling-2026-09-29.md).

[W&B grids and verified artifact](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-hard-sampler-step001800-20260929). Artifact: `helloimlixin-rutgers/laser/church-hard-sampler-step001800-20260929-diagnostic:v0`. All 19 published entries passed remote manifest digest/size checks. CUDA peak fields in results are cumulative process peaks, not independent per-policy peaks.
