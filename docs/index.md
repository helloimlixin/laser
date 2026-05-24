# LASER Docs

This folder holds short notes for the maintained code path.

## Core Notes

- [Mathematical Note](/scratch/xl598/Projects/laser/docs/math_note.md): strict
  analysis of the current stage-1 bottleneck, stage-2 prior, and paper
  readiness.

## Code Map

- Stage 1 train: [train_stage1_autoencoder.py](/scratch/xl598/Projects/laser/train_stage1_autoencoder.py)
  (model in [laser.py](/scratch/xl598/Projects/laser/src/models/laser.py) and
  [bottleneck.py](/scratch/xl598/Projects/laser/src/models/bottleneck.py))
- Token cache: [cache.py](/scratch/xl598/Projects/laser/cache.py)
- Stage 2 train: [train_stage2_prior.py](/scratch/xl598/Projects/laser/train_stage2_prior.py)
- Stage 2 sample: [sample.py](/scratch/xl598/Projects/laser/sample.py)
- Stage 2 runtime: [s2.py](/scratch/xl598/Projects/laser/src/s2.py)

## Naming

The pipeline has one maintained entry point per stage:
`train_stage1_autoencoder.py` → `cache.py` → `train_stage2_prior.py` → `sample.py`.
The older `train.py`, `train_s2.py`/`train_ar.py`, `sample_s2.py`/`sample_ar.py`,
`gen_s2.py`/`generate_ar.py`, `build_token_cache.py`, and `extract_token_cache.py`
scripts have been removed in favor of these.

## Next Docs To Add

- token cache format and invariants
- run layout and checkpoint naming
- experiment notes and failure cases
