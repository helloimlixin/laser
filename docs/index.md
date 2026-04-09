# LASER Docs

This folder holds short notes for the maintained code path.

## Core Notes

- [Mathematical Note](/scratch/xl598/Projects/laser/docs/math_note.md): strict
  analysis of the current stage-1 bottleneck, stage-2 prior, and paper
  readiness.

## Code Map

- Stage 1: [laser.py](/scratch/xl598/Projects/laser/src/models/laser.py) and
  [bottleneck.py](/scratch/xl598/Projects/laser/src/models/bottleneck.py)
- Stage 2 train: [train_s2.py](/scratch/xl598/Projects/laser/train_s2.py)
- Stage 2 sample: [sample_s2.py](/scratch/xl598/Projects/laser/sample_s2.py)
  and [gen_s2.py](/scratch/xl598/Projects/laser/gen_s2.py)
- Stage 2 runtime: [s2.py](/scratch/xl598/Projects/laser/src/s2.py)
- Token caches: [build_token_cache.py](/scratch/xl598/Projects/laser/build_token_cache.py)
  and [extract_token_cache.py](/scratch/xl598/Projects/laser/extract_token_cache.py)

## Naming

The maintained short names use the `s2` prefix for stage-2 scripts, sample
artifacts, and run names. Older `*_ar.py` files remain as thin compatibility
wrappers.

## Next Docs To Add

- token cache format and invariants
- run layout and checkpoint naming
- experiment notes and failure cases
