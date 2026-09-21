# ImageNet VAR dictionary-learning trial (2026-09-16)

This trial exercises the existing causal LASER-between-scales formulation in
`src/models/multiscale_laser_var.py` and `src/training/var_laser.py`. It is a
separate smoke run and does not resume or alter the production scratch run.

- Bottleneck: shared 4,096-atom dictionary, two OMP-selected atoms per site,
  257 quantized coefficient values.
- Scales: `1,2,3,4,5,6,8,10,13,16`.
- Transformer sequence: exactly 680 spatial positions, matching original VAR
  (`sum(p^2)` over the ten scales). Sparse depth stays inside a per-position
  causal output head and therefore does not enlarge the self-attention sequence.
  With sparsity two, the representation contains 1,360 atom/coefficient pairs
  (2,720 categorical decisions), but they are not 2,720 transformer positions.
- Causality: the input at scale `s` is reconstructed only from scales `< s`;
  within a site the prior predicts atom then coefficient for each sparse depth.
- Initialization: fully random tokenizer and VAR-d16 prior.
- Data path and transforms: full ImageNet manifests, 288px Lanczos resize and
  deterministic 256px random crop, identical to the matched scratch recipe.
- Trial budget: two tokenizer updates and three prior updates. The third prior
  update was produced while resuming the saved smoke checkpoint to complete the
  preview after fixing current-SciPy compatibility in the pinned FID evaluator.
- Effective batches on one H200: tokenizer 128 (`32 x 4`), prior 768 (`96 x 8`).
- W&B run: `imagenet-var-laser-dict-trial-20260916`.

The focused model tests passed before launch (11 tests), covering cross-scale
no-leakage, sparse-depth causality, round-trip reconstruction, gradients, and
matched scratch initialization.
