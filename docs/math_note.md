# Mathematical Note on LASER

Date: April 5, 2026

## Summary

The current stage-1 model is best described as a straight-through sparse
projection autoencoder, not classical K-SVD. The current stage-2 story splits
in two: quantized mode is a proper discrete autoregressive model, while the
real-valued path is usually autoregressive support prediction plus coefficient
regression rather than a clean likelihood model.

## Stage 1

For each latent site or patch `u`, the intended sparse approximation problem is

```text
min_a  1/2 ||h_u - D a||_2^2    subject to ||a||_0 <= K
```

with dictionary `D in R^(m x M)`. The maintained defaults are `M = 1024`,
`K = 8`, `embedding_dim = 4`, `patch_size = 4`, so patch-mode sparse coding
runs in dimension `m = 64`.

The implemented forward path infers sparse codes with a pursuit routine,
reconstructs `h_dl`, and uses the straight-through replacement
`h_st = h + sg(h_dl - h)`. Patch mode uses OMP-style support growth; non-patch
mode uses one-shot top-`k` correlation selection. So the honest description is
"OMP-like in patch mode, top-`k` in non-patch mode."

Atoms are normalized and dictionary gradients are projected to the sphere
tangent space, which is good. But there is no theorem-backed recovery claim at
the maintained defaults: a standard OMP condition is `mu(D) < 1 / (2K - 1)`,
which for `K = 8` means `mu(D) < 1 / 15 ~= 0.0667`, and the default coherence
penalty is off.

## Stage 2

Quantized mode defines a proper discrete autoregressive factorization over
sparse token grids. Real-valued mode is mathematically weaker unless the model
uses an explicit density such as the Gaussian NLL path.

Sparse-latent AR is feasible. With `M = 1024`, `K = 8`, and 256 coefficient
bins, an unordered support-plus-coefficient description costs about
`log2 C(1024, 8) + 8 log2 256 ~= 128.7` bits per site, which is workable.

The maintained cache writers now canonicalize support order by ascending atom
id before stage 2, and cache loading canonicalizes legacy rows as well. That
removes the old nuisance where stage 2 modeled an ordered tuple for an
unordered sparse object.

## Assessment

The idea is real and publishable with cleanup, but the current implementation
is not mathematically clean enough for a NeurIPS best-paper claim. The next
fixes are: choose one clean probabilistic story for coefficients, unify the
sparse solver story, and report coherence, support stability, permutation
entropy, and bits per site.
