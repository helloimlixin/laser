# DCTransformer sampling check, September 29, 2026

Primary sources are Nash et al. (ICML 2021), Generating Images with Sparse Representations, and its PMLR supplement. Both PDFs were downloaded and read; exact source hashes are in sources.json.

Paper page 4, equation 2, models each nonzero DCT tuple with p(channel | previous tuples) × p(position | channel, previous tuples) × p(value | channel, position, previous tuples). Each field has a categorical predictive distribution. Section 3.2 stacks separate decoders while preserving conditioning on the available tuple history. Section 3.3 (page 6) samples a continuation chunk, adds those values to the partial DCT image, and repeats until the stopping token or chunk limit.

This supports probability-based conditional generation. It does not describe selecting a support by OMP residual distance, nor ranking all complete tuples with a joint top-k operation. Representation construction uses fixed DCT blocks, quantization, sparse nonzero triples and a low-to-high-frequency sequence order.

No numeric top-k, top-p/nucleus, or generation-temperature prescription was found in either complete PDF. This absence does not establish that the unpublished sampling implementation applied no filters. No official local sampler was found. The public benjs/DCTransformer-PyTorch repository explicitly labels itself unofficial; its main tree b00594fbfb86f9287404d64146012aa2cf07492f contains transforms.py and demo.ipynb, not a trained Transformer sampler. It therefore cannot verify the authors' exact sampling hyperparameters.

Our current compound model has a valid analogous factorization p(atom | previous pairs) × p(coefficient | atom, previous pairs). Its generation truncation is an extra choice: atom k700, p1, temperature1; coefficient k2048, p0.85, temperature0.9. The new training teacher's physical reconstruction-energy distribution is a distinct training mechanism, not the generation top-k metric.

The earlier agreement that atom-first conditional sampling was inherently wrong for sparse coding was too strong. Joint-pair top-k is a coherent experimental sampler, but is not established as the DCTransformer recipe. Its isolated CPU implementation passed9 tests and160 exhaustive-reference cases; no GPU benchmark or deployment was performed after the user's DCTransformer steering. Current 300-epoch training and its evaluation settings remain active.

A clear diagnostic suggested by the published factorization is a current-checkpoint comparison with full eligible atom/coefficient vocabularies, temperatures1 and no truncation, retaining support uniqueness and complete-pair history. This is a proposed baseline, not a claim to reproduce undocumented DCTransformer sampling settings or to improve FID.

Sources: [paper](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf), [supplement](https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf), [unofficial repository](https://github.com/benjs/DCTransformer-PyTorch).
