# Church sampling scope correction — September 29, 2026

The user clarified that the request to follow DCTransformer concerns sparse-code sampling on the existing RQ model. Creating and training a fresh DC-style architecture was an assistant scope error. Its supervisor and workers were stopped, with its checkpoints and audit history preserved. The interrupted step-1240 FID is unscored. The fresh architecture is not the active recipe.

The intended model is the preserved 404,738,048-parameter compound RQ model at step 10540, whose historical 50k FID is 9.493208036. Its checkpoint SHA256 is `8e9a209aec912909b33ec8fd92c18c42d41f09a5d57e431b05a39d3285d9609b`. The original training already completed 300 epochs; its final and best checkpoints remain archived on W&B.

The existing sampler draws an atom from its conditional distribution, draws the coefficient conditioned on that selected atom and previous completed pairs, then adds the completed pair to history. This expresses the conditional categorical generation principle described in DCTransformer. Fixed LASER positions do not require a sampled position field. The published paper does not establish joint-pair top-k ranking or numerical top-k, top-p and temperature settings, and a verified official sampling implementation was not found. Full-vocabulary temperature-1 sampling is an explicit comparison setting, not a claim to reproduce undocumented implementation details. [Paper](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf).

The required sampling-only comparison already completed on that frozen checkpoint, with 50,000 images per setting and matched logical RNG streams:

| Setting | FID | W&B |
| --- | ---: | --- |
| Historical: atom k700, T1; coefficient p0.85, T0.9 | 9.493208036 | [Historical evaluation](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-untruncated-best10540-20260929-historical) |
| Full eligible atom and coefficient vocabularies, no nucleus filtering, both T1 | 11.169160294 | [Untruncated evaluation](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-untruncated-best10540-20260929-untruncated) |

Both evaluations preserve model weights, tokenizer, coefficient quantization and depth scales. Source, previews, feature statistics and online verification receipts are retained in `outputs/church-untruncated-20260929/`. No additional training or duplicate 50k evaluation is needed to recover these results. The historical setting remains the measured quality baseline; the explicit untruncated setting remains available for comparison.
