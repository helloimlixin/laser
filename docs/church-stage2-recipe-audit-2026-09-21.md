# Stage-two loss, LR, and sampling audit, September 21, 2026

The current compound run uses the appropriate categorical cross-entropy loss
family and a correctly implemented cosine learning-rate schedule. Its peak LR,
AdamW betas, and weight decay match the original RQ-Transformer paper. It remains
a compound-token adaptation, with different targets, loss weighting, batch size,
training duration, and sampling settings.

The run reached generation FID 11.8516 at epoch 50, compared with the previous
one-epoch-tokenizer run's best 17.2044. Its 90-epoch training continued throughout
this CPU-only audit.

## Primary sources

- [RQ-Transformer paper](https://arxiv.org/pdf/2203.01941): equations 14–15, section 3.2.3, appendix A.3, table 5, and appendix B.1/B.3.
- [Released Church checkpoint](https://github.com/kakaobrain/rq-vae-transformer#pretrained-checkpoints): archived stage-two config independently confirms top-k 1400, top-p 1, temperature 1.
- [DCTransformer: Generating images with sparse representations](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf): section 3, equation 2, and section 3.3.
- [DCTransformer supplement](https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf): appendix C and table 3.

All three PDFs were downloaded and extracted locally. Their URLs and SHA256
digests are recorded in `outputs/church-stage2-recipe-audit-20260921/sources/manifest.json`.
DCTransformer refers to Nash et al. (ICML 2021), as in the repository's earlier
Church notes. Its LSUN model uses variable-aspect-ratio images with long side
384; its published FID is not a controlled comparison against our 256-square run.

## Recipe comparison

| Setting | Original RQ-Transformer, Church | DCTransformer, LSUN | Current compound run |
|---|---|---|---|
| Optimizer | AdamW, betas (0.9, 0.95) | Adam; betas not specified in the cited supplement | Fused AdamW, (0.9, 0.95) |
| Peak LR | 5e-4 | 5e-4 | 5e-4 |
| Schedule | Cosine | 1000-step linear warmup, then cosine | Cosine, no warmup |
| Weight decay | 1e-4 | Not specified in the cited supplement | 1e-4 |
| Batch | 2048 images | 512 chunks | 1536 images, 8×192 |
| Budget | 300 epochs | 300 billion target elements | 90 epochs, 7380 optimizer updates |
| Supervision | Distance-based soft code CE and stochastic code contexts | Categorical channel/position/value likelihood | Hard atom CE, soft coefficient CE, stochastic coefficient contexts |
| Relative head weighting | Mean over code positions/depths | Joint categorical factorization | Atom:coefficient 1.5:1 |
| Sampling | k=1400, p=1, temperature=1 | Draws from predicted categorical distributions; no numeric k/p/temperature prescription identified | Atoms k=250, p=1; coefficients k=0, p=.85; both temperatures 1 |

The RQ paper does not specify a stage-two warmup in appendix A.3. Its stage-one
warmup description must not be treated as a stage-two instruction.

Using our 126227-image population, the paper's Church duration and batch imply
about 18490 updates; exact rounding depends on the unpublished training loader.
Our 7380 updates are about 40% of that count. The requested 90 epochs are 30%
of the paper's image-epoch budget. The shortened cosine schedule reaches low LR
earlier in absolute image exposure. This is an explicit duration difference,
not an optimizer bug. The user requested 90 epochs, so this audit did not extend it.

DCTransformer's chunk/token budget cannot be converted directly to LASER image
epochs: it models long variable-length sequences of channel-position-value
triples, whereas our images contain 256 fixed compound events. Its 1000-step
warmup would occupy 13.6% of our complete 7380-step schedule. The current driver
only honors `warmup_epochs` with its `warmup-linear` scheduler; reproducing a
warmup followed by cosine would require that scheduler to be implemented explicitly.

## Verified loss and target behavior

With the enabled settings, the actual backward objective is
`(1.5 * atom hard CE + coefficient soft CE) / 2.5`, averaged across all spatial
positions and sparse depths. Geometry, coefficient regression, CRPS, and prefix
auxiliary weights are zero. This is a weighted training objective, rather than
the unweighted joint likelihood of the two categorical variables.

A CPU probe extracted the actual objective from the frozen training source and
compared it with independent cross-entropy calls and analytic softmax gradients.
The loss matched exactly; maximum gradient errors were 3.73e-9 for atoms and
9.31e-10 for coefficients. Across 492 production log records through step 4920,
the weighted-loss identity held within 1.34e-6 and the cosine LR formula within
1.47e-18. All examined losses were finite and all geometry weights were zero.

RQ-Transformer softens over alternative code embeddings and stochastically
samples residual-code contexts. Our cached atom support remains deterministic;
only coefficient inputs are sampled. The coefficient target is proportional to
`exp(-(physical_coefficient - physical_bin)^2 / 0.125)`. Distances include the
tokenizer's depth-specific coefficient scales. RQ's reported temperature 0.5 is
in a different representation, so comparing the scalar temperatures alone does
not compare their noise levels.

The current targets have approximately 0.25 standard deviation in physical
coefficient units, equivalent to about 65, 134, 205, and 323 effective bins by
depth on the cache probe. The target entropy contributes about 5.05 nats to the
coefficient CE; coefficient KL subtracts this fixed target entropy and is more
useful for judging predictive error. Narrowing the temperature to 0.03125 would
halve this physical standard deviation to 0.125. This establishes a testable
change, not evidence that narrower targets will improve generation.

DCTransformer's categorical integer-value prediction motivates a hard-target
control. Its paper does not establish that hard targets are best for learned
LASER coefficients, nor does it prescribe a separate latent geometry loss.

## Sampling and logging

The released Church RQ-Transformer's k=1400 is substantially wider by vocabulary
count than our atom k=250. Our coefficient nucleus p=.85 has no direct counterpart
in its single-code sampler. Both settings affect generated images without
changing model weights or training losses, making them the first cheap comparison.

The W&B configuration includes legacy generic fields `temp=.125` and `top_p=.92`.
The former is the coefficient **training-target** temperature; the latter is a
hardcoded legacy metadata field. The actual sampler uses the explicitly prefixed
`atom_temperature`, `coeff_temperature`, `atom_top_k`, `atom_top_p`, `coeff_top_k`,
and `coeff_top_p` settings listed above. Both production FID and scheduled
previews receive those actual arguments. No inference temperature .125 or
inference nucleus .92 is used in this run.

## Proposed controlled comparisons

1. Freeze one best checkpoint and screen the 2×2 sampler combination of atom
   k in {250, 1400} and coefficient p in {.85, 1.0}, holding both temperatures
   at 1 and atom p at 1. Use identical seeds and sample counts, then confirm the
   selected sampler and baseline with 50000 samples and an independent seed.
2. Keep LR and sampling fixed while independently testing atom weight 1.0 and
   coefficient-target temperature 0.03125. A DC-style hard-target control would
   be a separate comparison. Do not change all loss knobs simultaneously.
3. A peak-LR comparison of 3.75e-4 against 5e-4 is defensible: 3.75e-4 is the
   simple linear batch-ratio adjustment from the paper's 2048 to our 1536.
   Linear scaling is a heuristic, not a requirement or a proven optimum.

The current run already uses the papers' peak LR and has stable, improving
training. Sampling and target calibration are the higher-priority checks.
At the time of the audit, no new training run or sampling sweep was launched. The four
sampler definitions and isolated training changes are recorded in
`outputs/church-stage2-recipe-audit-20260921/proposed-comparisons.json`.

The user subsequently selected comparison 1. Its completed independent
50000-image confirmation favored atom top-k 250 and coefficient top-p 1.0:
FID 10.9382 versus baseline 11.7321 on the same frozen epoch-50 checkpoint.
The 1400-atom settings did not win screening. Structural distortions remain
visible. Full results, online publication, and the corrected disjoint rank-seed
protocol are recorded in `church-compound-sampling-2026-09-21.md`.
The proposed training comparisons remain unlaunched.
