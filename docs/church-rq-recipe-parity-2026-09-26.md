# Church: original RQ recipe and stochastic-teacher parity

The priority is now to match the original RQ training recipe and stochastic
mechanism before trying further noise reduction, EMA, or learning-rate changes.
The current LASER runs match many optimizer settings, but they do **not** use
the original RQ stochastic teacher. This audit has not changed their objectives
or restarted training.

## Reference and evidence

The reference is the [RQ paper](https://arxiv.org/html/2203.01941#A3), the
[released quantizer](https://github.com/kakaobrain/rq-vae-transformer/blob/main/rqvae/models/rqvae/quantizations.py),
and the original local run
`helloimlixin-rutgers/laser/church-original-rqvae-released-tokenizer-control-20260917`.
The downloaded public quantizer, loss, and optimizer files match the frozen
local upstream files byte for byte. Their SHA256 identities are recorded in
`outputs/church-rq-recipe-parity-20260926/upstream-source-verification.json`.

The public repository did not release its stage-two training loop. The original
local run's W&B config and logged driver were recovered. Its imported
`src.original_rq_training` helper was absent from the recovered logged source
and diff; the current local helper is inspectable, but its exact historical
version is not authenticated by those files. Claims of bitwise reproduction
of the authors' training loop would therefore be too strong.

The paper and local original run specify batch 2048; the public example YAML
says 256. The released checkpoint/local original sampler uses top-k 1400;
that example YAML says 250. These are distinct sources, not equivalent recipes.

## What matches, and what needs to change

| Setting | Original RQ reference | Current LASER |
|---|---|---|
| Optimizer | AdamW, betas 0.9/0.95, weight decay 1e-4 | Same mathematical settings; fused implementation |
| LR and budget | 5e-4 to zero, cosine, 300 epochs | Same planned schedule, 18,600 updates |
| Effective batch | 2048 | 2048 |
| Warmup / gradient clipping | Zero / norm 1 in released YAML and local driver | Same |
| Church image transform | Resize256, CenterCrop256, no random augmentation | Same transform; different tokenizer |
| Residual / attention / embedding dropout | 0.1 / 0 / 0 | **0.2 / 0 / 0**, verified in instantiated modules |
| Teacher choices | Fresh full-codebook soft distribution at each depth | **16 cached full-refit OMP trajectories per site** |
| Later-depth residual | Subtract the actual sampled earlier code vectors | Bank posterior given sampled earlier atom/coefficient IDs |
| Training temperature | 0.5 in squared physical vector-distance units | OMP 0.0625; coefficient 0.0625 in normalized coefficient units |
| Objective | Soft code CE, equally weighted residual-code events | **1.5 atom CE + coefficient CE**, divided by 2.5 |
| Training arithmetic | Local original uses FP16 autocast and GradScaler | BF16 autocast and FP32 loss |
| Backbone | Width1024, 24 spatial + 4 depth blocks, 16 heads | Same backbone dimensions, additional pair/coefficient modules |
| Token vocabulary | 16384 learned vectors | 16384 atoms and 2048 coefficient bins |

The numerical coefficient temperature cannot be copied directly. Current
coefficient bins are normalized by depth scales
`[7.662354, 4.158035, 2.633324, 1.651212]`. Consequently, a common normalized
temperature implies different physical noise at each depth. RQ instead uses
one residual-vector distance distribution to define both the sample and label.

The finite-bank atom posterior is a valid conditional distribution for its
specified bank model. Its limitation is a different, much smaller set of
possible targets; this audit is not evidence that its conditional computation
is wrong. Independent coefficient draws also do not trigger fresh recoding
from the perturbed residual across all dictionary atoms.

## Concrete causal-teacher prototype

`src/training/residual_pair_targets.py` implements the candidate law

```
q(atom, coefficient | residual)
    ∝ exp(-||residual - physical_coefficient * dictionary_atom||² / temperature)
```

It sums over coefficient alternatives to obtain the atom marginal, samples
that atom, then samples its conditional coefficient. Both exact soft targets
are retained. The selected physical contribution is subtracted before computing
the next depth. There is no trajectory bank or training top-k/top-p truncation.
Repeated atoms are allowed. Chunking bounds temporary memory without truncating
the distribution, although the complete teacher remains computationally costly.

This law is exactly the RQ Gibbs rule on an expanded vocabulary of atom/scale
vectors. It is an adaptation of RQ to LASER, not the original 16384-vector RQ
model. At 16384 atoms and 2048 bins it has 33,554,432 joint choices per depth.
The unweighted pair objective is `mean(atom soft CE + coefficient soft CE)`
per sparse event. The coefficient term conditions on the sampled current atom;
averaging these draws gives the coefficient expectation in the joint CE.
Dividing by two would change the overall gradient scale; retaining the current
1.5 atom multiplier would change the relative objective as well.

Full-refit OMP revises earlier coefficients after selecting later atoms.
The prototype instead fixes earlier sampled pairs. This is a substantive
representation change requiring reconstruction validation. The old orthogonal
coordinate path is not an automatic substitute: its constructor does not use
the current full-pair autoregression switch, so it would need its own causal
model audit before use.

Refitting is standard OMP behavior, not a bug. The selected atom IDs remain in
the support; all active coefficients are re-solved by least squares. Once the
encoding is complete, its final pairs are a valid autoregressive training
sequence. Their statistical dependence on later pairs does not by itself
invalidate autoregressive likelihood training. The difficulty arises when
trying to substitute a local residual-based soft target for the conditional
distribution of those final-refit pairs.

A separate possible adaptation preserves OMP and samples complete trajectories
fresh on every visit. Hard sampled atom targets then give a Monte Carlo joint
likelihood objective; coefficient soft targets must remain consistent with the
same trajectory. This would remove the finite-bank restriction, but would not
reproduce RQ's exact full soft atom labels. It should be named and evaluated as
an explicit adaptation, not conflated with the causal expanded-book prototype.

## Verification and launch status

Nine focused tests pass: explicit joint-distance agreement, fresh stochastic
visits and exact replay, prefix stability when depth is extended, repeated
atoms, spatial shapes, invalid temperatures, and joint-objective gradients.
An independent CPU comparison against the actual released RQ code finds:

- Identical sampled original-RQ codes under matched RNG, with soft-target
  maximum error `1.67e-16`.
- Pair-teacher atom marginal maximum error `1.67e-16` against an explicitly
  expanded upstream codebook, at the actual sampled prefix at every depth.
- Original soft-CE loss discrepancy `3.96e-8` and gradient discrepancy
  `1.95e-10`; its released logarithm adds `1e-7` to the denominator.

These are mathematical checks, not evidence of improved image FID. The
prototype is not connected to production training. The machine-readable
`candidate-recipe.json` explicitly records that it has not launched.

A full-vocabulary smoke probe subsequently passed on eight genuine latent sites
from four fixed random training images. Mean squared residual norm was 2.3951
for deterministic full-refit OMP, 2.4631 for stochastic full-refit OMP at0.0625,
and 6.4070 for the causal pair teacher at physical temperature0.5. All targets
were finite, with no coefficient samples on the grid edges. This tiny probe
does not measure image quality or establish which difference causes the larger
error: the coder and noise mechanism both changed. It provides no basis to
replace OMP immediately or launch a long run. CPU throughput here is not a
production GPU benchmark.

The fresh parity candidate sets dropout to 0.1, removes the extra atom loss
weight, and retains batch2048, the optimizer, and the 300-epoch cosine schedule.
It starts with physical distance temperature0.5, recording the representation
difference rather than claiming identical noise strength across tokenizers.
It needs a prequantization latent cache, reconstruction validation, compatible
training/sampling masks, precision checks, and a throughput preflight.

All three available GPUs should be used for that preflight and eventual run.
Physical batches should be maximized while retaining the exact global batch:
2048 is not divisible by three, so unequal final local counts require correct
image weighting under DDP. Padding this to 2049 or increasing the effective
batch would no longer match the selected recipe. Existing full-pair causal
fixes and complete checkpoint/RNG recovery must be retained.

Sampling settings are a separate comparison. Top-k1400 over atoms is not
top-k1400 over the expanded pair vocabulary. Any transferred sampler will be
labeled explicitly and evaluated against the same official FID reference with
paired independent seeds.

The completed [feature audit](church-feature-rebound-2026-09-26.md) remains
valid: selected LASER averaged FID50k 9.8116 versus the local original RQ's
10.0879 on two matched evaluations. The released pretrained RQ result near8.00
is a different baseline. Lower sampling temperatures did not improve the
screening result. Neither fact establishes which training mismatch causes the
remaining gap.
