# What the successful FFHQ 8.17 run actually did

The exact reference is [ffhqcmp0804205803](https://wandb.ai/helloimlixin-rutgers/laser/runs/ffhqcmp0804205803),
not the later constant-LR v5b run linked in the historical results page.
Its W&B configuration, summary, complete requested training history, and uploaded
trainer source were retrieved again on September 23. The final recorded
`val/fid_50k_vs_full_train` is **8.174392700195312**, epoch 200, update 109,200.
W&B records 50,000 generated and 70,000 real images.

Evidence is saved in `outputs/ffhq817-transfer-audit-20260923/`. The downloaded
trainer is byte-identical to `src/ffhq_v4_archived.py`, SHA256
`9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f`.
This verifies the uploaded trainer, not a complete historical runtime or an
independent reproduction of FID. In particular, its base evaluator limits real
features to the fake sample count; the later full-training-reference metric is
explicitly recorded in W&B but its complete evaluation wrapper is not included
in this source file. The separately named full-training-reference FID history
retrieved here begins at epoch 135; earlier generic FIDs should not be assumed
to have identical reference population handling.

## The successful run also had coefficient CE above six

The final logged FFHQ training batch has:

| Coefficient metric, nats | Value |
|---|---:|
| Cross-entropy | 6.609153 |
| Soft-target entropy | 6.521400 |
| KL from target to prediction | 0.087752 |

The decomposition is `CE(q,p) = H(q) + KL(q || p)`. Raw CE was never supposed
to converge to zero with these soft labels. Across logged training batches in
the final epoch, median CE is 6.6066 and median KL is 0.08448. These are training
measurements, not held-out FFHQ likelihoods.

The current balanced Church run's first 300-image validation probe gives
`6.54713 = 4.61572 + 1.93141`. On its separate fixed 128-image conditioning
probe, coefficient KL decreases from 1.93683 at epoch one to 1.80399 at epoch
two. The current residual error is substantial, but neither raw CE nor KL is
a directly matched cross-dataset comparison: the target distributions,
conditioning noise, training progress, and train/validation populations differ.
Target entropy is a mathematical lower bound; uncertainty in an autoregressive
prediction can prevent attaining it.

## Verified recipe differences

| Component | Successful FFHQ run | Current balanced Church run |
|---|---|---|
| Sparse representation | 2,048 atoms, 2 pairs/site, 128 events | 16,384 atoms, 4 pairs/site, 256 events |
| History architecture | Width 1,024; 24 spatial layers, 4 within-site layers | Width 1,024; 24 atom layers and 8 coefficient layers over all events |
| Current-atom input to coefficient head | Projected frozen dictionary vector | Learned atom-ID embedding |
| Atom/history fusion | Two separate tokens through a two-layer causal micro-transformer | Four individually normalized feature fields summed before the coefficient decoder |
| Completed-pair embedding | Dictionary vector, coefficient embedding and physical contribution concatenated through a normalized adapter, plus physical residual | Learned atom ID, coefficient embedding and projected physical contribution added |
| Coefficient classifiers | One per sparse depth | One per sparse depth |
| Coefficient vocabulary | 2,048 uniform centers in normalized [-3,3] | 2,048 nonuniform centers in raw physical units |
| Soft target | Squared normalized distance / 0.5 | Squared physical distance / 0.125 |
| Training contexts | Sample coefficient IDs; reuse cached supports | Sample coefficient IDs and one of 16 cached OMP trajectories/site |
| Global batch | 128 | 2,048 |
| Updates/epoch | 546 | 62 |
| Training duration | 200 epochs / 109,200 updates completed | 300 epochs / 18,600 updates planned |
| LR | 5e-4, cosine to zero, no warmup | Same peak and schedule family |
| Classification weights | Atom 1.5, coefficient 1 | Same |
| Auxiliary geometry loss | Weight .05, delay 2 epochs, ramp 3 | None |
| Sampling | Atom top-k 250; coefficient top-p .85; temperatures 1 | Atom top-k 250; coefficient top-p 1; temperatures 1 |
| Cache | Prebuilt atom/coefficient pairs | Prebuilt raw full-training trajectory bank |

FFHQ's coefficient scales are `[36.208333333333336, 8.583333333333334]`.
Its normalized target standard deviation is approximately .5 away from the
finite boundaries, corresponding to physical standard deviations about
`[18.1042, 4.2917]`. Current Church targets have physical standard deviation
approximately .25 before finite/nonuniform-grid effects. Copying temperature
0.5 between these coordinate systems would not reproduce the same perturbation.
The historical encoder contains a normalized coefficient clamp; the original
cache's empirical saturation rate has not been recovered in this audit.

The FFHQ trainer uses a fixed cached support per image; its stochastic code
flag must not be read as evidence of the current 16-alternative stochastic OMP
bank. Its very low final training atom NLL (0.1349 on the last logged batch)
does not establish comparable held-out performance for Church.

## What is worth transferring

The clearest architectural candidate is **explicit dictionary-vector
conditioning with separate atom/history feature handling**. In the archived
model, the coefficient head receives `[history_hidden, projection(D[atom])]`
as two attention positions, then adds the resulting atom-position output to
the history residual. This preserves a dedicated path for atom geometry.
The current coefficient head receives the current atom through an ID embedding;
the current dictionary vector only contributes to later completed-pair history.
The recent field normalization fixes magnitude imbalance but does not recreate
the historical representation or fusion.

The current epoch-two corruption probe supports investigating that distinction:
shuffling the current atom raises coefficient KL by 0.3770 at depth zero,
but only `[0.00402, 0.00080, 0.00235]` at depths one through three. Shuffling
completed-pair history raises mean coefficient KL by 0.3071. Thus history is
being used, while learned use of the current atom is still weak at later depths.
This is an early diagnostic, not proof that a dictionary-vector branch will
improve generation.

A controlled candidate should add the dictionary-vector/current-atom fusion
while preserving both complete-history causal decoders, raw unclipped
coefficients, the same cache, and the same targets. Compare coefficient KL
by depth and atom-shuffle penalties at matched updates and image exposure;
confirm generation quality with the same FID protocol before selecting it.

The smaller FFHQ batch is a separate optimization variable: it produced about
5.87 times as many total updates as this run's entire planned budget, despite
fewer total image presentations. More updates are not proven to be the missing
ingredient. The earlier Church hierarchical model reached **FID50k 9.95524 at
epoch 57 / update 3,534** with batch 2,048, this same raw coefficient cache and
temperature .125, no geometry loss, and unrestricted coefficient sampling.
That same-dataset evidence makes representation/conditioning a more focused
first comparison than copying FFHQ's entire optimizer and sampling recipe.

There are also historical choices to treat cautiously. The old RQ hierarchy
summarizes completed sites and resets the within-site attention context; copying
it would not preserve the user's current requirement that both decoders attend
directly to every previous event. The old auxiliary geometry objective multiplies
an expected atom by the coefficient expectation conditioned on the ground-truth
atom, which is not the joint expectation of the compound model. The signed-atom
counterexample and correction are documented in
[the joint-geometry audit](lsun-church-joint-geometry-2026-09-12.md).
Its presence in a successful run does not justify restoring that inconsistency.

This audit changes no training weights, targets, optimizer state, sampling
settings, or frozen production source. At the recorded check, the current
worker, supervisor and monitor were alive at epoch-two FID evaluation, with
epoch-one checkpoints online. Monitoring and automatic recovery remain active.
