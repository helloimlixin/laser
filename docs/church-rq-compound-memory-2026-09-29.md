# Compound plane-memory RQ with shared physical coefficient bins

The user corrected the preceding integer-vocabulary experiment: use separate compound atom/coefficient fields and explicitly restore the earlier DC adaptation's shared 2,048-bin physical grid. This is a fresh stage-2 model, keeping the cached 20-block plane encoder and six-block causal self/cross decoder. The frozen Church tokenizer and encoder latents are unchanged. No coefficient refitting, reward, contrastive loss, or depth-dependent physical scaling is introduced.

Runtime: `/tmp/laser-church-rq-compound-memory-20260929`.

Live W&B run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-rq-compound-memory-20260929

## Representation and factorization

One event stores two fields `(atom_id, coefficient_id)`, with 16,384 atom choices and 2,048 shared coefficient values. The exact existing FP32 bin table spans approximately [-22.9871,22.9871], SHA256 `919327b20b21af1b508711bd0f068b0ad1725f6dc1cbfd876dd698ad30ef5a01`. Every depth uses the same table. There is no Cartesian-product categorical classifier and no separate zero sentinel.

The model factorizes each complete event as `p(a|H) p(c|a,H)`. `H` contains all individual earlier-plane pairs and all earlier pairs in the current plane. The history transformer advances once per pair. The coefficient head consumes the same history state plus the selected atom's identity and frozen dictionary vector; it cannot see the current coefficient. The pair is committed only after both fields are drawn. The model has 392,392,704 parameters across 503 tensors.

Sampling draws from the complete atom distribution and then all coefficient bins at temperature one. Repeated atoms remain possible, consistent with the residual teacher. This deliberately replaces the integer run's top-1400/95%-mass policy with conditional ancestral sampling; applying truncation independently to the fields would define a different joint law.

## Physical pair soft targets

The intended teacher law is `q(a,c|r) ∝ exp(-||r-c D_a||²/T)` with T=0.125. At each depth it computes the coefficient-marginal atom distribution, samples an atom, computes its full conditional coefficient distribution, samples the coefficient, and subtracts that exact physical contribution. No previously chosen coefficient is refitted.

The objective is atom-marginal soft CE plus coefficient-conditional soft CE for the same teacher-sampled atom, with no factor of one half or separate loss weights. The coefficient term is an unbiased estimator of the teacher-weighted conditional loss. The two target fields are coupled through the same physical-vector distribution; they are not independent marginal labels.

For speed, broad Gaussian coefficient partitions far from both grid boundaries use a bounded lattice approximation. Boundary, narrow-kernel, and nonuniform-grid cases use direct finite-bin summation. Every coefficient conditional uses the actual bin centers. The approximation explicitly accounts for FP32 grid rounding: a 3,603-case dense FP64 scan measured maximum log-partition error 2.62e-6 against a conservative 4.74e-5 bound. This is a documented numerical approximation, not exact finite-grid equality. It avoids materializing a 16,384-by-2,048 joint target per event.

The original DCTransformer uses hard categorical likelihood for channel, position, and coefficient value. Its supplement also biases training-chunk selection toward earlier low-frequency content. The physical-vector soft teacher here is our extension, not a claim about the original objective. Sources: [paper, section 3](https://proceedings.mlr.press/v139/nash21a/nash21a.pdf), [supplement, Appendix C](https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf).

## Training and handoff

The new model and AdamW start fresh with global batch 2,048 across eight GPUs, microbatch 128 and two accumulation steps. The planned 300 epochs total 18,600 updates; preview 64 samples every 200 updates and evaluate official continuous-pixel FID on 50,000 images every 620 updates. Upload full LAST at step 200 and LAST/BEST at every FID/final boundary. The reference features, FP32 tokenizer decoder/Inception path, and eight logical evaluation streams are retained.

The original request to wait for epoch 30 was fulfilled before the currently running integer plane-memory model launched. This representation correction switches that current model only at a newly armed completed checkpoint after CPU review; it preserves full LAST and all selected BEST before terminating identified processes. Any FID scheduled at the selected checkpoint must finish. The outgoing run's final LAST/BEST and lineage are published and durably preserved.

CPU tests cover factorized loss/gradient equivalence, physical teacher histories, isolated RNG, all 256 cached predictions, current-coefficient/future leakage, conditioning on the current atom, complete earlier-pair influence, and cache advancement once per pair. Before production, all eight GPUs must pass production-size training and cache checks, a disposable fixed-64-image learnability probe, sampling capacity at 1,024 images per GPU, and resume verification. Pilot weights are discarded. Production starts only after these checks and outgoing online checkpoint publication succeed.


## Conditioning correction before launch

The first compound prototype passed causality/cache/gradient tests and learned both heads, but failed the predeclared lower-plane-use check on64fixed trainingexamples. Even at the first site of later planes, where no current-plane prefix is available, support prediction barely responded to shuffling the earlier planes. The failed results and original sources/checkpoints are preserved under `superseded-no-prefix`; no production run used those weights.

The corrected model adds a bias-free256-to1024projection of the physical sum of completed pairs at the query site to each decoder input. This is additional nativeRQ-style residual-state conditioning; all192individual global memoryslots remain available through cross-attention. Dense and cached inference compute the same causal prefix. The firstplane receives an exactlyzero prefix. Targets, physical coefficient table, objective, sampling, and schedule are unchanged.

The same128-update probe and original thresholds passed with this path. Later-plane lower-history shuffling increased support CE by10.35/10.74/8.31nats and conditional coefficient CE by0.59/0.61/0.55nats, with current atoms and current-plane histories preserved. First-site support CE improved from5.01/6.10/8.22 to1.22/2.75/6.27. This demonstrates use of earlier-plane conditioning on the diagnostic subset; it does not isolate the cross-attention path from the added localprefix or establish FID improvement.17candidateCPUchecks cover causality, both-head cache parity, projection gradients, zero-prefix equivalence and continued remote-site memory access. Production factory initialization exactlymatches that tested candidate.

## Verified production launch

Production launched fresh on all eight GPUs on 2026-09-29 at 18:31 UTC, after the outgoing integer run retired at step 1200 and its full LAST/BEST checkpoints were verified on W&B and in durable storage. The new controller PID is 301004 and initial torchrun PID is 301008. All eight initial model hashes match, all Adam states started empty, all 503 parameter tensors receive gradients, and separate per-rank model/teacher RNG streams were restored after W&B initialization. The source artifact is `church-rq-compound-memory-20260929-source:v0`; all 57 frozen source hashes and 68 artifact entries were independently verified.

Final validation at the production batch size measured 1.036 s/update and 47.98 GiB peak allocated memory at global batch 2048. Full conditional cached generation produced 8192 grids in 2.615 s (3133 images/s aggregate, generation only); this excludes decoding, features, and FID. All 256 predictions passed both-head dense/cache checks in FP32 and BF16. The unchanged 128-update diagnostic learning probe reduced joint KL from 12.9363 to 0.3881, and lower-history shuffling increased later-plane CE by 10.94/11.35/8.85 nats. These are training-subset diagnostics, not FID or held-out evidence.

Checkpoint continuation passed the predeclared numerical gate with maximum model difference 5.96e-8 and Adam-moment difference 5.82e-11. RNG streams, cursor, plan, source identity, and loaded checkpoint tensors matched exactly; continuation is not bitwise identical. Final receipts are preserved under the durable output directory.

At step 200, the first 64-image preview was independently downloaded from W&B and matched the local image hash. W&B selected-checkpoints:v0 contains the full 4,709,419,478-byte LAST checkpoint (SHA256 `5bbc25045d8afdb58a0a42d61dd80a8e7cf02243d393c91d7896be23273d1242`), including all 503 model/Adam tensors, scheduler step 200, and eight model plus eight teacher RNG streams. BEST is not yet defined; the first 50,000-image FID is scheduled at step 620. Source, recovery assets, preview, and LAST are also copied to the durable output directory.
