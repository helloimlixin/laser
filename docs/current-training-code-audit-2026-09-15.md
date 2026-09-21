# Training code audit — 2026-09-15

**Fix status:** the findings below describe the pre-fix code. Repairs, regression
results, remaining runtime scope, and the new recovery command are documented in
[training-code-fixes-2026-09-15.md](training-code-fixes-2026-09-15.md). The old
working sources are archived with the reproduction artifacts; the live training
snapshot remains unchanged. The 50k sampling confirmation has also completed.

The review found seven actionable correctness or recovery issues and two experimental concerns. Three additional working-tree failures were reproduced in the follow-up review: discarded microbatches for dictionary updates, incorrect final-window gradient scaling, and stalled schedules in gradient dictionary mode. None establishes the cause of the older run's lower epoch-50 FID. The current training process was left running; its source, weights, optimizer, logging, and evaluation settings were not changed. A separate 50,000-sample evaluation was launched as requested.

The scope was the active pipeline under `outputs/church-consistent-rqvae-20260914`: the stage-1 sparse bottleneck and alternating dictionary update, stage-2 preparation and compact quantizer, accumulation and loss, transformer conditioning and cached inference, evaluation, and recovery. The follow-up also covered the working-tree LASER stage-1 optimization hooks and dictionary schedules. The active pipeline imports frozen snapshots. Working-tree fixes do not automatically apply to those snapshots. This is not a review of every experimental model elsewhere in the repository.

Source hashes and reproduction results are in [the audit directory](../outputs/code-audit-20260915/). Findings distinguish reproduced failure conditions from behavior observed on actual cached latents.

1. **P1: OMP can return catastrophic finite solutions for nearly dependent supports.**

   The [frozen OMP implementation](/workspace/Projects/laser/outputs/church-consistent-rqvae-20260914/stage1-source/src/models/dictionary_learner.py:496) clamps a small or negative Cholesky Schur complement to `1e-10`, then continues selecting atoms and solving. It neither rejects the dependent candidate nor validates the resulting least-squares residual. Clamping the pivot changes the system; it does not reliably stabilize the intended solve. The later `nan_to_num` calls cannot detect very large finite coefficients.

   A normalized six-dimensional, eight-atom dictionary with nearly collinear columns and sparsity four reproduces the failure. With seed 93 and perturbation scale 0.001, the routine returns a maximum coefficient of **22,030,592** and reconstruction SSE **43,887,037.7**. A float64 SVD least-squares solve on exactly the same selected supports gives SSE **31.1573**. Ten of 32 signals have increasing error as support depth grows. At perturbation 0.0001 the coefficients remain finite but the reconstruction SSE reaches approximately `1.2e44`.

   The [working-tree implementation](/workspace/Projects/laser/outputs/code-audit-20260915/pre-fix/src/models/dictionary_learner.py:819) has optional ridge and coherence controls, but their defaults are zero ridge and no coherence restriction; the same default-case failure reproduces there. The frozen stage-1 implementation does not have those controls.

   A robust repair should reject numerically dependent candidates or terminate with an explicit inactive mask, and use a stable QR/SVD fallback when needed. If ridge is enabled, it must define a consistent regularized objective. Validate reconstruction error and solve residuals instead of accepting any finite coefficient tensor.

   **Observed-run limit:** a separate check using the actual dictionary and 128 latent vectors from two validation examples was healthy: largest selected-matrix condition number **1.799**, maximum coefficient **14.569**, and no increasing prefix errors. The returned SSE and float64 least-squares SSE agreed within `1.4e-9`. The stress-case failure is real, but it has not been demonstrated as a cause of this run's performance.

2. **P1: the frozen alternating dictionary updater can abandon a collective on one rank.**

   After selecting active atoms using globally reduced counts, [this branch](/workspace/Projects/laser/outputs/church-consistent-rqvae-20260914/stage1-source/src/models/dictionary_learner.py:287) returns if none of those atoms appears in the local rank's support. Peers can still enter the subsequent statistics all-reduces. Missing local cached batches can also cause an earlier asymmetric return.

   A two-process CPU/Gloo reproduction uses two occurrences of atom 0 on rank 0 and one occurrence each of atoms 1 and 2 on rank 1, with minimum usage two. Only atom 0 is globally active. Rank 1 returns zero updates; rank 0 fails in a collective when its peer exits. With a live peer in other work this becomes a collective mismatch or timeout.

   **The working tree already handles this case:** its updater gathers batches, computes the update on rank 0, and broadcasts a result along a fixed collective sequence. Running the same reproduction there returns one update on both ranks. That implementation is not part of the frozen stage-1 snapshot. Stage 1 has finished, so this is a risk for reusing that driver, not evidence that the current stage-2 process is stalled.

3. **P1: full stage-2 recovery states are saved, but the driver cannot resume them.**

   The [checkpoint writer](/workspace/Projects/laser/outputs/church-consistent-rqvae-20260914/drivers/stage2.py:236) saves model, optimizer, scheduler, scaler, per-rank RNG, epoch, and batch offset. However, the [CLI and initialization](/workspace/Projects/laser/outputs/church-consistent-rqvae-20260914/drivers/stage2.py:82) have no resume option or state-restoration path. The driver requires a new output directory, initializes a fresh model, uses W&B `resume='never'`, and starts the epoch loop at zero. The supervisor also refuses an existing pipeline receipt.

   Consequently, an interruption can produce a valid `last.pt` and a “paused” status, but rerunning the launch command fails instead of continuing. Recovery currently requires another driver or manual implementation. Add an explicit restore path with sampler position, RNG, optimizer/scaler/scheduler state, and W&B continuation; verify it by comparing uninterrupted training with a save/restart trajectory. Saving the latest checkpoint only is the requested storage policy and is not itself a bug.

4. **P2: the shared coefficient-level fitter does not minimize final reconstruction error.**

   [The fitter](/workspace/Projects/laser/outputs/church-consistent-rqvae-20260914/stage2-source/src/adaptive_scaled_atom_rq.py:88) averages projections of the residual *before each depth's contribution*. Those values are appropriate one-step projection targets, but a shared coefficient influences multiple depths and may appear repeatedly in a reconstruction. The update ignores these interactions. It also records MSE before the update, applies the new levels unconditionally, and does not retain the best measured codebook.

   A one-dimensional example suffices. Let the atom be 1, levels be `[-1, +1]`, depth be four, and the target be 4. Four positive tokens reconstruct the target exactly. One fitting pass with the production prior weight of four changes the positive level to **1.75**. Greedy quantization then returns `[+1.75, +1.75, 0, 0]`, reconstructing **3.5** and increasing MSE from **0 to 0.25**. The prior initially has zero penalty, so this is not a necessary reconstruction/regularization tradeoff.

   [The working-tree normal-equation solver](../src/compact_rq_fitting.py) already accounts for repeated tokens and cross-atom interactions, but the active preparation driver calls `fit_atom_levels` instead. A replacement should solve the final fixed-code reconstruction objective, enforce valid levels, reassign codes, and accept or retain candidates based on an explicitly measured objective. Unused helper code alone does not fix the active pipeline.

   **Observed-run limit:** the production trace improves over its eight pre-update measurements. The counterexample establishes that improvement is not guaranteed; it does not establish that this particular fitting run regressed.

5. **P2: stage-1 gradient accumulation discards earlier microbatches from the dictionary update.**

   Each [training forward](../outputs/code-audit-20260915/pre-fix/src/models/dictionary_learner.py:1547) assigns a new dictionary to `_last_dictionary_update_batch`. The [manual optimizer path](../outputs/code-audit-20260915/pre-fix/src/models/laser.py:3497), as well as the automatic optimizer hook, consumes that cache only when the optimizer steps. If there are several forwards per optimizer step, the earlier signals, supports, and coefficients have already been overwritten. `dictionary_update_accumulation_steps` pools surviving caches across optimizer steps; it does not recover the missing microbatches.

   A two-atom identity dictionary and two one-signal microbatches reproduce this directly. The first signal selects atom 0 and the second selects atom 1. Two forwards followed by one dictionary update retain one signal and update only atom 1. A single forward containing the same two signals updates both atoms. The resulting dictionaries differ. This makes the dictionary's effective fitting batch smaller than the encoder/decoder batch and dependent on microbatch partitioning.

   Accumulate every relevant forward's detached fixed-code batch until the optimizer boundary, then apply the configured window across those complete steps. Handle discarded/failed optimizer steps explicitly. **Scope:** working-tree stage-1 paths with optimizer accumulation greater than one. This is separate from the active stage-2 accumulator, which passed the unequal-microbatch comparison.

6. **P2: the manual stage-1 optimizer underweights a final partial accumulation window.**

   [The step condition](../outputs/code-audit-20260915/pre-fix/src/models/laser.py:3429) correctly flushes the final batch, but [backward](../outputs/code-audit-20260915/pre-fix/src/models/laser.py:3485) always divides by the configured accumulation count. With three equal-size microbatches and accumulation two, the final optimizer step receives half of its intended mean gradient. Unequal microbatch sizes are also weighted equally rather than by their sample counts.

   An isolated harness executes the unmodified `_adversarial_training_step` method with a constant unit gradient, SGD, no clipping, and the discriminator gated off. Its optimizer-step gradients are `[1.0, 0.5]`, whereas averaging each actual window gives `[1.0, 1.0]`. Starting at weight 10 with learning rate one ends at **8.5 instead of 8.0**. This is a control-flow reproduction, not a complete Lightning training trajectory.

   Normalize by the actual examples in each accumulation window, including the final partial window. **Scope:** the working-tree manual adversarial path with accumulation enabled; the discriminator-first path rejects accumulation greater than one, and the active stage-2 helper already uses actual batch sizes.

7. **P2: delayed initialization and coefficient curricula never advance in gradient dictionary mode.**

   Both the [coefficient curriculum](../outputs/code-audit-20260915/pre-fix/src/models/dictionary_learner.py:371) and [delayed data initialization](../outputs/code-audit-20260915/pre-fix/src/models/dictionary_learner.py:1384) read `_dictionary_update_step`. However, [the post-step hook](../outputs/code-audit-20260915/pre-fix/src/models/dictionary_learner.py:473) returns immediately in `dictionary_update_mode='gradient'`, before incrementing that counter. No other path increments it. The constructor accepts these schedule options with gradient mode, which is its default.

   Six actual SGD steps change the dictionary, but the schedule counter remains **zero**. With initialization requested at step two and coefficient quantization starting at step two with a two-step warmup, initialization remains false and the quantization fraction remains **0 instead of 1**. The existing curriculum unit test manually writes the counter, so it does not exercise this integration failure.

   Advance a training-step counter independently of the dictionary update algorithm, or explicitly reject schedules unsupported by a given mode. **Scope:** working-tree gradient dictionary training with a positive delayed start or coefficient warmup; the completed alternating-mode stage-1 run did not use this combination.

The three follow-up reproductions and source hashes are saved in [accumulation-reproductions.json](../outputs/code-audit-20260915/accumulation-reproductions.json); run [accumulation_probe.py](../outputs/code-audit-20260915/accumulation_probe.py) with the CPU command shown below. A partial dictionary accumulation window being absent from checkpoints was also inspected: the code explicitly documents rebuilding a fresh window after resume. That deliberate behavior is not counted as an additional bug.

**Sampling deserves a controlled comparison, but token count alone is misleading.** The current expanded vocabulary is **32,769**, versus **16,384** in the original model. Both compared LASER runs use the same 32,769-token vocabulary and `temperature=1`, `top_k=1400`, `top_p=1`. The [evaluation function hard-codes these values](/workspace/Projects/laser/outputs/church-consistent-rqvae-20260914/drivers/stage2.py:58), so editing a generic sampling configuration will not change this evaluator.

I measured the epoch-132, step-8184 checkpoint on eight validation examples: 512 spatial positions per depth, using hard target prefixes. This was CPU float32 inference; production generation uses GPU float16 autocast. These are conditional probability diagnostics, not generated-sample FID measurements or a representative estimate over the entire dataset.

| Depth | Mean mass retained by top-k 1,400 | Mean mass retained by top-k 2,800 | Mean number of tokens to reach 95% mass |
|---|---:|---:|---:|
| 1 | 99.67% | 99.80% | 149 |
| 2 | 98.94% | 99.57% | 333 |
| 3 | 97.53% | 99.30% | 763 |
| 4 | 95.67% | 99.03% | 1,218 |

Although top-k 1,400 keeps only 4.27% of token identities, it keeps most conditional probability mass in this sample. Its effect differs by depth and context. Increasing k can restore useful diversity or admit an unreliable tail; a larger vocabulary does not determine which outcome wins. Generated-prefix diagnostics are still needed because teacher-forced and sampled contexts differ.

The [eight-setting screen](church-sampling-sweep-2026-09-15.md) has now completed on the fixed epoch-137 checkpoint. It compared the baseline with k=2,800 and 5,600, sampling temperatures 0.9 and 1.1, pure nucleus p=0.95 and 0.98, and depth-specific k=`[1400, 1400, 2800, 2800]`. Pure nucleus p=0.98 had the lowest 4,096-sample score; an independent repeat also favored it. These are diagnostic screening results. The completed confirmation compared the baseline and p=0.98 using **50,000 generated samples each against all 126,227 training images**, with seed 73000 + rank and the same checkpoint, tokenizer, reference, batch sizes, precision, and original preprocessing. The completed FID50k values are **12.2517 for the baseline** and **11.9058 for p=0.98**. Its [results](../outputs/church-sampling-sweep-20260915/fid50000-seed73000/results.json) and [launch record](../outputs/church-sampling-sweep-20260915/fid50000-launch.json) preserve the protocol.

The standalone evaluator now defaults to 50,000 generated samples and requires `--diagnostic` for another count. It verifies the reference receipt against the audited full training-set size and records both real and generated counts in local results and W&B. The existing training process still emits its historically labeled 4,096-sample monitor every five epochs and runs FID50k every 50 epochs. Those smaller monitoring scores are not reportable FID under the requested protocol.

For pure nucleus sampling, remove the top-k cap. The released sampler applies top-k first, renormalizes, and then applies top-p. Combining p=0.95 with k=1,400 therefore truncates 95% of the already restricted distribution, rather than retaining 95% of the original model distribution.

**The training-temperature calibration also matches only one property.** It selects temperature by the sampled/hard final latent-MSE ratio. This does not match token entropy or the amount of label smoothing across models. The saved calibration shows:

| Depth | Original RQ target entropy, temperature 0.5 | LASER target entropy, temperature 0.125 |
|---|---:|---:|
| 1 | 0.177 nats | 0.071 nats |
| 2 | 0.865 nats | 0.198 nats |
| 3 | 1.275 nats | 0.395 nats |
| 4 | 1.744 nats | 0.663 nats |

LASER's targets are considerably sharper under this criterion. That is a plausible regularization difference to test, not a proven explanation of overfitting or the difference between the two LASER runs. Training target temperature and autoregressive sampling temperature are separate parameters; changing one does not calibrate the other.

**Checks that passed:** 15 existing tests ran against the active stage-2 snapshot, including expanded-codebook equivalence, stochastic/soft target equivalence, loss values and gradients, causal conditioning, and accumulation. An additional cached/full transformer comparison had maximum logit difference `7.15e-7`. A check of the actual scaled-token training accumulator with unequal microbatches matched the full-batch optimizer update exactly. The actual-data OMP check above also passed. The reviewed data-protocol records and code do not expose a new resize/crop mismatch. The subsequent sampling experiment generated its own diagnostic samples using the verified full-training-set reference; the real statistics did not need rebuilding.

These checks cover specific mechanisms, not an end-to-end guarantee. The findings should be addressed in versioned code for a subsequent run; changing files inside the current protected snapshot would invalidate its provenance.

To reproduce the numerical counterexamples on CPU:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' /tmp/laser-church-venv/bin/python outputs/code-audit-20260915/numerical_probe.py
```

Additional evidence: [numerical failures](../outputs/code-audit-20260915/numerical-reproductions.json), [actual-data OMP check](../outputs/code-audit-20260915/actual-dictionary-omp-check.json), [distributed reproduction script](../outputs/code-audit-20260915/distributed_probe.py), [training checks](../outputs/code-audit-20260915/training-checks.json), [test results](../outputs/code-audit-20260915/snapshot-tests.xml), [sampling diagnostics](../outputs/code-audit-20260915/sampling-diagnostics.json), and [target-temperature comparison](../outputs/code-audit-20260915/target-temperature-comparison.json).
