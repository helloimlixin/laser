The user authorized testing the sparse-code audit findings one at a time.
The serial experiment driver is running in
`outputs/church-sparse-fixes-20260921/`, resuming the original
[W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-stochastic-rawcoeff90-h200x5-20260921).
These are controlled interventions, not established explanations of the FID
regression. A setting is promoted only if its measured FID50k improves the
best retained score.

The prior LR-only continuation was stopped after preserving its complete
epoch-71 / step-9301 checkpoint at
`/tmp/laser-sparse-fixes-20260921/control-archive/last.pt`. The original best
remained epoch 60, FID50k **10.68846321**. `handoff.json` records the stopped
processes, archived full state, and last committed control artifact. The
original persistent epoch-60 source is untouched.

The priority order and explicit interventions are:

| Order | Intervention | Comparison |
|---|---|---|
| 1 | Reduce coefficient target temperature from 0.125 to 0.03125 | Same physical bins, sparse support bank, model, optimizer moments, LR schedule, and sampler |
| 2 | Integrate Gaussian probability over nonuniform coefficient cells | Keep the temperature selected after trial 1; change only target measure |
| 3 | Depth-specific atom cutoffs `[250,250,500,1000]` and `[250,500,1000,2000]` | Evaluate both on the same selected frozen checkpoint; retain the best sampler |
| 4 | Fresh stochastic OMP support draws on every training visit | Same native encoder latents, atom temperature 0.0625, physical coefficient units, and current target/sampling policy |
| 5 | Continue the best selected policy through epoch 90 | Full model/optimizer/RNG resume with adaptive LR and retained best-FID state |

Each training intervention begins with twelve isolated real GPU preflight
updates and then a six-epoch pilot. Each pilot starts from the best complete
checkpoint available at that stage. A pilot that fails to improve that
checkpoint's FID does not carry its weights or policy into the next trial.
The two sampling candidates share the same model checkpoint, avoiding a
moving model baseline. Their sampling settings are saved with any selected
full checkpoint. The final continuation uses whichever combination actually
won. Six epochs is a screening horizon, not proof that a rejected change
could never help with longer training.

Every completed training epoch and every sampling candidate uses the same
50,000-image FID reference and fixed five-rank RNG protocol. Training uses five
H200s, batch 192 per GPU, global batch 960, and no accumulation. New training
phases restart the same 30-epoch cosine envelope at 3e-5, with floor 1e-6,
FID patience 3, absolute improvement threshold 0.02, reduction factor 0.5,
and one evaluation of cooldown. This matches the LR-only control's launch
policy. FID is the comparison metric; raw loss and coefficient cross-entropy
cannot be compared directly across temperatures because target entropy changes.

The probability policies and depth sampler are implemented in
[src/training/sparse_code_policies.py](../src/training/sparse_code_policies.py).
`centers` reproduces the old distance-softmax probabilities exactly. `cells`
uses Gaussian CDF differences at nearest-neighbor cell boundaries, including
infinite outer boundaries. Both input coefficient sampling and coefficient
soft supervision use the same chosen measure. Changing the target measure
does not change the coefficient vocabulary or token meanings.

Fresh OMP uses the verified native FP32 encoder cache. Its SHA256 matches the
same source used to build the existing support bank. Row alignment was checked
by verifying the existing cached coefficients' least-squares normal equations
against those native latents. Fresh supports pass the same test and contain
no repeated atom within a site. The GPU implementation disables TF32 for OMP,
bounds temporary correlation matrices with 32-image chunks, and uses the
checkpointed rank RNG. It preserves the original stochastic OMP distribution;
it does not uniformly deduplicate the bank and thereby change empirical weights.

Eight focused sparse-policy/LR tests pass, including exact original-probability
and RNG equivalence, removal of nonuniform-center bias, Gaussian tail handling,
variance scaling with temperature, preservation of coefficient sampling when
atom cutoffs change, and scheduler resume consistency. The frozen runtime's
11 existing tests and its six autoregressive tests with the active attention
implementation also passed during the audit. Every new training phase must
pass its GPU preflight before production starts.

Working full checkpoints and immutable upload staging remain on local `/tmp`
to avoid the earlier network-storage write failure. Each full latest/best
artifact is verified against the committed W&B file manifest. Uploads have a
bounded queue and can lag training; a phase cannot complete until its final
full latest/best upload is verified. The original epoch-60 source remains
persistent under `/workspace`. New phase provenance and policy metadata are
published online separately.

For progress, read:

- `status.json`: current phase, supervisor PID, and worker PID.
- `plan.json`: ordered trial plan.
- `selected.json`: currently selected complete checkpoint and policy, once the first trial finishes.
- `<phase>/train/evaluations.jsonl`: every FID and LR decision.
- `<phase>/train/checkpoint-upload.json`: most recent committed full checkpoint artifact.
- `<phase>/decision.json`: phase result and promotion/rejection decision.
- `complete.json`: written only after the sequential experiments and final continuation finish.

The first intervention's twelve-update preflight passed at step 7872.
Production restarted independently from epoch 60 / step 7860, is online,
and uses target temperature 0.03125. Logged target entropy is approximately
3.94 nats and steady training throughput approximately 3030 images/second.
