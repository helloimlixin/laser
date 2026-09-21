# Training correctness fixes — 2026-09-15

The confirmed audit failures have been addressed in the working code. The active
training process, its frozen dependency snapshots, existing tokenizer, coefficient
table, and token cache retain their original versions. These changes apply to
subsequent training/preparation runs; stage-2 recovery uses the new driver below.

| Finding | Repair and verification |
|---|---|
| Unstable OMP solves on nearly dependent supports | Relative pivot checks and explicit reconstruction-objective checks select a double-precision SVD fallback. Ridge uses the same regularized objective in both solvers. A rejected extension retains the previous coefficients and an inactive zero slot. CPU and CUDA stress cases pass. |
| Frozen updater's asymmetric collective return | The working tree already used a fixed gather/update/broadcast sequence. Added a two-rank regression where a globally active atom is absent on one rank, plus a check that inconsistent optimizer-step outcomes fail on both ranks. |
| Earlier microbatches discarded by dictionary fitting | Each training forward queues detached signals and codes. All microbatches are pooled at the optimizer boundary, then accumulated over complete optimizer steps. Zero coefficient slots do not count as atom usage. |
| Incorrect partial-window gradients | The manual training loop weights microbatch losses by example count and normalizes the final gradients using the actual window size. Tests exercise unequal batch sizes and incomplete final windows through Lightning. |
| Stalled gradient-mode schedules | The schedule counter advances after successful optimizer steps in both dictionary modes. Skipped steps discard pending fixed-code statistics and do not advance the counter. |
| Incorrect shared coefficient-level objective | The fitter solves sparse normal equations for the complete fixed-code reconstruction, including repeated tokens and cross-token interactions. It reassigns codes, measures the candidate, backtracks, and restores the old table unless both reconstruction SSE and the regularized objective do not increase. |
| Missing stage-2 recovery | A new driver restores model, optimizer, scheduler, scaler, counters, per-rank RNG, and deterministic cached-data sampler position. It validates the training configuration, token/cache identity, and world/batch sizes, resumes the W&B run, and retains only `last.pt`. |

The manual optimizer repair also fixes AMP clipping: dictionary gradient projection
and gradient clipping now run in Lightning's hook after unscaling. Actual optimizer
steps are observed so an AMP skip does not update the dictionary or advance its
schedule. When generator and discriminator gradients share a backward pass, the
pending discriminator gradients are adjusted if the first optimizer changes the
shared scaler's scale. CPU AMP tests cover growth, skipped generator updates, and
clipping for both optimizers.

In the OMP stress case with perturbation 0.001, reconstruction SSE decreased from
the audited failure of approximately **43.9 million** to **28.204688** on both CPU
and CUDA. A stable least-squares solve on the repaired implementation's selected
support gives **28.204688**. The support can change after correcting residual
correlations, so this is not a comparison on the original failing support.
There are no increasing prefix errors in the three tested conditioning regimes.
The repeated-token coefficient-fitting counterexample now preserves its exact
reconstruction and zero error.

**Validation:** 127 targeted tests pass. They cover the sparse bottleneck,
distributed dictionary updates, real Lightning optimization hooks, coefficient
fitting, token/soft-target equivalence, causal cached sampling, and checkpoint
recovery. Resume tests compare uninterrupted training against restarts within an
epoch, after its final batch, and at the next epoch boundary, for each rank's
stream. Losses, parameters, optimizer state, scheduler, scaler, and RNG states
match exactly. Separate CUDA probes exercise OMP and coefficient fitting.

The real checkpoint's metadata and training configuration pass the new recovery
checks. Its full 386,882,561-parameter model and all 460 optimizer-state entries
also reload successfully on CPU, with exact model, scheduler, and scaler-state
comparisons. All 93 frozen dependency hashes and the original live driver hash remain
unchanged. Production training has not been restarted with the new driver.
These tests do not establish that retraining with the repairs will improve FID.

Evidence: [test results](../outputs/code-audit-20260915/fix-tests.xml),
[CPU/CUDA numerical results](../outputs/code-audit-20260915/fix-numerical-results.json),
[actual checkpoint recovery checks](../outputs/code-audit-20260915/fix-resume-preflight.json),
and [the pre-fix working sources](../outputs/code-audit-20260915/pre-fix/src/).
The method documents were updated to describe the corrected
[OMP/dictionary mechanism](batch-omp-online-dictionary-learning.md) and
[coefficient-table fitting](sparse-codes-for-autoregressive-generation.md).

To validate a full checkpoint without starting training:

```bash
/tmp/laser-church-venv/bin/python scripts/tools/train_consistent_rq_stage2.py \
  --cache outputs/church-consistent-rqvae-20260914/preparation/cache \
  --calibration outputs/church-consistent-rqvae-20260914/preparation/temperature-calibration.json \
  --output outputs/church-consistent-rqvae-20260914/stage2 \
  --run-id church-laser-consistent-rq32k-20260914 \
  --resume outputs/church-consistent-rqvae-20260914/stage2/last.pt \
  --validate-resume
```

When the original training process has exited, continue its full checkpoint with
the same W&B authentication and run ID:

```bash
/tmp/laser-church-venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=2 scripts/tools/train_consistent_rq_stage2.py \
  --cache outputs/church-consistent-rqvae-20260914/preparation/cache \
  --calibration outputs/church-consistent-rqvae-20260914/preparation/temperature-calibration.json \
  --output outputs/church-consistent-rqvae-20260914/stage2 \
  --run-id church-laser-consistent-rq32k-20260914 \
  --resume outputs/church-consistent-rqvae-20260914/stage2/last.pt
```

The recovery driver rejects an output/checkpoint still owned by the original live
process and takes an exclusive writer lock. Resume provenance is written under
`stage2/resumes/`. Batch size is inferred from the checkpoint unless explicitly
supplied; changing it during resume is rejected. Original end-of-epoch checkpoints
do not identify whether evaluation finished, so their boundary is re-evaluated;
new checkpoints record that phase explicitly. Evaluation forks the training RNG.
The iterator recovery assumes a deterministic latent cache and does not claim
to restore random data augmentation in arbitrary loaders.

The new driver computes **FID50k against all 126,227 training images**, at epoch
one and every 50 epochs by default (`--fid-every` controls the cadence). It keeps
the original resize/crop and Inception protocol. Smaller historical FID values
are not reused as the best FID50k. Sampling is configurable with
`--sampling-temperature`, `--top-k` (zero disables the cap), and `--top-p`.

The separate sampling confirmation on the unmodified epoch-137 checkpoint is
complete: **12.2517** for temperature 1/top-k 1400, and **11.9058** for temperature
1/top-p 0.98 with no top-k cap. Each setting used exactly 50,000 generated samples
against the entire training set. This is one 50k seed per setting; see the
[sampling report](church-sampling-sweep-2026-09-15.md) and
[W&B run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-sampler-ep137-fid50k-20260915).
