# Stochastic compound-pair experiment

This experiment changes the fixed OMP atom supervision in the previous
Lloyd–Max run into stochastic, valid atom/coefficient trajectories. It retains
the selected three-epoch stage-one checkpoint and starts a new 90-epoch
transformer from scratch.

W&B: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-ft3best-stochastic-compound90-h200x8-20260921

The previous model already generated both atoms and coefficients
autoregressively. Its *training cache* fixed one atom support at each image
location. Changing that supervision is an experiment to improve learning and
generalization, not a correction to a previously non-autoregressive sampler.

## Factorization and supervision

For each event, the model represents

`p(atom_t, coefficient_t | past_pairs)`
`= p(atom_t | past_pairs) * p(coefficient_t | past_pairs, atom_t)`.

Both predicted components become input to subsequent events. Training uses
teacher forcing on sampled valid trajectories; it does not replace the history
with arbitrary model-generated atoms while retaining incompatible targets.

At each OMP selection step, the cache builder samples an unused atom with
probability proportional to `exp(residual_correlation**2 / 0.0625)`. Correlations
and least-squares solves use physical FP32 latent coordinates with TF32 disabled.
The builder refits the coefficients for the complete selected support. The
stage-one encoder, decoder, dictionary, and projections remain frozen.

Each spatial location has sixteen prebuilt four-pair alternatives. On every
training visit, one alternative is drawn independently at each of the 8×8
locations, retaining all four atoms and their matching coefficients together.
CUDA RNG drives this choice and is included in the existing per-rank checkpoint
state. DataLoader workers make no random target selections.

This is a finite Monte Carlo bank, not the original RQ-VAE quantizer or unlimited
online support resampling. Sampling hard atom labels from the bank gives a
Monte Carlo categorical objective over those alternatives. The coefficient
soft targets and stochastic coefficient inputs retain temperature 0.125 and
the same 2048 Lloyd–Max centers as the control.

## Calibration and cache checks

Five temperatures (0, 0.0625, 0.125, 0.25, 0.5) were evaluated on 64 sampled
training images and 64 validation images. The selected value was the largest
candidate satisfying at most 5% extra latent reconstruction MSE, at most 0.005
extra LPIPS, and at least 5% changed supports on the validation subset.

At 0.0625, validation LPIPS was 0.241409 versus greedy OMP's 0.241199. Latent
reconstruction MSE increased 1.67%, and 38.28% of spatial locations changed
support. This verifies representation fidelity; it is not evidence of improved
generated-image FID.

The full 126227-image, sixteen-variant cache took 270 seconds to build using
eight H200s. It occupies 3,102,430,651 bytes. Across the full bank:

- 85.37% of spatial locations have multiple distinct supports.
- 40.18% of sampled supports differ from the original greedy support.
- 16.36% of individual atom positions differ.
- Mean native latent MSE is 0.00948197; stochastic MSE is 0.00966417 (+1.92%).
- Added Lloyd–Max rounding MSE is 4.76e-7.
- Coefficients outside the finite bin-center range occur at fraction 3.67e-7.

Identities:

- Stage one: `762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`
- Cache: `dd7724fed7bb8cf41b1aa392acb0331d21b9a44f802547ffc46d9bacfbb57d14`
- Bank specification: `a7a62b6b0ed9e31dcff0f68ae030dbf9b4422d5e632c53e24545fca42b41cab8`
- Coefficient centers: `58e0d9fba276b813b8283d3104b46041a9c2ea1d32c21ce00927940581d9e618`

Cache selection adds no encoder or OMP computation to training. The eight-GPU
driver transfers bank rows, selects a trajectory on GPU, and then uses the
existing compound loss and transformer.

## Training and recovery

The architecture remains the 404,738,048-parameter full-pair transformer, with
two-layer atom-conditioned coefficient refinement and depth-specific
coefficient heads. The objective remains
`(1.5 * atom CE + coefficient soft CE) / 2.5`; geometry, regression, and CRPS
weights remain zero. This keeps the comparison focused on target variability.

The recipe uses eight H200s, 192 images per GPU, global batch 1536, fused AdamW
with peak LR 5e-4, betas (0.9,0.95), weight decay 1e-4, gradient clipping 1,
and a 90-epoch cosine schedule. The transformer uses BF16 training with the
existing FP32 short-attention path. There are 82 updates per epoch and 7380
planned updates. Preflight weights are isolated from production initialization.

Previews are scheduled every 500 optimizer steps. FID uses 50000 generated
images against the full 126227-image training reference at epoch 1 and every
10 epochs. Sampling remains atom temperature 1/top-k 250/top-p 1 and coefficient
temperature 1/top-p 1. Full latest and best-FID checkpoints are uploaded online
through the verified asynchronous uploader.

The integer comparison was checkpoint-paused at step 4341, epoch 70.016129,
with optimizer, scheduler, scaler, and all eight RNG states. Its immutable
resume anchor has SHA-256
`144f38e56e4df4c118b53660048afdebc717aee65332b4e1ba1e7ddd06561029`.
The new supervisor queues that same run for continuation after this trial.

Eleven focused tests pass in both the working tree and frozen runtime; the
working tree's extended objective suite passes 29 tests total. They
check least-squares consistency of sampled supports, the atom sampling
distribution, no duplicate support atoms, preservation of paired trajectories,
RNG replay, future-token independence, conditioning coefficients on the current
atom, and cached-versus-parallel pair predictions. The full GPU preflight also
records per-rank evidence that repeated visits draw different atom supports.

The eight-GPU preflight completed 64 updates, generated two previews, and saved
the full model, optimizer, scheduler, and eight rank RNG states. Stable training
throughput was 4956–5026 images/sec. Fresh production training then advanced
past step 180 at approximately 5016 images/sec. On all eight production ranks,
repeated draws from the same input banks changed 47.4–49.0% of spatial supports.
The cache and provenance artifacts were verified committed online.

The first production FID50k completed at epoch 1 (169.8931), and the scheduled
500-step preview was generated. Both full latest and best-FID files were verified
online in `church-laser-ft3best-stochastic-compound90-h200x8-20260921-selected-checkpoints:v0`.
The integer comparison's paused step-4341 checkpoint was also verified online.
These are launch and recovery checks; improved generation quality remains
unestablished while the experiment trains.

All launch, cache, calibration, preflight, online-upload, and source-manifest
receipts are under `outputs/church-compound-stochastic-pairs90-20260921/`.
