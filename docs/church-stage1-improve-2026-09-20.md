# Improving Church stage one, September 20, 2026

The official Church RQ-VAE checkpoint was downloaded, checksum-verified, loaded
strictly, and evaluated alongside both LASER Church tokenizers. Its bundled
configuration differs from the example YAML in the upstream repository.

**Completed September 21:** both three-epoch trials finished and the LR 1e-5
epoch-three checkpoint was selected. It scores full-population rFID 2.6393 and
matched-subset rFID 6.0640. The full compound cache and selected best/last
checkpoints were uploaded and their remote manifests verified. The previous
compound run also resumed and completed all 90 epochs using its original tokenizer.

## Released checkpoint and matched measurements

[Official checkpoint listing](https://github.com/kakaobrain/rq-vae-transformer#pretrained-checkpoints)
links to the [Church archive](https://twg.kakaocdn.net/brainrepo/models/RQVAE/deeb3e0ac6e09923754e3e594ede7b01/church.tar.gz).
The whole archive MD5 is `deeb3e0ac6e09923754e3e594ede7b01`.
Its stage-one model SHA256 is
`ba008ec2e192a6d4084a8fd511927a789c68a8459d6e8bfc22122ae02887b800`.
The bundled configuration specifies three epochs, constant generator and
discriminator LR 4e-6, Adam betas (0.5, 0.9), no weight decay, and global batch
128 (4 × 32). The GitHub example YAML instead specifies one epoch at 4e-5.
The checkpoint itself contains model weights, not optimizer step counters.

All rows below reconstruct the same first 4096 training images, using the same
4096 real-image reference, original RQ-VAE Inception features, RGB resize and
center crop to 256, FP32, and TF32 disabled. LPIPS and PSNR use all 300 separate
Church validation images. Lower rFID and LPIPS are better; higher PSNR is better.

| Tokenizer | Matched 4096-image rFID | Held-out LPIPS | Held-out PSNR |
|---|---:|---:|---:|
| Released Church RQ-VAE | 8.1311 | 0.25490 | 18.9718 dB |
| Current one-epoch LASER | 8.9197 | 0.26260 | 19.8012 dB |
| Earlier three-epoch LASER | **6.5315** | **0.24665** | 18.7783 dB |
| New three-epoch LASER, LR 4e-6 | 6.6606 | **0.24471** | 19.1784 dB |
| New three-epoch LASER, LR 1e-5, selected | **6.0640** | 0.24729 | 19.1091 dB |

These are reconstruction screening scores, not generated-image FID or
full-population rFID. The earlier LASER checkpoint improves perceptual metrics
but has worse pixel PSNR. The comparison does not isolate the effect of epoch
count: that earlier run also used gradient dictionary updates and a different
training driver. The released RQ-VAE has a discrete residual codebook and is a
reference model, not a replacement for the compound-token LASER bottleneck.

## Controlled full-model trials

Both trials initialize from the same ImageNet rFID 4.21 LASER checkpoint, reset
finetuning optimizer states, and run three complete epochs: **2961 generator
and 2961 discriminator updates each**. Four H200s per trial, batch 32 per GPU,
preserve global batch 128 and the discriminator's 32-image local BatchNorm scope.
The two independent trials use all eight H200s concurrently.

- [LR 4e-6](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-stage1-full-ft3-lr4e6-h200-20260920): duration, learning rates, and batch size from the released checkpoint config.
- [LR 1e-5](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-stage1-full-ft3-lr1e5-h200-20260920): same experiment with a larger fixed learning rate.

Encoder, decoder, both projections, sparse bottleneck, and discriminator all
update. The sparse dictionary uses the audited alternating residual update;
it updates outside Adam. All component parameter hashes are compared each epoch.
The backbone/GAN objective retains MSE, LPIPS weight 1, latent weight 0.25,
adaptive adversarial weight 0.75, hinge discriminator loss, and 1:1 G/D steps.
Precision remains FP32 with TF32 disabled. Activation checkpointing is disabled
to use H200 memory and avoid recomputation; this changes execution, not the loss.

Both four-GPU preflights passed actual training, full-component update checks,
evaluation, preview generation, and checkpoint save/reload. Each production
epoch evaluates reconstruction FID on all 126227 unique training images and
LPIPS/PSNR on all 300 validation images. Reconstruction previews are recorded
every 500 optimizer updates and recovery checkpoints every 250 updates.
Best-rFID and last full checkpoints are uploaded online from immutable snapshots;
the uploader verifies COMMITTED status, file sizes, and manifest MD5 values.

## Selection, token cache, and existing stage two

After both trials, their best full-population-rFID checkpoints receive the same
4096-image screen as the fixed baselines. The selector chooses the lowest
matched rFID among LASER candidates whose held-out LPIPS is no worse than the
current tokenizer. This can select the earlier three-epoch checkpoint if it
beats both new trials; no improvement from the new optimization is assumed.

The winner receives a complete prebuilt compound cache for all 126227 images:
FP32 encoder/OMP, int16 atom indices, FP16 normalized coefficient storage, four
depth-specific scales fitted to the 99.9th percentile, and 2048 coefficient bins.
Checkpoint and cache hashes, exact image coverage, fresh-encoding agreement,
and a decoded preview are recorded before publication.

The existing geometry-free 90-epoch compound run was checkpointed at step 5412,
epoch 66, with optimizer, scheduler, and all eight ranks' RNG states preserved.
Its supervisor resumes automatically after stage-one selection/cache building,
or if the stage-one pipeline fails. Its tokenizer is not changed mid-training.

## Locations and live status

Experiment root: `outputs/church-stage1-improve-20260920`.

[Online comparison and publication run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-laser-stage1-selection-20260920)
already contains the verified baseline artifact, released configuration,
provenance, and reconstruction previews. Its detached CPU publisher waits for
selection and cache validation, then uploads selected best/last/tokenizer states
and the complete cache, verifying each remote manifest entry. The publisher's
source hash is recorded separately in `publication-source.json` so the running
training source manifest remains unchanged.

- `reference/released-verified.json`: archive/config/model verification.
- `baselines/complete.json`: all fixed-baseline measurements.
- `source-manifest.json`: pinned training, evaluation, selection, and cache source.
- `status.json`: live pipeline phase.
- `trials/{lr4e6,lr1e5}/status.json`: current step and throughput.
- `trials/{lr4e6,lr1e5}/complete.json`: final metrics and verified checkpoint artifact, once complete.
- `selected/selection.json`: eventual comparison and selected checkpoint.
- `selected/cache/complete.json`: eventual cache integrity and encoding checks.
- `publication.json`: eventual verified W&B publication of the selected checkpoint and cache.
- `publication-status.json`: live publisher phase or explicit failure.

## Completed results

Both trials completed all 2961 generator and 2961 discriminator updates. All
six component groups changed in every epoch: encoder, decoder, both projections,
sparse bottleneck, and discriminator. Full-population reconstruction FID on
126227 unique training images improved each epoch:

| Trial | Epoch 1 | Epoch 2 | Epoch 3, best and last |
|---|---:|---:|---:|
| LR 1e-5 | 4.3571 | 3.1969 | **2.6393** |
| LR 4e-6 | 4.1733 | 3.5300 | **3.1537** |

The selected LR 1e-5 model wins matched rFID. LR 4e-6 has slightly better
held-out LPIPS (0.24471 versus 0.24729); selection does not imply dominance on
every metric. The selected model's matched rFID and held-out LPIPS both improve
on the previous one-epoch tokenizer and the released RQ-VAE reference.

The selected tokenizer SHA256 is
`762c51a10267ed6fa55709ff0d6cf997940d21056b16c7a0a0399b3ebf93868d`.
The cache covers exactly 126227 images and has SHA256
`c99a079ce660013925e9a8514a12e86e070dd29c0f7c4e10cbb4e48e304f8d21`.
Fresh encoding agrees on every probed atom index; maximum normalized coefficient
error is 0.00097609 from FP16 storage. Cache construction and validation took
73.9 seconds. Both trial checkpoint artifacts, the selected model artifact, and
the cache artifact are COMMITTED with verified file sizes and MD5 values.

The prior geometry-free compound run finished epoch 90, step 7380, with best
50k generated-image FID **17.2044 at epoch 50**. Its full best/last checkpoints
were also uploaded and verified. It used the original one-epoch tokenizer
throughout; no new stage-two run using the improved tokenizer has been launched.
All eight GPUs were idle at the September 21 status check.
