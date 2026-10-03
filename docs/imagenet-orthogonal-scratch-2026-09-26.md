# ImageNet prior with prefix-stable orthogonal coefficients

The previous joint decoder used the final dictionary coefficients from four-step OMP. Those coefficients define a valid autoregressive sequence, but their partial sums do not reproduce the encoder's intermediate OMP projections: adding an atom refits earlier dictionary coefficients. This experiment fixes that representation mismatch while retaining the frozen ImageNet rFID4.21 stage-1 tokenizer, its atom selection order, and its continuous final reconstructed latent.

For each ordered support, sequential Gram–Schmidt constructs unit directions `q_d` using atoms through depth `d` only. The model predicts the pair `(a_d, gamma_d)`, where `gamma_d = q_d^T z`. Cache conversion uses the equivalent triangular transform of the final least-squares coefficients. Prefix contributions are `gamma_d q_d`; later atoms never alter a completed contribution. OMP and least-squares refitting remain in offline encoding, where they select the support. Neither refitting nor OMP occurs during generation.

Both the spatial/depth backbone and the two-layer global coefficient-history decoder use these orthogonal contributions. The coefficient head conditions on the current selected direction, completed pair history, and the exact local prefix sum. Sampling constructs the previous pair from its own complete support prefix, including across spatial boundaries. Basis calculations use FP32 reductions outside autocast, avoiding both BF16 and TF32 matrix contractions. Generation masks previously selected atoms at each site.

The two augmented pixel views for all 1,281,167 ImageNet training images are retained. The converted cache has a distinct format and content identity; the trainer rejects using orthogonal caches in a dictionary-coordinate model. Raw targets stay FP32 and unclipped. New per-depth scales map the maximum absolute coefficient plus a 0.5 physical-unit noise margin to the bin range. The 2,048-bin posterior keeps physical temperature 0.03125 (SD 0.125), rather than reusing normalized noise.

Validation on 128 real image views found:

- Full-support orthogonal coefficients versus each separately refitted OMP prefix: maximum difference 3.81e-6; prefix reconstruction difference 7.15e-7.
- Direct projection of the encoder latent onto each direction: maximum coefficient difference 1.14e-5. In comparison, the old first dictionary coefficient changed by up to 3.94 across OMP refits.
- Continuous cache reconstruction difference: maximum 7.15e-7. A separate 65,536-site cache conversion audit also passed.
- Quantized decoder reconstruction MSE ratio to the original representation: 1.0000020; decoder-to-decoder PSNR 67.60 dB. This is a paired reconstruction probe, not a new full-dataset reconstruction FID measurement.
- Discrete teacher SD 0.1249999–0.1250002; added latent energy 0.0764% on 2,048 images.
- 72 focused tests pass, including future-token perturbations, incomplete-buffer cached/dense agreement across sites, sampling, FP32 basis under autocast, prior history behavior, scheduler recovery, and compiled short attention.

The superseded run is preserved at step 26,312 with all optimizer states and four rank RNG streams. Its best measured 50k-sample FID was 30.29766 at epoch 40; it is the comparison baseline. The new stage-2 model starts from scratch, with no inherited model weights, optimizer, scheduler or RNG. Stage 1 remains frozen, so this stage-2 job has no discriminator optimizer.

Production configuration: 4 H200s, 252 images per GPU, two accumulation steps, global batch 2,016; equal atom/coefficient loss weights; two coefficient-history layers at width 512; BF16, compiled transformer blocks and short attention. Peak LR 4.921875e-4, two warmup epochs, cosine decay over 100 epochs to 1e-5. Sampling settings and the original RQ ImageNet metric reference remain as in the comparison run. A separate 12-update fresh preflight is discarded before production initialization.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-orthogonal-scratch-20260926

Local runtime, conversion/audit scripts and logs: `/mnt/laser-imagenet-orthogonal-scratch-20260926`.
Persistent cache, configuration, evidence archive and checkpoints: `outputs/imagenet-rfid421-orthogonal-scratch-20260926`.

This change fixes prefix semantics. It does not establish improved generation FID or superiority over RQTransformer; the new run must be evaluated.

Checkpoint verification uses an independent pinned snapshot. The initial full-file hash of the rotating step-200 checkpoint was interrupted by normal retirement at step 400. A filesystem probe confirmed that this mount returns EOF after an open file is unlinked. The replacement verification pins the local immutable serialization, copies it to a persistent path outside rotation, counts every byte read, and compares full SHA256 values. Training and the step-500 sampling preview continued successfully throughout.

Production verification (2026-09-26T18:13:44.403194+00:00): passed step 840; recent steady throughput median 1640.8 images/s. Checkpoint step 400 has all 870 model/Adam states finite, matching optimizer/scheduler counters, all four RNG streams, strict model reload and exact scheduler restoration. Its complete persistent payload matches the local SHA256. The new cache artifact is committed in W&B and online configuration confirms scratch initialization.
