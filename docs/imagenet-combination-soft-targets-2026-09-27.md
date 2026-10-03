# ImageNet: stochastic full-combination targets, 2026-09-27

The new teacher scores the complete sparse reconstruction, including jointly
refitted coefficients and cross-terms between dictionary atoms. It continues
`imagenet-rfid421-joint-best-8h200-20260926` from update 55,895, epoch
88 + 15/635. All 870 model tensors, 870 Adam states, eight RNG streams, the
data cursor and original learning-rate schedule are retained. The old run's
best measured FID is 22.1948 at epoch 85. No FID improvement from the new teacher
has been established.

Candidate run: https://wandb.ai/helloimlixin-rutgers/laser/runs/imagenet-rfid421-combination-soft-20260927

For each latent vector, deterministic greedy OMP supplies an anchor support.
For each of its four positions, every dictionary atom is scored as a replacement
while the other three atoms remain active. The score is the reconstruction error
after refitting the entire support, not atom similarity alone. The two best
distinct replacements at each position and the anchor form nine candidates.
The candidate set is deterministic for a given image; a fresh support is sampled
on each training visit. Its mass is proportional to
`exp(-||z - D_S c*_S||² / 0.125)`.

This is a finite neighborhood approximation. It does not enumerate all
16,384-choose-4 supports or all 2,048^4 coefficient combinations. The nearest
replacement search spans all dictionary atoms, but the final teacher retains
nine complete supports. Joint coefficient centers are least-squares solutions.
Coefficient covariance follows `(temperature/2) * inverse(D_S.T @ D_S)`.
Sequential Gaussian conditionals are normalized over the actual finite bins;
this approximates the continuous quadratic-distance law and is not an exact
discrete global Gibbs distribution.

The coefficient temperature is **0.00002091737317762316**, in physical units.
It was fitted on 128 calibration images to 80% of the reference's total target
entropy. This is substantially smaller than the previous 0.03125 coefficient
kernel. The full teacher also samples supports, so its overall perturbation is
larger than the previous deterministic-support teacher, while remaining below
the original RQ reference in the measurements below.

Both heads receive exact prefix-conditional soft targets for the specified
nine-support law. The support posterior incorporates preceding sampled atoms
and coefficients. The coefficient target mixes all compatible supports and
accounts for their correlated coefficient conditionals. No target conditions
on its own coefficient sample or a future emitted pair. The existing equal-head
loss scale is retained to avoid an unrequested optimizer gradient-scale change.

The retained model emits atom/coefficient pairs in an interleaved order.
Earlier final-refit coefficients reveal much of the remaining support identity.
Consequently the **conditional atom-label entropy is very small**, although the
prior over complete supports has 0.778 nats on calibration images and support
sampling changes the anchor support about 54% of the time there. Much of this
support uncertainty appears in the first coefficient's conditional distribution.
Per-depth entropy is not aligned with the original RQ factorization; compare the
sum over a complete four-pair code, while also inspecting the individual heads.

The reference is the released ImageNet tokenizer accompanying the original
1.4B RQ-Transformer. Checkpoint SHA256:
`e8eb312e6c0dbd8e490ca95f69e3c3b417162625ff9320bbfff791cbf832b694`.
Its distance and stochastic-code functions have identical ASTs to the downloaded
official implementation. Reference temperature is 0.5. Both encoders use the
same saved fresh ImageNet crop/flip pixels. The probe contains one randomly
selected training image from each of 384 randomly selected classes: 128 calibration
images and 256 disjoint check images. Geometry is FP32 with TF32 disabled.
These images are held out only from temperature calibration; this is not a
validation-set generalization study of stage two. Check metrics average three stochastic draws; decoded checks use 32 of the
held-out images and one draw.

| Held-out diagnostic | Previous teacher | New combination teacher | Original RQ |
|---|---:|---:|---:|
| Target entropy, nats per complete latent code | 15.409 | 2.407 | 2.838 |
| Perturbation energy / deterministic latent energy | 0.0735% | 1.4617% | 6.0608% |
| Sampled / deterministic latent reconstruction error | 1.0240 | 0.9771 | 1.0094 |
| Decoded pixel MSE, deterministic | — | 0.00826294 | 0.00974511 |
| Decoded pixel MSE, sampled | — | 0.00820393 | 0.00978812 |
| Decoded perturbation MSE | — | 0.00116745 | 0.00335817 |

Absolute latent error is not comparable across differently scaled encoders;
ratios use each tokenizer's own deterministic baseline. Pixel MSE uses [0,1]
pixels. These are paired reconstruction diagnostics, not rFID or generation FID.
The new teacher's held-out coefficient endpoint probability averages 1.53e-5.

At the parent's update 55,600, a read-only eval on 128 identical held-out images
gave pair cross-entropy 12.7443 under the old teacher and 13.0548 under the new
teacher, a 2.44% increase. Coefficient KL rises from 1.5031 to 4.8795 because the
new targets are much sharper; lower target entropy must not be confused with
proof of easier prediction. These probes bound the injected uncertainty and
distortion, but do not guarantee better learning or final FID.

Training retains 8 H200s, 252 images per GPU, accumulation 1, global batch 2,016,
635 updates per epoch and the original 63,500-update endpoint. LR at transition
is 0.00002755146495315228 and retains the parent's warmup-cosine trajectory.
FID50k remains every five epochs and sample grids every 200 updates. The new
run logs atom CE/KL, coefficient CE/KL, and entropy for each depth separately.

Teacher benchmarking at the actual 252-image physical batch measured 0.3714 s
for the original path. Increasing the candidate site chunk from 256 to 4,032
reduced the new path from 0.7693 s to 0.5569 s. Both peaked near 8.18 GB allocated
in this isolated teacher test; full training throughput must be measured online.

Thirty-four focused frozen-runtime tests passed, covering full-combination
replacement scoring against exhaustive small examples, least-squares fits,
Gram covariance, exact finite-law conditionals, loss/gradient agreement,
autoregressive causality, fresh augmentation and resume/schedule behavior.
One unrelated VAR-backend test was excluded because its external dependency
is absent. The integrated production objective also matched the independent
soft-label loss and gradients.

The transition checkpoint is 17,597,811,245 bytes. Its local and durable copies
have identical SHA256
`cd6c12045359b76771e16d2cbfed7355974f2cbd6a54c30d51b978e3a8a40bbf`.
All weights and Adam moments are finite. The original checkpoint and best model
remain preserved. Calibration records, exact runtime source, configuration,
benchmark results and source hashes are in
`outputs/imagenet-rfid421-combination-soft-20260927/evidence/`.

Online startup completed on all eight ranks. The first 20 retained updates on
each rank had finite weights and Adam moments. Through update 55970,
median warm throughput was 1715.6 images/s and mean-per-batch
complete-code target entropy was approximately 2.423 nats, consistent
with the calibration probe. The live recovery-checkpoint verification is recorded
separately in `checkpoint-verification.json`.

The first independent recovery checkpoint at update 56000 passed
strict model, Adam and scheduler reloads. All 870 model tensors and all 870 Adam
states are finite, with eight rank RNG streams. The 17,597,811,757-byte local and
persistent payloads share SHA256 `706de5686cb3450111f1625d4c1e0d70673265481d957599522a1afc102d43e5`.
