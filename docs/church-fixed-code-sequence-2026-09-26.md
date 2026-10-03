# Fixed-code autoregressive LASER Church run

The user clarified that the existing sparse codes should be reused unchanged.
This experiment keeps the original cache and trains the support/coefficient
autoregressive prior from scratch. It does not introduce a new residual coder.

Runtime: `/mnt/laser-church/fixed-code-sequence-fresh-20260926`.
W&B: [fixed-code run](https://wandb.ai/helloimlixin-rutgers/laser/runs/church-fixed-codes-joint-sequence-scratch-b2048-h200x3-20260926).

The copied cache has SHA256
`4c03555ebedca1bf7b74f1db30e91752586889337a58547e856416c3aeeae586`, identical
to the original `church-compound-rqopt-scratch-b2048-4h100-20260924` cache.
It contains 126,227 images, each with an 8×8×4 sequence of atom IDs and FP32
coefficients. Its stage1 tokenizer is unchanged. That original encoder used
OMP; no OMP, re-encoding, or coefficient refitting is executed by stage2.

The stored atom IDs and coefficient values remain unchanged. Coefficients map
deterministically to the existing 2,048 normalized bins and depth scales. This
retains the original discretization rather than adding noise: its physical
coefficient MAE is 0.0029509 and maximum error is 0.0112296. No values fall
outside those bins. Training targets and teacher-forced history use the same
fixed token IDs. No alternative trajectory is selected from a bank, and no
coefficient is sampled while constructing training targets.

Both support and coefficient heads train against hard categorical labels. The
loss remains support CE plus conditional coefficient CE, averaged over pairs.
The architecture retains the RQ backbone, complete pair history, two separate
causal sequence decoders, and the local atom/coefficient conditioner. Other
settings remain dropout0.1, AdamW5e-4, betas0.9/0.95, weight decay1e-4, clipping1,
global batch2,048, and cosine300epochs without warmup or augmentation.

The former stochastic-code run was preserved at epoch169/step10,478, including
582 Adam states, the scheduler, and all three rank RNG states. Its best FID
checkpoint (10.0861, epoch76) was preserved separately. It is not an initialization
source for the new model.

Fixed-target validation passes on 127 sampled training images and the existing
300-image training and validation probes. Targets replay exactly, consume no
sampling RNG, and leave input coefficients unchanged. Seventeen tests passed for
pair causality, cache behavior, and exact global batching. GPU benchmark and
launch receipts are recorded separately under the runtime directory.

The held-out histories remain the same but coefficient targets are now hard.
Do not compare their new coefficient CE/KL numerically against the previous
soft-target KL without accounting for that objective change. Generation quality
is assessed by the same official FID50k protocol and sampler; no improvement is
claimed at launch.

## Launch verification

Production started from fresh weights on all three H200 GPUs, PID36065. Its
initialization SHA256 matches the preceding sequence architecture and every
capacity trial. Production's step10 loss and both head losses exactly match the
fresh benchmark; benchmark-trained weights were not loaded.

All three physical training capacities completed full checkpoint writes:

| Maximum microbatch per GPU | Images/sec across three GPUs |
| --- | ---: |
| 342 | 1884.20 |
| 448 | 1880.72 |
| 512 | 1436.22 |

The selected microbatch is342 with two accumulation steps. Exact rank image
weighting keeps the global batch at2,048. Generation batches6144,8192,10240
achieved177.94,178.44,161.87 images/sec/GPU respectively;8192 was selected.

The full-model dense/cached comparison covers all256 events, with maximum
absolute error9.06e-6 for support logits and8.58e-6 for coefficient logits.
Distributed resume successfully advanced a benchmark from step10 to12 with all
582 optimizer states and three RNG states restored.

Epoch1 completed at step62 with FID50k130.0707. This is an initialization-stage
measurement, not evidence of improvement. Its full checkpoint passed finiteness,
optimizer/scheduler alignment, cache-identity, and rank-RNG validation. The
unchanged code cache and executable source provenance are committed online;
model-checkpoint and predecessor upload receipts are tracked by the production
verifier under the runtime directory.

The predecessor's full last/best checkpoint artifact is now verified committed
online as `church-fixed-codes-joint-sequence-scratch-b2048-h200x3-20260926-predecessor-checkpoints:v0`.
At the last handoff check, production had reached step180 at about1,883 images/sec.
Its own model-checkpoint upload was still in progress; the running verifier
records `production-accepted.json` after checking the committed remote digests.
