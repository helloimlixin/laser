# Church implementation audit and control replacement

> September 19 update: the user selected the rFID-4.21 compound continuation instead. Compact-token job `61690160` and its startup observer were canceled before launch. See [the active compound continuation](church-compound-continuation-2026-09-19.md). The audit findings below remain applicable to the earlier experiment.

The user requested stopping the released-tokenizer control, investigating the
successful FFHQ run, and relaunching the Church LASER continuation on the
control's machines. Measurements below were fetched from W&B on September 18.

## What the runs establish

| Run | FID-50k | Training represented |
|---|---|---|
| `ffhqcmp0804205803` | 8.1743927 at epoch 200 | 109,200 optimizer updates; global batch 128 |
| `church-laser-ft3ep-scratch-adaptive-lr-20260916` | Best 11.8919821 at epoch 80; 13.5761926 at epoch 90 | Recoverable epoch 91, step 5,642; global batch 2,048 |
| `church-original-rqvae-released-tokenizer-control-20260917` | Best 10.0731054 at epoch 80; 11.3783753 at epoch 240 | Continued through epoch 247 before stop request was handled |

FFHQ's training was substantially different: a 2,048-atom, two-depth compound
prior, 2,048 coefficient bins, normalized stochastic coefficient targets,
atom-conditioned micro-transformer, depth-specific coefficient heads, and
geometry regularization. Its cosine schedule completed 200 epochs, with FID
improving from 18.60 at epoch 70 to 11.50 at epoch 100 and 8.17 at epoch 200.
The last logged atom prediction accuracy was 97.22%, atom NLL 0.135, and
coefficient KL 0.0878. These measure training fit, not held-out generalization.

The Church continuation uses a frozen three-epoch tokenizer, 32,769 compact
residual codes, four residual depths, and a different target distribution
(calibrated physical-distance temperature 0.125). Its 16,384 atoms have two
nonzero coefficient levels each, plus one shared zero code. It is not the
FFHQ compound implementation. FFHQ completed about 19.4 times as many optimizer
updates as the recoverable Church checkpoint; Church's 300-epoch schedule has
only 18,600 planned updates. Changing batch size now would change the saved
experiment and scheduler interpretation, so this continuation preserves them.

The runs also use different image populations: FFHQ evaluates against all
70,000 training images, Church against 126,227 training images. Cross-dataset
FID differences do not isolate a tokenizer or implementation effect. The
Church loss history suggests substantial overfitting: latest training soft CE
was 4.63 while official-validation soft CE was 10.53, above its earlier best
7.95. The released-tokenizer control also shows a large gap and a persistent
FID plateau. Neither a causal bug nor exposure bias alone is established by
these observations. There is no controlled ablation identifying why FFHQ's
recipe works better on its own dataset.

## Checks and scoped fixes

The actual epoch-91 checkpoint was loaded on an allocated L40S with the exact
recovered tokenizer and frozen source. Two real validation images exercised
encoding, targets, and decoding. Tests found:

- No future-token leakage: changing token 37 onward changed none of logits
  0 through 37; maximum difference was exactly zero.
- Cached/full FP32 prediction agreement across every one of the 256 events:
  maximum absolute logit difference 0.0000401.
- Stochastic residual-prefix targets agreed with a direct expanded-codebook
  distance calculation, maximum probability difference 0.0001731, within the
  checked FP32 cancellation tolerance.
- Chunked soft CE and its gradient agreed with the native FP32 calculation.
- Decoding was finite, and the entire frozen tokenizer hash was unchanged.

Two continuation defects were repaired in `scripts/tools/` and the run's
isolated runtime:

1. The selected-checkpoint restore loop assigned identical source and
   destination paths, so it never copied missing files; the legacy fallback
   wrote into the shared output directory while validation looked in local
   checkpoint storage. All selected files now resolve into the allocation's
   local checkpoint directory, preserving already staged files and rejecting
   missing files explicitly. Regression tests exercise movement from both
   source directories and retention of staged files.
2. Preview logging wrote `generation/epoch` and `generation/samples` without a
   new FID. For example, the last epoch-90 FID appeared next to preview epoch
   90.32258 in the summary. Previews now write only their own `preview/*` keys.

Validation now also records target entropy and prediction KL for this legacy
scalar-temperature policy. Previously these diagnostics were restricted to a
newer versioned target policy. This adds observability without changing the
training distribution or optimizer.

Fifteen focused tests passed. The actual checkpoint preflight reverified strict
model/optimizer restoration, next LR, plateau behavior, batching on 4/8/16 GPUs,
frozen tokenizer, source image ordering/pixel probes, exact real FID reference,
and the 50,000-image evaluation partition. The GPU checks above test the frozen
model computation, which the continuation fixes do not change.

## Launch and evidence

The existing queued job `61690160` was held for patching, then changed from four
A100s to eight L40S GPUs as four nodes × two. It requests exactly
`gpu[032-034,042]`, with dependency `afterany:61681659`, preserving its original
submission identity. A scheduler test accepted that shape. The control's rank-0
trainer was sent SIGTERM, allowing its existing distributed checkpoint/upload
handler to stop at an optimizer boundary. Its automatic startup monitor had
already completed; the LASER startup supervisor was stopped while this session
owns the handoff.

LASER retains its W&B identity and resumes the latest committed artifact
(resolved once for every node), initially expected at epoch 91 / step 5,642.
Global batch remains 2,048 with local batch 32 and eight accumulation steps.
Its saved effective LR is 0.0001974101465, with the saved 0.5 multiplier.
The original stage-one tokenizer remains frozen.

As documented in the earlier Amarel continuation handoff, FID sampling uses
fixed global batches of 100 with seed 71,000 plus batch index. This differs from
the historical two continuous streams, so the first new FID rebaselines plateau
comparison while retaining the LR multiplier. Historical best checkpoints stay
pinned in artifact v93. Improved FID is not established by startup checks.

Audit data, GPU report, test output, and preflight log:
`outputs/church-ft3ep-audit-20260918/`.
Runtime originals are retained in the run's `audit-20260918-backup/` directory.
The updated `ready.json` records the patched runtime hashes.
Live launch outcome is recorded in `outputs/church-ft3ep-audit-20260918/handoff.json`.

Final handoff observation at 19:15 UTC: the control exited successfully after
saving step **15,313**, epoch **247.0161**, and W&B confirmed committed artifact
**v249** with the best-three checkpoints. Slurm immediately assigned the released
GPUs to higher-priority jobs. Replacement job **61690160 is PENDING**, requesting
the exact original four nodes; no new training progress is claimed. The plain
Python observer `scripts/tools/watch_church_laser_start.py` is running detached.
It records queue/startup state and requires eight confirmed L40S ranks plus
increasing finite-loss steps over at least 60 seconds before marking healthy.
It performs no resubmission or configuration changes. Current observation is
`outputs/church-ft3ep-audit-20260918/launch-watch.json`.

The downloaded latest sample grids were inspected. FFHQ shows coherent faces;
Church shows recognizable, varied buildings but some malformed architectural
details and generated watermark-like patterns. These visual observations do
not identify a causal implementation defect or establish memorization.
