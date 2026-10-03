# Fresh Church DCTransformer adaptation, 2026-09-29

Requested return to the earlier DCTransformer architecture after implementation audit. This is an adaptation to frozen OMP sparse codes, not a reproduction of the DCT image representation or its published FID.

The 336,167,227 parameter model predicts each support atom and then its coefficient conditional on that atom and completed pairs. Four depth-major 8x8 planes use causal overlap of eight prior events. The prefix encoder receives only completed earlier planes. The coefficient decoder receives the selected current atom. Repeated atoms at each site are masked consistently with greedy OMP data.

The original architecture is unchanged. All four coefficient depths now use exactly the same 2048-entry physical grid, with unit depth scales. The canonical original support sequence is preserved; physical coefficients are converted once from the legacy normalized cache and deterministically quantized without refitting, clipping, alternate banks or stochastic targets. Exhaustive codec and small decoded-image audits passed.

Fresh model and Adam state; 300 epochs, 18,600 updates, eight H100 GPUs, global batch 2048, local microbatch 128, two accumulation steps. Peak LR 0.0005, 1000-update warmup, cosine decay, gradient clip 1. Hard joint pair NLL is trained as half the sum of support and conditional coefficient cross entropy. Both terms have equal weight. No reward, covariance penalty, contrastive term or geometry loss.

Sampling uses all eligible atoms and all 2048 coefficient bins at temperature 1, drawing coefficients conditional on the sampled atom. Preview 64 Church images every 200 updates. Official continuous-pixel 50,000-sample FID with fixed reference and streams every 620 updates and at completion. Full LAST upload at step 200 and LAST/BEST at every FID boundary and completion; local full snapshots every 200 updates. Frozen tokenizer, canonical cache, exact bin table, source, optimizer, scheduler and rank RNG states are retained for recovery.

Prelaunch gates: independent active-gate causal/cache audit; exhaustive data and decoder audit; production-size eight-GPU smoke with FP32/BF16 cached sampling checks; disposable 128-update fixed-batch learnability/context-use check; split versus uninterrupted checkpoint-resume comparison. Probe weights and optimizer are never used for training. Passing these checks establishes implementation consistency and basic learning capacity, not lower FID or generalization.

The original DC reference uses DCT channel/value events and a different image and training protocol. Our physical sparse sums compress support identities, OMP depth is not frequency order, and exposure bias remains. An old checkpoint probe found weak support use of the reconstructed prefix; the new learning probe measures both heads and context dependence before spending on production training.

References: https://proceedings.mlr.press/v139/nash21a/nash21a.pdf and https://proceedings.mlr.press/v139/nash21a/nash21a-supp.pdf

## Measured prelaunch results

The production-size eight-GPU smoke passed on every rank. All 488 parameter tensors received finite nonzero gradients after three updates, all Adam steps matched, all rank parameters agreed, and all 59 ReZero gates moved. Steady update time was 0.688s with 54.3 GiB peak memory. Full versus cached predictions were checked at all 256 events with nonzero residual gates; FP32 max logit error was below 4.3e-6 and BF16 maximum KL below 9.1e-6. Generated paired tokens had unique support at every site and decoded to finite 256x256 images.

A separate randomly initialized model trained for 128 updates on 64 fixed training images. Atom NLL fell from 9.8794 to 0.0167; coefficient NLL from 7.7943 to 0.0015. Shuffling causal pair history substantially worsened both losses, and shuffling the current support worsened coefficient prediction. This demonstrates basic capacity and conditional use on a tiny memorized set, not held-out quality. Prefix shuffling alone barely changed support predictions, so strong use of reconstructed physical context remains unproven. Probe state was discarded.

The outgoing coverage95 run was retired with its complete step-5400 checkpoint preserved, plus its independent best checkpoint. Its latest logged training progress was 5480; uncheckpointed later updates are not claimed to have been preserved. The latest measured FID belongs to step4960, not5400.

Runtime: `/tmp/laser-church-dc-shared-20260929`. Durable sources, receipts and recovery: `/workspace/Projects/laser/outputs/church-dc-shared-20260929`.

## Resume audit and production launch

Loading restored every model and Adam tensor exactly. After one resumed update versus the fourth uninterrupted update, all eight RNG states, scheduler, cursor, source, configuration and logged losses matched exactly. The bitwise continuation test failed: 70 model tensors had differences up to 3.7253e-9, and 306 Adam tensors up to 1.1642e-10. All differences were finite FP32 values, consistent with backward/reduction roundoff. The complete failed bitwise report is preserved. A separate explicit numerical acceptance audit passed using absolute tolerance1e-7 and exact nonfloating state; no bitwise continuation guarantee is made.

The fresh production controller launched with PID264516 and torchrun PID264582, with no resume checkpoint. W&B run: https://wandb.ai/helloimlixin-rutgers/laser/runs/church-dc-shared-20260929 . All planned 300 epochs remain scheduled. Online artifacts and initial progress will be verified separately.


## Verified online handoff

Fresh training reached step440 with no restart at 0.686s/update (2984 images/s). All frozen production sources still match the launch manifest. Source:v0 (39 files) and recovery:v0 (four assets) were independently verified remotely. The step20064-image grid was downloaded from W&B and matched by SHA256. Full LAST200 is committed as `church-dc-shared-20260929-selected-checkpoints:v0`:4,034,699,346bytes,488modeltensors,488Adamstates,scheduler200 andeight rankRNGstates; local/remote/durable checks passed. BEST publication begins after the first measured FID atstep620.

The first samples are incoherent early-training compositions, not evidence of good FID. There is no new measured FID at this handoff.

Detailed launch receipt: `outputs/church-dc-shared-20260929/launch-complete.json`; independent publication audit: `outputs/church-dc-shared-20260929/diagnostics/launch-publication-audit.json`.
