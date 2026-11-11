# Guarded continuation from rFID 2.4863

This relaunch restores step 42536, epoch 4, microbatch cursor 5000 from
`helloimlixin-rutgers/laser/imnet-s1-laser-best2934-continue10-1b200-20261104-tokenizer:v4`.
Its measured full-validation rFID is 2.4862945044553726. The verified source SHA256 is
`fd96ca9a1607a07e185b61f593345aac2d0b937ec6d83aa80ebeaa1fd9b1b528`.

## Current relaunch and first-trial outcome

The initial 1e-5 trial described below was stopped by its actual FID guard at
step 42636: full rFID was 2.6507255178057676, above the 2.53602039454448 limit.
No additional training update ran after that evaluation.

An independent control then loaded the original best snapshot without training
and reproduced 2.4862945044553726 exactly on all 50000 images. The report is
`/tmp/laser-s1-best2486-b200/baseline-control/verification.json`.

The current trial is `imnet-s1-laser-best2486-lr1e6-guarded-1b200-20261104`.
It restores the same untouched best and uses 1e-6 for both learning rates,
cosine decay to 2e-7 over 2000 updates, absolute LR minimum 5e-8, and the same
full-FID regression and plateau policies. Its local root is
`/tmp/laser-s1-best2486-lr1e6-b200`, and its persistent output is
`outputs/imnet-s1-laser-best2486-lr1e6-guarded-1b200-20261104`.
`reproducibility/launcher-environment.json` there records the environment needed
to select this trial's parameters in the reusable launcher. Both trials passed
the real-data three-update preflight; neither kept those private updates.

## Initial trial policy and implementation

Both generator and discriminator learning rates start at 1e-5, approximately five
times below the previous continuation. Cosine decay reaches 2e-6 after 5000
updates, then holds that floor. Two consecutive full evaluations without a
0.005-rFID improvement halve both rates further (absolute minimum 5e-7).
Four such consecutive checks stop training. Any full rFID over 2% above the
current best stops immediately, before another optimizer update.

The generator, discriminator, both Adam states, rank-zero RNG, data epoch and
microbatch cursor are preserved. Architecture, objective, optimizer betas,
weight decay, precision and image/evaluation protocols are unchanged. The
single B200 uses microbatch 64 and accumulation 2, effective batch 128. Source
optimizer ages are 42536 and 32536.

The private preflight runs three actual resumed ImageNet updates, checks both
new learning rates, then restores the checkpoint and verifies model/RNG round
trips and optimizer ages 42539 and 32539. Production restarts from the source;
preflight updates are discarded. The current best's full 50k-image FID is inherited
with matching hardware, batching, transforms, reference and Inception backend;
it is not presented as a new measurement.

Full 50000-image evaluation occurs at step 42636 (100 new updates), then at
43036, 43536 and every 500 continuation updates, plus epoch endpoints.
The first regression threshold is 2.53602039454448. Improvement lowers it.
Latest and best snapshots are retained separately. Best snapshots are mirrored
to persistent storage and receive the W&B `best-rfid` alias.

LR multiplier, nonimprovement count, evaluation count and reduction count are
stored inside full checkpoints and remote checkpoint metadata. Each evaluation
publishes its resulting state and guard decision. Checkpoints are also published
every 500 absolute training steps. The outer limit remains epoch 10; quality
and plateau guards may stop earlier. There is no automatic restart loop after a
quality stop.

Persistent output:
`outputs/imnet-s1-laser-best2486-lr1e5-guarded-1b200-20261104/`.

Local runtime/log:
`/tmp/laser-s1-best2486-b200/runtime/` and
`/tmp/laser-s1-best2486-b200/training.log`.

Monitor `supervisor-status.json`, `status.json`, `metrics.jsonl`,
`guard-decision.json`, `learning-dynamics.json`, and `best-reconstruction.json`
in persistent output. A checkpointed manual stop uses
`touch /tmp/laser-s1-best2486-b200/run-output/STOP`.

Training reuses the verified ImageNet archives already staged from the previous
run. No dataset download is performed. Credentials remain in the private root
credential file and are not included in configuration or artifacts.

`test_dynamics.py` covers the actually observed 2.4863-to-3.0579 regression,
plateau reduction/stopping, improvement resets, serialized controller state,
LR bounds and evaluation timing. Its three tests passed before launch.
