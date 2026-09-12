# MDCTCodec LASER coefficient range fix

The first fresh matched experiment used a coefficient bound of 0.9775147,
calibrated on its randomly initialized encoder and dictionary. As the encoder
learned, raw coefficient magnitudes grew far beyond that bound: p99 reached about
46 and the observed training batch clipped every coefficient. Those runs cannot
establish the quality of a properly configured LASER codec. Their saved checkpoints
are preserved; the replacement pair starts from the exact same original fresh
initializations, rather than inheriting training through saturation.

The new experiment root is `outputs/mdctcodec_matched_6kbps_rangefix`.
`protocol.json` records the policy, source protocol hash, and unchanged initial
checkpoint and data manifest hashes. All shared weights, inputs, crop sequences,
optimizers, losses and 200,000 generator/discriminator update budgets are retained.
The released MDCTCodec is still a separate reference, with unknown matched-budget
training history. This remains a single-seed comparison of complete bottleneck
training methods, not a comprehensive SOTA claim.

After each LASER training batch's discriminator and generator updates, a range
observer reads p99.9 of the raw OMP coefficient magnitudes from the normal training
forward pass. Its target is 1.1 times that percentile. The bound grows immediately
when needed and moves 0.001 of the way toward a smaller target otherwise. Both
optimizer passes within one batch use the same bound. The initial value remains
the original calibration. This policy prevents a fresh-encoder calibration from
becoming a permanent bottleneck as the encoder scale evolves.

The observer updates one global FP32 bound, stored in the checkpoint's model
hyperparameters. Its history and clipping guard state are also checkpointed for
training resume. The model restores the bound before resumed training. Validation,
test and inference use the saved bound and do not adapt it to their recordings.
The format remains two 13-bit atom IDs plus two 7-bit signed127 coefficients per
frame: five bytes at 150 frames/second, or nominal 6 kbps. The shared model bound
is model metadata outside the frame payload; no per-frame or per-recording scale
is transmitted. Measured payload rates continue to include padding.

Training logs the bound used, next bound, raw coefficient p99.9 and fraction clipped
to W&B. After 1,000 updates, a 100-batch mean clipping fraction above 5% raises an
error. Non-finite bounds also fail immediately. The failure callback attempts an
emergency checkpoint, and the automatic reporter stops the peer arm if either
reports failure. This guard is operational protection, not a quality criterion.

Seventeen focused regression tests pass, including range drift, checkpointed
observer state, frozen validation behavior, sustained-clipping failure, data-order
resume auditing, RVQ equivalence, LASER gradients and checkpoint upload cadence.
Both GPU preflights completed 1,200 generator/discriminator updates in six reduced
200-batch epochs, with identical audited crop order and finite losses. The range
observer's 100-batch mean clipping was 0.0025% at update 1,000 and 0% at 1,200.
Both online epoch-five artifacts are COMMITTED and contain the latest checkpoint
plus three ranked checkpoints. These short runs use only two validation files
for integration testing; their scores are not quality results.

Downloaded epoch-five checkpoints restored both optimizers at update 1,000 and
continued to 1,200 with the same next-epoch inputs as uninterrupted training.
LASER restored its saved bound of 28.5552177, rather than the initial 0.9775147,
and reached 27.2204266 after the continuation with zero final-window clipping.
Loading the same remote checkpoint through the base inference model and through
the training model produced identical serialized payloads and decoded waveforms.
The inference pass left the saved bound unchanged. Evidence is in
`preflight_verified.json` and `restore_verified.json`; the reproducible verifier
is `scripts/verify_mdctcodec_rangefix.py`.

Both arms retain validation on 128 fixed full recordings, checkpoint selection by
ViSQOL audio48k, and latest plus top-three online checkpoint uploads every five
completed epochs and at successful exit. After both full budgets complete, the
reporter checks exact crop-order equality and evaluates the validation-selected
models plus released MDCTCodec on the unchanged locked 200-file test set.

Preparation and launch:

```bash
python scripts/train_mdctcodec_matched.py --root outputs/mdctcodec_matched_6kbps_rangefix --prepare-range-fix-from outputs/mdctcodec_matched_6kbps
python scripts/launch_mdctcodec_matched.py --root outputs/mdctcodec_matched_6kbps_rangefix
```

The launcher requires verified GPU preflight, remote restore evidence and a source
archive. Resume with the same root, arm and output directory plus `--resume` pointing
to that arm's saved checkpoint. Production launch IDs and status are in
`launch.json` and each arm's `run.json`; logs are in each arm's `train.log`.
