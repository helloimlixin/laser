# MDCTCodec–LASER 6 kbps: lower learning-rate continuation

Candidate: [puutcpau](https://wandb.ai/helloimlixin-rutgers/laser/runs/puutcpau), GPU 1.
Control: [2ng5thjx](https://wandb.ai/helloimlixin-rutgers/laser/runs/2ng5thjx), GPU 0.
Both runs completed their configured 300 total epochs. Their final evaluations
and latest-plus-top-three checkpoint uploads finished successfully. See
`docs/mdctcodec-final-6kbps-2026-09-12.md` for the final results.

The candidate starts from the control's best checkpoint available at launch:
epoch 254, validation ViSQOL 4.1434107, Lightning global step 434,120, generator
update 217,060. The immutable copy and provenance are in
`outputs/vctk_mdctcodec_stage1_6kbps_low_lr/initial.ckpt` and `initialization.json`.
This parent is also retained in the control's epoch-255 W&B model artifact.

## Changes being tested

The encoder, decoder, and discriminator now start at effective LR **0.00005**.
A cosine schedule decays them to **0.000005** across 38,385 generator updates:
45 remaining epochs, each containing 853 generator updates. No LR warmup is used.
Optimizer moments, model weights, and training counters are restored.

`lr_schedule_start_step=217060` defines the continuation origin. The explicit
`lr_schedule_total_steps=38385` defines its duration independently of historical
training. Both values use generator updates: Lightning's global step includes
both optimizers. A GPU resume preflight verified that all three optimizer groups
begin at 0.00005 after restoration. The saved preflight advanced both optimizers
once, to global step 434,122 and generator step 217,061.

Configuration: `configs/vctk_mdctcodec_stage1_6kbps_low_lr.yaml`.
The 6 kbps representation, coefficient bound, loss weights, batch size, training
crops, precision, dictionary updates, and original 32-file checkpoint-ranking
validation set match the control. This comparison changes the learning rates and
their schedule. It is a single-seed experiment; restarted data sampling is not
guaranteed to replay the control's exact sequence of crops.

## Evaluation and preservation

Both runs save local latest and top-three ViSQOL checkpoints every epoch, then
upload their selected set online at completed-epoch boundaries divisible by five.
The candidate's first scheduled upload is epoch 260, after epochs 255–259.
The artifact callback now skips a resumed run's initial upload boundary when
no local restartable `last.ckpt` exists, avoiding an incomplete `latest` artifact.

An independent validation manifest contains 128 recordings: the original four
per speaker plus 12 randomly selected later utterances per speaker. Extra files
are at least two seconds long. The 200 locked test recordings are excluded.
The manifest is saved before comparison and asserts uniqueness, speaker balance,
and zero test overlap. It does not modify training or the test benchmark manifest.

`scripts/compare_mdctcodec_schedules.py` waits for **committed epoch-265 artifacts**
from both runs. This fixes the comparison boundary to ten new epochs after the
parent checkpoint, even if one run advances faster. It downloads each artifact's
best checkpoint, selected on the original val32, then evaluates both on the same
val128 using full 48 kHz waveforms and the packed five-byte-per-frame payload.
It publishes ViSQOL, STOI, actual payload rates, paired utterance bootstrap
intervals, source artifacts, hashes, and manifests online. It reports the result
without automatically stopping or switching either run.

Each run also has a finalizer that benchmarks its best validation checkpoint on
the same 200 test recordings after successful training completion, with MDCTCodec,
EnCodec, and DAC baseline comparisons. Test results do not select the LR schedule.

## Verification and reproduction

18 focused tests pass, including existing schedule behavior, cosine continuation
origin and endpoints, restored optimizer LR replacement, upload cadence and
incomplete-fork protection, MDCT transforms, sparse gradients, and 6 kbps payloads.
The comparison helper was exercised with a committed control artifact and real
ViSQOL scoring on two validation recordings. All 128 manifest entries are unique,
16 belong to each of eight speakers, and none overlaps the locked test set.

Launch with the recorded initial checkpoint present:

```bash
MDCT_CONFIG=vctk_mdctcodec_stage1_6kbps_low_lr CUDA_VISIBLE_DEVICES=1 \
  bash scripts/launch_vctk_mdctcodec_stage1.sh
```

Changing the initialization checkpoint requires updating the cosine origin and
duration to its generator counter and remaining epoch budget. To resume this same
candidate after interruption, preserve those schedule values and pass its latest
checkpoint with `wandb.id=puutcpau`, `wandb.resume=must`, and
`checkpoint.resume_in_place=true`.

Logs, process records, the manifest, and resume verification are under
`outputs/vctk_mdctcodec_stage1_6kbps_low_lr/`. The broad comparison writes
`schedule_comparison/results.json` and `schedule_comparison/run.json` when ready.
