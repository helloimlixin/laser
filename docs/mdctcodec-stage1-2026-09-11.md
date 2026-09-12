# MDCTCodec–LASER stage 1: 6 kbps VCTK

Training: [2ng5thjx](https://wandb.ai/helloimlixin-rutgers/laser/runs/2ng5thjx).
Initial 6 kbps comparison and audio examples:
[7o5fghcf](https://wandb.ai/helloimlixin-rutgers/laser/runs/7o5fghcf).
Training remains active; the initial results precede adaptation to 6 kbps.
After one full adaptation epoch, the packed-payload test score is **4.1756 ViSQOL**
at 6.008 kbps (nominal 6.000), with STOI 0.8927:
[fdsl0gw1](https://wandb.ai/helloimlixin-rutgers/laser/runs/fdsl0gw1).
Its immutable evaluated checkpoint is
`outputs/mdctcodec_6kbps_after_one_epoch/checkpoint.ckpt`.
The complete early-training comparison, including both DAC checkpoints, is
[g3gha3av](https://wandb.ai/helloimlixin-rutgers/laser/runs/g3gha3av).

## Codec and continuation

Recovered the encoder, decoder, multi-resolution MDCT discriminator, and LASER
recipe from [s2er91dm](https://wandb.ai/helloimlixin-rutgers/laser/runs/s2er91dm),
the strongest recorded validation run among 41 MDCT-related runs in the account.
Configuration, source diff, and latest/top-three checkpoints are preserved in
`outputs/mdctcodec_recovery/s2er91dm`. The source diff was against commit
`2ae6b2ec30ee2c66c871f03154f5fb41a13ac059`. Only stage 1 dependencies and tests
were restored; unrelated workspace work was retained.

The model preserves MDCTCodec's native 48 kHz ConvNeXt-v2 encoder/decoder and
replaces RVQ with LASER: 8,192 atoms, latent dimension 32, OMP ridge 0.05, alternating
residual dictionary updates accumulated over eight steps. Its 80-sample MDCT window,
40-sample hop, and 8x temporal reduction produce 150 latent frames per second.
The decoder strictly loads the authors' released weights, including a layer
omitted from their published decoder source.

The 6 kbps version uses **two atoms per frame and 7-bit signed coefficients**:
150 × 2 × (13 + 7) = 6,000 bits/s. Quantization is active during training and
validation, including dictionary updates. The coefficient bound
18.45128059387207 is the 99.9th percentile of absolute OMP coefficients on 1,024
training crops, without using validation or test data. The signed grid has 127
levels, including zero; the remaining 7-bit code is reserved.

`src/mdctcodec_bitstream.py` packs each frame in exactly five bytes. The benchmark
reconstructs from this serialized payload. Headers, shared model weights, the
coefficient bound, and original sample count are outside the reported payload.
Padding short files explains the measured 6.008 kbps.

The initial four-atom continuation was stopped when 6 kbps was requested. Its last
stable state (epoch 237, global step 405,118) is copied to
`outputs/mdctcodec_6kbps_init/resume.ckpt`, with provenance beside it. The new run
restores both optimizers and switches to K=2 with quantization-aware training.
Configuration: `configs/vctk_mdctcodec_stage1_6kbps.yaml`, inheriting the recovered
stage 1 recipe. It runs on GPU 0 until total epoch 300: batch 48, 7,960-sample crops,
bf16 mixed precision, AdamW initial LR 0.0002, betas 0.8/0.99, epoch decay 0.999.
Lightning global steps include both generator and discriminator updates.

## Data and checkpoints

The mic2 split contains 40,936 training files, 32 validation files (four per
held-out speaker), and 2,845 test files. Validation and test are disjoint; the first
64 files used by earlier runs for validation are excluded from the new test set.
No gain normalization changes native waveforms. Validation scores full utterances
with official Google ViSQOL v3.3.3 in 48 kHz audio mode.

The historical best ViSQOL 4.2528 used only four validation utterances and is not
comparable to the new balanced 32-utterance validation. The new run starts its
top-three checkpoint selection afresh using quantized 6 kbps reconstruction.

Local checkpoints save every epoch. At completed-epoch boundaries divisible by
five (240, 245, …, 300), the callback uploads the latest and best three checkpoints
by `val/audio_visqol_audio48k` as one online W&B model artifact. It runs after
checkpoint selection, at the next epoch start, and also uploads at successful
training end. All checkpoints include optimizer state. Artifact collection:
`helloimlixin-rutgers/laser/model-2ng5thjx-selected-checkpoints`, with aliases
`latest`, `best-plus-last`, and `epoch-N`.
The first production upload at boundary 240 is verified online. Since this new
run had completed only epochs 238 and 239 at that boundary, it contains the two
available ranked checkpoints plus latest; subsequent uploads fill the top three.
Verification is in `outputs/vctk_mdctcodec_stage1_6kbps/verified_artifacts.json`.
The committed online `last.ckpt` was downloaded again and verified to contain
K=2, Q=7, the correct coefficient bound, both optimizers, and global step 408,530.
That recovery check is recorded in `remote_restore_check/verified.json`.

A separate six-one-batch-epoch integration test
[y6zgr0yk](https://wandb.ai/helloimlixin-rutgers/laser/runs/y6zgr0yk) verified actual
online uploads after five completed epochs and at training end, each containing
`last.ckpt` and three ranked checkpoints. This is an infrastructure test, not a
quality result. Verification is saved in
`outputs/mdctcodec_upload_integration/verified_artifacts.json`.

## Initial matched 6 kbps comparison

All systems use the same 200 full test utterances, 25 per held-out speaker and at
least two seconds long. These measurements precede the new 6 kbps training.
Baseline weights are official releases; their training corpora differ. Every
reconstruction is scored at 48 kHz in ViSQOL audio mode.

| Model | Model sample rate | Payload kbps | ViSQOL | STOI |
|---|---:|---:|---:|---:|
| LASER, two atoms, 7-bit coefficients | 48 kHz | 6.008 | 4.1690 | 0.8913 |
| Released MDCTCodec RVQ | 48 kHz | 6.008 | 4.2824 | 0.9214 |
| DAC 16 kbps checkpoint, seven codebooks | 44.1 kHz | 6.041 | 3.5424 | 0.9435 |
| DAC standard 8 kbps checkpoint, seven codebooks | 44.1 kHz | 6.041 | 3.5958 | 0.9464 |
| EnCodec stereo, duplicated mono input | 48 kHz | 6.058 | 2.8716 | 0.8647 |
| EnCodec mono | 24 kHz | 6.013 | 2.8191 | 0.8704 |

Initial LASER trails MDCTCodec by 0.1134 ViSQOL at the same payload rate. These
results do not establish a current state-of-the-art claim. The 24 kHz EnCodec model
has narrower bandwidth and should be read separately. DAC uses its official
16 kbps model with seven active quantizers, producing nominal 6.03 kbps. EnCodec
payload rates exclude gain values and headers. LASER's float-coefficient score
4.1695 is a diagnostic upper bound, not a compressed result.

`scripts/report_mdctcodec_vctk.py` publishes the 6 kbps table, paired bootstrap
intervals, and recoverable source artifact to W&B. Initial records and manifests:
`outputs/mdctcodec_benchmark_6kbps_initial`. Original per-file baseline records:
`outputs/mdctcodec_benchmark_vctk200` and `outputs/mdctcodec_benchmark_native48`.
Runtime measurements used concurrent jobs and should not support speed claims.

`scripts/finalize_mdctcodec_stage1.py` is running as a watcher. On successful
training completion it evaluates the checkpoint with the best validation ViSQOL,
using saved K=2, Q=7 and the saved coefficient bound, on the same 200 test files.
It publishes the comparison and report online. It fails explicitly if training
does not produce a final checkpoint.

## Reproduction and recovery

```bash
bash scripts/setup_mdctcodec_audio.sh
python scripts/recover_mdctcodec_wandb.py
bash scripts/launch_vctk_mdctcodec_stage1.sh
```

The launcher defaults to 6 kbps and forces W&B online mode. Those commands start
adaptation directly from the original recovered checkpoint. The active run uses
`ckpt_path=outputs/mdctcodec_6kbps_init/resume.ckpt` to retain the short continuation
already completed here.

To restart the active run, download its selected-checkpoints artifact and pass
`ckpt_path=/absolute/path/to/last.ckpt`, `checkpoint.resume_in_place=true`,
`wandb.id=2ng5thjx`, and `wandb.resume=must`. Credentials use W&B's standard
environment or netrc mechanism; no API key is stored in the repository.
A fresh experiment can use `ckpt_path=null`.

Logs, process records, and environment manifest:
`outputs/vctk_mdctcodec_stage1_6kbps/`. ViSQOL was built from v3.3.3 with Ubuntu
Armadillo 10.8.2 headers because its original SourceForge archive is unavailable.
Identity-audio preflight scored 4.7321. EnCodec 0.1.1's short-tail overlap-add error
is handled with at most 479 padding samples, counted in rates and trimmed before
scoring. Upstream reference code is pinned at
`1aff8b6287f4e2e66511bd4bc46f3a6bcf96edaf`.

Seven dedicated audio tests pass: MDCT reconstruction/energy, sparse gradients,
discriminator branches, five-epoch upload cadence, EnCodec padding, 6 kbps payload
round trips/invalid input, and quantized bottleneck grid/gradients. Actual GPU
optimizer-state restoration and online artifact uploads were exercised. The
broader core suite passed 97 tests; two image/VQ-VAE tests also fail on the untouched
base commit (channel multipliers and the old `ddpm` alias expectation).

References: [MDCTCodec paper](https://arxiv.org/abs/2411.00464),
[authors' code](https://github.com/PB20000090/MDCTCodec),
[EnCodec](https://github.com/facebookresearch/encodec),
[DAC](https://github.com/descriptinc/descript-audio-codec),
[ViSQOL](https://github.com/google/visqol).
