# MDCTCodec audio research dashboard

The [live audio dashboard](https://wandb.ai/helloimlixin-rutgers/laser/runs/s0hn2b6t)
compares LASER and RVQ at the same generator update count, together with the
released MDCTCodec as a separate pretrained reference. It belongs to the existing
matched training group and is linked in both training runs' notes and
`audio_media_run_url` config fields. An independent CPU observer reads committed W&B checkpoints;
the training processes continue with their existing model/optimizer state.

Online publication was verified at matched epochs 75 and 80 (63,900 and 68,160
generator updates). Each snapshot contains 32 WAV listening samples, 57 PNG
figures, 24 serialized codec payloads and an eight-row W&B comparison table.
The epoch-80 snapshot was picked up automatically after both uploads arrived.
The committed epoch-75 revision contains 117 files, and a downloaded WAV matched
the original-level local export exactly. Nine focused tests passed. Verification
records are `online_verified.json` and `cadence_verified.json` in the dashboard's
output directory.

The observer starts with the newest common uploaded checkpoint at launch. It
then publishes every five completed epochs and the final 200,000-update snapshots,
waiting for both arms to upload the corresponding checkpoint. It does not pair
different epochs or select whichever checkpoint sounds best. Earlier snapshots
can be explicitly requested using `--start-epoch` with a new output directory.

The eight fixed examples are the lexically first validation recording from each
held-out speaker. The selected paths and source hashes are recorded in
`outputs/mdctcodec_matched_6kbps_rangefix/audio_media/manifest.json`. These are
diagnostic previews; the locked 200-file test remains reserved for the final
comparison. Scores on this eight-file subset do not establish SOTA performance.

## Logged media

- `audio/examples`: a table with the same reference/LASER/RVQ/released listening
  samples, per-example ViSQOL audio48k, log-mel plots, and waveform/MDCT panels.
- `audio/reference`, `audio/laser`, `audio/rvq`, `audio/released_mdctcodec`: WAV
  listening galleries at the original 48 kHz sample rate and original levels.
- `figures/log_mel`: 128-band log-mel power spectrograms on a common −100 to 0 dB
  color scale, plus `figures/log_mel_error` on a shared ±30 dB difference scale.
- `figures/stft`: full-band 0–24 kHz log-power spectra, showing high-frequency
  artifacts that mel compression can make less apparent.
- `figures/waveform_mdct`: waveform and signed MDCT comparisons inspired by
  [Figure 4 of MDCTCodec](https://arxiv.org/html/2411.00464v1#S4.F4). The MDCT uses
  the codec's 80-sample window and 40-sample hop. Each fixed reference determines
  one symmetric color range, reused across models and epochs.
- `figures/waveforms`: full waveforms, a reference-selected 40 ms high-energy zoom,
  and reconstruction-minus-reference residuals.
- `figures/frequency_profile`: Welch power spectral density and relative energy
  in the 0–1, 1–4, 4–8, 8–16, and 16–24 kHz bands.
- `figures/temporal_error`: framewise log-spectral distance and 10 ms residual RMS.
- `figures/token_diagnostics`: LASER atom usage and signed coefficient histograms,
  RVQ usage by level, and the fraction of codes observed in this preview subset.

The paper also provides [reference and decoded audio listening examples](https://pb20000090.github.io/MDCTCodecSLT2024/).
The log-mel, error, PSD, and token panels here are additional research diagnostics;
they are not all figures taken from that paper.

## Measurement details

Reconstruction uses actual serialized five-byte frames. Each LASER checkpoint's
saved coefficient bound stays fixed throughout preview inference. WAV exports
use FLOAT samples and W&B receives file paths, avoiding per-clip normalization.
No additional gain matching or time alignment is applied.

The STFT uses a 2,048-sample periodic Hann window and 240-sample hop. Power is
`abs(STFT / sum(window)) ** 2`, followed by a 128-band triangular HTK mel filterbank
for the mel plot. Taking magnitude-squared before mel filtering avoids phase
cancellation from summing complex coefficients. Plots use `10 log10(power)` with
a power floor of `1e-12`; each reconstruction retains its original scale.

`preview/*` logs ViSQOL audio48k and speech16k, PESQ-WB16k, STOI16k, the explicitly
defined preview log-spectral distance, log-mel MAE, RMS gain, SNR, waveform MAE,
and waveform endpoint occupancy. These metrics are computed on whole recordings.
The preview LSD averages the per-frame RMS difference across all STFT log-power
bins, including DC/Nyquist; this implementation is documented and is not claimed
to reproduce the paper's LSD setting. Available full-128-file validation ViSQOL
scores are logged separately under `validation128/*`.

`preview_tokens/*` logs empirical entropy, effective code count and usage over
these eight recordings, plus LASER's saved bound, training clipping-window mean,
and raw preview coefficient clipping. Quantized endpoint occupancy is reported
separately from true clipping. Empirical token entropy does not change the fixed
6 kbps payload rate and is not an estimate of a deployed entropy-coded bitrate.

Each snapshot has a committed `audio-evaluation` artifact containing WAVs, codec
payloads, PNGs, metrics and checkpoint provenance. Source code, selection and
plotting configuration are also stored online. Input checkpoint artifacts are
registered as dependencies. On restart the observer resumes its own W&B run,
retains the published-step ledger and prevents duplicate local observers with a
file lock. A failed upload is retried without affecting either training process.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/tools/watch_mdctcodec_audio_media.py
```

The process record, heartbeat, errors and logs are under the `audio_media` directory.
Focused tests cover dB gain preservation, silent inputs, real power spectra,
serialized token decoding, validation-only example selection, and plot generation.
