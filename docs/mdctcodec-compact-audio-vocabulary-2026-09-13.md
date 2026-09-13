# Compact joint vocabularies for the frozen audio codec

This experiment tests whether the eight-level scaled-atom approach used for LSUN Church also preserves audio reconstruction quality. It evaluates calibrated 8-, 16-, and 32-level variants of the frozen MDCTCodec-LASER tokenizer, including explicitly labeled candidates at a nominal 6 kbps.

Run: https://wandb.ai/helloimlixin-rutgers/laser/runs/w4hd8re7

Output: `outputs/mdctcodec_audio_levels_20260913`.

## Completed validation results

The prespecified screen recommends **8 nonzero coefficient levels plus a 64-entry residual-vector correction** at nominal 6 kbps. All 13 variants completed all 128 recordings. The original serialized control reproduced its checkpoint's stored validation ViSQOL to the displayed precision, and encoder/dictionary/decoder parameter hashes stayed identical.

| Representation | Nominal kbps | ViSQOL audio | PESQ-WB | STOI |
|---|---:|---:|---:|---:|
| Original OMP, 127 levels | 6.0 | 4.11888 | 2.43955 | 0.90834 |
| Joint scaled RQ, 8 levels | 5.1 | 4.10698 | 2.43884 | 0.90157 |
| Joint scaled RQ, 16 levels | 5.4 | 4.10828 | 2.47041 | 0.90415 |
| Joint scaled RQ, 32 levels | 5.7 | 4.11183 | 2.46450 | 0.90436 |
| **8 levels + 64-entry correction** | **6.0** | **4.13028** | **2.57816** | **0.90965** |
| 16 levels + 16-entry correction | 6.0 | 4.12390 | 2.54888 | 0.90693 |
| 32 levels + 4-entry correction | 6.0 | 4.11825 | 2.47540 | 0.90432 |
| Matched stage1 RVQ control | 6.0 | 4.16770 | 2.69526 | 0.92284 |

The selected candidate's paired differences versus the original are ViSQOL audio **+0.01140**, 95% speaker-bootstrap interval **[−0.00324, +0.02432]**; PESQ **+0.13862**, interval **[+0.08398, +0.18428]**; and STOI **+0.001304**, interval **[+0.000016, +0.002496]**. ViSQOL speech changed by −0.01099, interval [−0.03506, +0.01636]. The screen supports retaining quality with the compact hybrid representation; it does not establish a ViSQOL improvement or superiority to the matched RVQ codec. All three refined candidates passed the prespecified tolerances; the smallest vocabulary was therefore selected.

The pure 8-level version has a small ViSQOL reduction and a STOI loss of 0.00677 at its lower 5.1 kbps rate. This is why the recommended 6 kbps variant includes the correction token. Actual payload rate is 6.00695 kbps for both the original and refined candidates, including MDCT frame alignment against original waveform duration; each encoded frame contains exactly 40 bits.

`selected_quantizer.pt` exports the chosen dictionary, eight levels, and correction codebook. `src/audio_compact_runtime.py` provides encode, decode, and decode-from-payload operations using the original frozen backbone. Its verification reproduced both stored payload bytes and waveform samples exactly for two predetermined validation utterances. This representation has three integer fields with vocabularies **[65,537, 65,537, 64]**. Stage2 TTS performance with these tokens has not been measured; the previously queued paired prior recipe remains separate.

## Controls and calibration

The frozen LASER checkpoint is the validation-selected model from the completed, matched 200,000-update codec experiment, SHA256 `3f4d31216ff54c45e0294d17eb51fa83709561c9a3528551c5fb5fed2fbeaf29`. The matched RVQ checkpoint is evaluated on the same recordings as an additional reference. Encoder, learned atom dictionary, decoder, and RVQ parameters are fingerprinted before and after inference and must remain identical.

Calibration uses 256 deterministically selected, speaker-balanced full training utterances, totaling 14.23 minutes. These belong to both the stage1 training set and the stage2 training-text split. They are disjoint from codec validation/test and stage2 validation/test audio. Symmetric scalar Lloyd fitting uses continuous two-step matching-pursuit coefficients in physical latent units. The fitted nonzero levels are shared across both residual positions, with a separate canonical zero token. The same fitted levels are used for OMP-support scalar rounding and joint scaled-atom selection.

Validation uses all 128 fixed codec validation recordings: 16 per held-out speaker, eight speakers, 7.39 minutes total. No final codec test audio or newly frozen stage2 test audio is evaluated by this screen. This is exploratory validation of frozen-codec quantizer changes, not a new independent test or a SOTA comparison.

The 13 variants are the original 127-level OMP codec, continuous OMP2 and MP2 diagnostics, the matched RVQ codec, and three variants for each level count: OMP support with scalar coefficient rounding; joint scaled-atom residual quantization; and joint quantization plus a small residual vector correction.

## Actual bitstreams

Each joint sparse token is `0` for the zero vector or `1 + atom_id * L + level_id` for a nonzero scaled atom. Joint selection uses the same `ScaledAtomRQ` implementation as Church, with two residual selections for audio. Repeated atoms are allowed and coefficients are not refitted after each selection.

| Nonzero signed levels | Joint vocabulary | Two joint IDs | Raw nominal rate | Extra correction vocabulary | Refined nominal rate |
|---|---:|---:|---:|---:|---:|
| 8 | 65,537 | 17 + 17 bits | 5.1 kbps | 64 vectors / 6 bits | 6.0 kbps |
| 16 | 131,073 | 18 + 18 bits | 5.4 kbps | 16 vectors / 4 bits | 6.0 kbps |
| 32 | 262,145 | 19 + 19 bits | 5.7 kbps | 4 vectors / 2 bits | 6.0 kbps |

All rates use 150 frames per second. Raw joint IDs are packed continuously across frame boundaries; there are at most seven final byte-padding bits per utterance. The refined variants occupy exactly 40 bits, or five bytes, per frame. Original sample count and frozen codebooks are declared stream metadata. The actual payload rate is computed from serialized file sizes and original waveform durations, including MDCT alignment and final-byte overhead.

The refinement is a **hybrid sparse-plus-small-RVQ bottleneck**, not a pure eight-level LASER codec. It selects one vector approximating the remaining encoder-latent residual from a training-only Lloyd codebook. A fixed zero entry guarantees that selecting a correction cannot increase latent squared error; waveform metrics must still be measured. These refinement entries carry information rather than padding the lower-rate representation to 6 kbps.

Every finite-rate waveform is reconstructed from parsed payload integers and the frozen codebooks. A parity check confirms that the original 127-level serialized latent matches the source codec's normal encoding path. Continuous-coefficient controls are explicitly marked as having no finite transmitted rate.

## Evaluation and selection

All variants use CUDA FP32, with matmul TF32 disabled and cuDNN TF32 enabled. Evaluation preserves original waveform levels, trims each reconstruction to the original sample count, and uses the established native ViSQOL audio48k and speech16k, PESQ-WB16k, and STOI16k implementations. Paired 95% intervals resample the eight validation speakers 5,000 times.

Before evaluation, the screen required a 6 kbps candidate to stay within 0.05 mean ViSQOL audio, 0.10 at the lower paired ViSQOL interval, 0.05 mean PESQ, and 0.005 mean STOI of the original codec. Among passing refined candidates, the smallest vocabulary is preferred. If none pass, the current codec is retained. These are practical screening tolerances, not a formal noninferiority test; variant selection uses validation results.

W&B receives the calibrated codebooks, per-file scores and payloads, all decoded audio, a rate/quality chart, coefficient-distribution plot, and predetermined listening examples from each held-out speaker. Waveform, MDCT, log-mel, spectral-error, and frequency-profile figures use consistent physical scales. The queued paired stage2 training recipe is not changed by this diagnostic runner.

## Reproduction

```bash
python scripts/tools/evaluate_audio_coefficient_levels.py \
  --root outputs/mdctcodec_audio_levels_20260913 \
  --calibration-items 256 --workers 8 --device cuda:0 --mode online
```

The two-recording smoke run completed the entire encoding, scoring, and visualization path before the full experiment. Sixteen unit tests passed with the vendored RQ package on `PYTHONPATH`, covering expanded-codebook search parity, conditioning, actual packed rates, invalid codes, canonical zero, residual refinement, and audio figures.
