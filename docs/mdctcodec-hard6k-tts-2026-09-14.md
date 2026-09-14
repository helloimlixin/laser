# Fresh paired K4 hard6k text-to-speech training

Launched 2026-09-14 08:23 UTC: W&B campaign
[ir6dx8cl](https://wandb.ai/helloimlixin-rutgers/laser/runs/ir6dx8cl),
supervisor PID 399334. Initial state: queued, waiting for stage1 completion.
Live status: `outputs/mdctcodec_tts_hard6k_20260914/status.json`.

The stage2 supervisor is launched with:

```bash
python -u scripts/tools/run_mdctcodec_tts_pair.py \
  --root outputs/mdctcodec_tts_hard6k_20260914 \
  --hard6k-stage1 outputs/mdctcodec_k4_a4096_hard6k_long_20260913
```

It waits for both 600,000-update codec runs, the automatic stage1 comparison,
and the stage1 supervisor's exit. GPU jobs start once the GPUs are free. Waiting
uses no GPU budget. An incomplete predecessor causes an explicit failure instead
of silently selecting unfinished codecs. The old K2 stage2 queue remains held.

## Codec selection and immutable inputs

For each arm independently, `prepare_mdctcodec_hard6k_tts.py` selects
`completion.best_checkpoint`, verifies it has the largest validation ViSQOL among
the retained best three, checks completed training budget and hard6k configuration,
then copies it to `frozen_codecs/{laser,rvq}.ckpt`. The protocol records source run,
selected update, validation score, completion-file hash, and checkpoint SHA256.
Checkpoint selection happens after completion, so a late improvement is included.

Both audio caches are rebuilt from these copies. Only the previously audited
40,892-utterance text/phoneme inventory is reused. Every transcript and original
sample count is rechecked. Cache shards and merged caches carry the exact codec
hash. The trainer verifies the codec, cache, protocol, and initialization hashes.
Production never uses diagnostic tokens or resumes an old K2 prior.

## Integer representation and rate

LASER predicts eight categorical fields per coded frame: four alternating atom
IDs and coefficient bins, vocabulary `[4096, 9] * 4`, in OMP selection order. Each
later atom prediction masks all previously selected atoms. RVQ predicts four
1024-way codebook IDs. EOS is a prior control symbol outside codec payloads.

The existing hard6k packet format serializes all four LASER atom/coefficient pairs
in 57 bits per coded frame, or RVQ's four IDs in 40 bits. Canonical sorting preserves
the atom/coefficient pairing. Packets include a 16-byte header with duration and
CRC. Cache workers verify that their integer representation reproduces the codec's
actual packet byte for byte. Decoder output uses the sample count in that packet.

Generated duration is determined by the minimum representable original sample
count that accommodates the generated frames and header. Both generators use
the same 10-second preview / 15-second evaluation ceilings, converted to their
respective frame budgets. All decoded hard6k packets obey 6000 bit/s including
headers and byte padding. Audio attention plots use actual generated duration.

## Paired prior recipe

Both priors start from identical shared text, temporal, and depth-transformer
weights, seed 20260914. Width 512, eight heads, three text layers, eight temporal
layers, two causal depth layers, dropout 0.1. Eight depth positions are allocated
in both models; RVQ uses the first four. Vocabulary embeddings and output heads
differ. LASER has 66,417,701 parameters and RVQ has 53,785,601. Both counts are
recorded when the final plan is constructed.

Train for 160 epochs: **114,193 optimizer updates per arm**, below the120,000-step
ceiling. Batch packing uses original audio duration on the native 150Hz grid,
8192-frame padded budget, maximum 16 utterances, accumulation 4. This matches
utterance batches, optimizer steps and learning rates despite different coded
frame counts. Loss masks use actual codec lengths. The paired data-order audit
hashes utterance IDs, phonemes, speakers and common native-frame lengths; its
`frames` counter is a common audio-duration counter, not actual coded tokens.

AdamW, peak LR 0.0003, 1000-update warmup and cosine decay to 1e-5, weight decay 0.01,
gradient clip 1; guided-attention weight 0.2 decays over 8000 updates. Both arms have
a 24-hour training ceiling. The entire new campaign has a 48 assigned-GPU-hour
ceiling including caching, preflight and evaluation. A budget interruption is
reported as incomplete; it does not produce a claimed matched final comparison.

After fresh cache construction, both production priors must pass two real
optimizer updates plus generated-audio previews and equal data-order audits
before either training job is launched. LASER runs on GPU 0 and RVQ on GPU 1.

## Selection, logging and evaluation

Stage1 selection uses validation **ViSQOL**. Stage2 selects using lowest Whisper
WER on 64 fixed validation prompts from 64 speakers, evaluated every 10 completed
epochs, earliest candidate on ties. Token NLL is only an additional within-arm
criterion. Online checkpoint uploads every 5 completed epochs contain latest
optimizer state, best 3 validation-WER priors, and best 3 within-arm-NLL priors.
The ViSQOL-selected frozen codec and token cache are uploaded as input artifacts.

Every 5 completed epochs, log 16 validation audio samples, references, waveforms,
log-mel spectrograms, blue text-attention maps, transcripts, duration, and packet
bitrate. The existing stage1 latest-plus-best3-ViSQOL uploads remain separate.

After both priors finish 160 epochs with identical audited update/data sequences,
evaluate their best-WER and final-epoch models on the reserved 100 distinct test
texts, one per known speaker. Include reference speech and the respective codec
reconstructions. Report WER/CER, UTMOS, ECAPA speaker similarity, RTF, packet rates,
and paired speaker-bootstrap intervals. Test scores never select checkpoints.

This is a controlled, single-seed, known-speaker LASER/RVQ comparison. Different
field counts, coded frame rates, and vocabulary-specific parameter counts are
reported. It is not a general TTS SOTA claim or a human listening evaluation.

## Verification

47 focused tests pass across codec transport, K4 causal teacher forcing versus
cached generation, distinct-atom constraints, shared initialization/data audits,
completion-gated ViSQOL selection, schedules, generation selection and media.
Separate diagnostic caches use 12 real VCTK utterances and current codec candidates;
they verify original-length packet equality on CUDA before production selection.
Diagnostic run files live under `outputs/mdctcodec_tts_hard6k_20260914/diagnostics`.
They are untrained smoke checks and are excluded from production caches and scores.
Both full-size priors also passed two optimizer updates, equal data-order audits,
and audio/media generation. Their generated packets measured exactly 6.0 kbps
including headers. The queue conservatively charges 120 GPU seconds for these
initial diagnostic checks before its tracked production jobs start.
