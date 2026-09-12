# MDCTCodec–LASER stage 2 TTS pilot — 2026-09-12

This pilot trains text-to-speech generation over the existing 6 kbps LASER audio
representation. It uses phonemes and a known VCTK speaker identity to predict a
variable-length codec sequence, which the frozen codec decodes to 48 kHz mono audio.
Intelligibility has not yet been established. This is a small-data experiment,
not a zero-shot voice-cloning system or a state-of-the-art TTS result.

Launched online at 01:55 UTC:
[W&B run 9tjq5m5d](https://wandb.ai/helloimlixin-rutgers/laser/runs/9tjq5m5d).
Local state and log: `outputs/mdctcodec_tts/stage2/`.
The initial sampler gives 714 optimizer steps per epoch and a 14,280-step cap
over 20 epochs; the eight-hour limit also applies.

The user subsequently requested a fair stage 1 comparison before further stage 2
training. The run stopped gracefully at step 3,756 after five completed epochs;
its latest optimizer/RNG checkpoint and best three models were verified committed
in W&B artifact version 1. After the
[corrected eight-codec comparison](https://wandb.ai/helloimlixin-rutgers/laser/runs/r2zvuoh9)
was committed, stage 2 resumed at approximately 02:38 UTC on the same W&B run.
Resume passed step 3,830 with finite losses. Codec choice remains the original
validation-selected epoch-287 model; confirmation-test scores did not change it.

Pre-launch checks: 12 codec/TTS tests passed; three GPU optimizer steps and a
one-step checkpoint resume completed with finite losses. Standalone generation
produced a valid packed bitstream and decoded waveform through the exact codec.
That near-initialization sample had 100% ASR word error and did not reach EOS;
it verifies execution only, not speech quality.

## Codec and data

- Frozen stage 1 checkpoint: original schedule, epoch 287, selected by highest
  stage 1 validation ViSQOL (4.150097).
- SHA256: `5c6a0d2ad5383e7b3a9e908fb0515a487b3297921c9a12c346075bc494b8c8cf`.
- 150 frames/second; each frame contains two 13-bit atom IDs and two signed
  7-bit coefficients. The real packed payload is 5 bytes/frame, or nominally
  6 kbps. The signed 127-level coefficient grid is preserved exactly.
- Full VCTK microphone-2 utterances, 0.5–12 seconds, with full transcripts.
  There are no random waveform crops paired with whole-sentence text.
- 40,892 utterances, 38.2877 hours, 100 speakers. All eight codec test speakers
  are excluded: p360, p361, p362, p363, p364, p374, p376, s5.
- Prior training: 38,288 utterances / 35.8648 hours / 11,638 unique texts.
  Validation: 1,371 / 1.2076 hours / 444 unique texts.
  Test: 1,233 / 1.2154 hours / 347 unique texts.
- Normalized text determines the split, so repeated sentences spoken by different
  people never cross a prior split. These are known-speaker TTS splits; the codec
  was already trained on these speakers' audio before the prior split was made.
- British English eSpeak phonemes with stress; vocabulary built only on prior
  training texts. Validation checkpoint selection uses a fixed, speaker-balanced
  subset of 256 utterances. The TTS test set is not used for model selection.

## Model and bounded training

The 61.9M-parameter prior has a three-layer phoneme encoder and an eight-layer
causal audio decoder with cross-attention to text (512 dimensions, eight heads).
Each temporal step predicts one complete audio frame. A recurrent depth decoder
predicts atom 1, coefficient 1, atom 2, coefficient 2 in order. Atom 2 cannot equal
atom 1. A separate EOS class on the first-atom head terminates generation and is
never written into the codec bitstream. Generation uses a KV cache.

Teacher forcing uses only earlier audio frames and earlier fields in the current
frame. Audio/text padding is masked. The objective is the mean of four categorical
cross-entropies, with increased EOS weight and an initially weak diagonal
cross-attention regularizer that decays to zero over 8,000 optimizer steps.

Configuration: `configs/vctk_mdctcodec_stage2_tts.yaml`.
AdamW, learning rate 0.0003, 500-step warmup followed by cosine decay, dropout 0.1,
bf16, gradient clipping at 1.0. Length buckets admit at most 8,192 padded frames
and 16 utterances per microbatch; four microbatches accumulate per optimizer step.
GPU 0 trains the prior; GPU 1 decodes previews and runs ASR.

The first pilot stops after 20 epochs, 20,000 optimizer steps, or eight training
hours, whichever comes first. Early generation may be noise or may fail to emit
EOS; the preview cap is ten seconds and this is logged explicitly.

## Evaluation and recovery

Local resumable checkpoints contain model, optimizer, RNG, epoch, batch position,
and W&B run ID. Save every 250 steps and every validation. Keep the best three
model checkpoints by validation token NLL. Upload the latest resumable checkpoint
and these three best checkpoints to W&B every five completed epochs and at exit.
Source/configuration, frozen codec, and the complete token cache are also archived.

Four fixed validation prompts from different speakers produce free-running WAV
and packed-bitstream previews at epoch 1 and every five epochs. W&B logs audio,
generated duration, EOS success, transcript, and ASR word error using the official
torchaudio Wav2Vec2 base model trained on LibriSpeech 960 hours. Reference recording
ASR error is logged alongside each preview to expose recognizer limitations.
Four examples are diagnostic, not a statistically reliable final TTS benchmark.
ViSQOL was useful for waveform reconstruction in stage 1; it is not used to rank
TTS generations with different timing. Falling teacher-forced loss alone does
not establish that generated speech follows the text.

Setup requires the audio dependencies and the system package `espeak-ng`:

```bash
python -m pip install -r requirements-audio.txt
apt-get update
apt-get install -y espeak-ng
python scripts/cache_mdctcodec_tts.py --device cuda:1
python scripts/train_mdctcodec_tts.py
```

Resume in place using `--resume outputs/mdctcodec_tts/stage2/checkpoints/last.pt`.
After downloading the W&B input artifact to a new machine, pass `--cache` and
`--codec-checkpoint` to override archived paths. Restore the VCTK files at their
recorded paths for reference-audio previews, or pass `--no-previews` to resume
training from tokens alone. The existing W&B run resumes using its checkpoint ID.

Generate from either the latest or a selected model checkpoint:

```bash
python scripts/generate_mdctcodec_tts.py \
  --checkpoint outputs/mdctcodec_tts/stage2/checkpoints/last.pt \
  --text 'This is a test of speech generation.' --speaker p225 \
  --output outputs/mdctcodec_tts/example.wav --asr
```

The command writes a WAV, a real 6 kbps `.bin` payload, and JSON provenance and
generation diagnostics. Use `--codec-checkpoint` after moving the frozen codec.

## Design references

The broad idea of a text-conditioned language model over acoustic codec tokens
follows [VALL-E](https://arxiv.org/abs/2301.02111). This implementation has its own
LASER-specific frame/depth factorization and is not a reproduction of VALL-E's
60,000-hour training recipe. ASR uses the
[official torchaudio model](https://docs.pytorch.org/audio/2.4.0/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html);
text processing uses [phonemizer](https://bootphon.github.io/phonemizer/api_reference.html).
