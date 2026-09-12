#!/usr/bin/env python3
"""Generate a 6 kbps LASER codec bitstream and 48 kHz speech from text."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import soundfile as sf
import torch

from src.models.laser_tts import LaserTTS, TTSConfig
from src.tts_runtime import ASREvaluator, CodecDecoder, encode_prompt, word_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--codec-checkpoint', type=Path, help='Override the archived codec path after download')
    parser.add_argument('--text', required=True)
    parser.add_argument('--speaker', default='p225')
    parser.add_argument('--output', type=Path, required=True, help='Output WAV path; also writes .bin and .json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--codec-device', default='cuda:1')
    parser.add_argument('--temperature', type=float, default=.8)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--max-seconds', type=float, default=10)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--asr', action='store_true', help='Also measure generated-text ASR word error')
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 12 or args.top_k < 0:
        parser.error('max-seconds must be in (0, 12], and top-k must be nonnegative')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    metadata = state['metadata']
    if args.speaker not in metadata['speaker_to_id']:
        parser.error('Unknown speaker. Available speakers: ' + ', '.join(metadata['speaker_to_id']))
    codec_path = args.codec_checkpoint or Path(metadata['codec_checkpoint'])
    if hashlib.sha256(codec_path.read_bytes()).hexdigest() != metadata['codec_sha256']:
        raise ValueError('Codec checkpoint hash differs from the training tokenizer')
    model = LaserTTS(TTSConfig(**state['model_config'])).to(args.device).eval()
    model.load_state_dict(state['model'], strict=True)
    phones, phone_units = encode_prompt(args.text, metadata, args.device)
    speaker = torch.tensor([metadata['speaker_to_id'][args.speaker]], device=args.device)
    with torch.inference_mode(), torch.autocast(torch.device(args.device).type,
                                                dtype=torch.bfloat16, enabled=args.device.startswith('cuda')):
        tokens, info = model.generate(phones, speaker, max_frames=max(1, round(args.max_seconds * 150)),
                                     temperature=args.temperature, top_k=args.top_k)
    decoder = CodecDecoder(codec_path, args.codec_device)
    audio, payload = decoder.decode(tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, 48000)
    args.output.with_suffix('.bin').write_bytes(payload)
    result = {'text': args.text, 'phonemes': phone_units, 'speaker': args.speaker,
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_step': state['step'],
              'codec_sha256': metadata['codec_sha256'], 'sample_rate': 48000,
              'payload_bytes': len(payload), 'nominal_kbps': 6.0, 'audio_samples': len(audio),
              'seed': args.seed, 'temperature': args.temperature, 'top_k': args.top_k, **info}
    if args.asr:
        result['asr_text'] = ASREvaluator(args.codec_device).transcribe(audio)
        errors, words = word_error(args.text, result['asr_text'])
        result['asr_wer'] = errors / max(1, words)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
