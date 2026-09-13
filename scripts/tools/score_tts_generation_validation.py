#!/usr/bin/env python3
"""Score fixed validation generations in the isolated benchmark environment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from scripts.tools.benchmark_mdctcodec_tts import edit_distance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--model-record', type=Path, required=True)
    p.add_argument('--reference-cache', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(); torch.set_num_threads(4)
    request = json.loads(args.request.read_text())
    model_record = json.loads(args.model_record.read_text())
    for name, digest in model_record['files'].items():
        assert hashlib.sha256((Path(model_record['path'])/name).read_bytes()).hexdigest() == digest
    model = WhisperModel(model_record['path'], device='cuda', device_index=torch.device(args.device).index or 0,
                         compute_type='float16')
    normalize = EnglishTextNormalizer({})
    cache = json.loads(args.reference_cache.read_text()) if args.reference_cache.exists() else {}
    identity = {'manifest_sha256': request['manifest_sha256'], 'asr_revision': model_record['revision']}
    if cache:
        assert cache['identity'] == identity
    else:
        cache = {'identity': identity, 'rows': {}}

    def transcribe(path):
        audio, sr = sf.read(path, dtype='float32')
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
        segments, _ = model.transcribe(audio, language='en', beam_size=5, temperature=0,
            vad_filter=False, condition_on_previous_text=False)
        return ' '.join(s.text.strip() for s in segments)

    for row in request['rows']:
        assert hashlib.sha256(Path(row['audio_path']).read_bytes()).hexdigest() == row['audio_sha256']
        hypothesis = transcribe(row['audio_path'])
        source = row['reference_path']
        source_sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if source not in cache['rows']:
            cache['rows'][source] = {'sha256': source_sha, 'text': transcribe(source)}
        assert cache['rows'][source]['sha256'] == source_sha
        ref, hyp, original = normalize(row['text']), normalize(hypothesis), normalize(cache['rows'][source]['text'])
        row.update(asr_text=hypothesis, words=len(ref.split()), word_errors=edit_distance(ref.split(), hyp.split()),
            characters=len(ref.replace(' ', '')), char_errors=edit_distance(ref.replace(' ', ''), hyp.replace(' ', '')),
            reference_word_errors=edit_distance(ref.split(), original.split()))
    rows = request['rows']; words = sum(r['words'] for r in rows)
    result = {**identity, 'epoch': request['epoch'], 'step': request['step'], 'items': len(rows),
        'wer': sum(r['word_errors'] for r in rows)/words,
        'cer': sum(r['char_errors'] for r in rows)/sum(r['characters'] for r in rows),
        'reference_wer': sum(r['reference_word_errors'] for r in rows)/words,
        'eos_fraction': sum(r['eos_reached'] for r in rows)/len(rows), 'rows': rows}
    args.reference_cache.write_text(json.dumps(cache, indent=2))
    (args.request.parent/'scores.json').write_text(json.dumps(result, indent=2))
    print('GENERATION_SCORES', json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    main()
