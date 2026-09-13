"""Fixed validation-only free-running speech evaluation for checkpoint selection."""
import hashlib
import json
from pathlib import Path
import subprocess

import soundfile as sf
import torch

from src.tts_data import TTSDataset


def select_generation_records(records, count, seed):
    forbidden = {r['text_key'] for r in records if r['split'] in ('train', 'test')}
    candidates = [r for r in records if r['split'] == 'validation' and
                  2 <= r['seconds'] <= 8 and 5 <= len(r['text_key'].split()) <= 35]
    candidates.sort(key=lambda r: hashlib.sha256(f'{seed}:generation-val:{r["path"]}'.encode()).hexdigest())
    chosen, seen = [], set()
    for r in candidates:
        if r['speaker'] in seen:
            continue
        assert r['text_key'] not in forbidden, 'Generation validation text leaks into train/test'
        chosen.append(r); seen.add(r['speaker'])
        if len(chosen) == count:
            return chosen
    raise ValueError(f'Need {count} eligible validation speakers, found {len(chosen)}')


class GenerationValidation:
    def __init__(self, cache, config, output, seed):
        self.config, self.output, self.seed = config, Path(output), seed
        self.data = TTSDataset(cache, 'validation')
        self.data.records = select_generation_records(cache['records'], config['items'], seed)
        manifest = {'split': 'validation', 'selection_seed': seed, 'codec_sha256': cache['codec_sha256'],
            'sampling': {'seed_offset': 30000, 'max_frames': config.get('max_frames', 2250)},
            'asr': 'pinned Whisper-large-v3, English, beam5, fp16, temperature0, no VAD/context',
            'items': [{k: v for k, v in r.items() if k != 'codes'} for r in self.data.records]}
        path = self.output / 'generation_validation_manifest.json'
        if path.exists():
            assert json.loads(path.read_text()) == manifest
        else:
            path.write_text(json.dumps(manifest, indent=2))
        self.manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    @torch.inference_mode()
    def evaluate(self, model, decoder, epoch, step, device):
        model.eval()
        target = self.output / 'generation_validation' / f'epoch-{epoch:03d}-step-{step:07d}'
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        with torch.random.fork_rng(devices=[device.index or 0]):
            for index in range(len(self.data.records)):
                torch.manual_seed(self.seed + 30000 + index)
                item = self.data[index]; record = item['record']
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tokens, info = model.generate(item['phones'][None].to(device),
                        torch.tensor([item['speaker']], device=device),
                        max_frames=self.config.get('max_frames', 2250))
                audio, payload = decoder.decode(tokens)
                path = target / f'{index:03d}_{Path(record["path"]).stem}.wav'
                sf.write(path, audio, 48000, subtype='FLOAT')
                path.with_suffix('.bin').write_bytes(payload)
                rows.append({'text': record['text'], 'speaker': record['speaker'],
                    'reference_path': record['path'], 'audio_path': str(path.resolve()),
                    'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'payload_bytes': len(payload), 'seconds': len(audio)/48000, **info})
                print('GENERATION_VALIDATION', epoch, index+1, len(self.data.records), flush=True)
        request = target / 'generated.json'
        request.write_text(json.dumps({'manifest_sha256': self.manifest_sha, 'epoch': epoch,
            'step': step, 'rows': rows}, indent=2))
        subprocess.run([self.config['python'], 'scripts/tools/score_tts_generation_validation.py',
            '--request', str(request), '--model-record', self.config['asr_model_record'],
            '--device', str(device), '--reference-cache', str(self.output/'generation_reference_asr.json')],
            check=True)
        result = json.loads((target/'scores.json').read_text())
        assert result['manifest_sha256'] == self.manifest_sha and result['items'] == len(rows)
        return result, target
