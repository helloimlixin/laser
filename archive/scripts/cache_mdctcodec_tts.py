#!/usr/bin/env python3
"""Cache full VCTK utterances with transcripts using the frozen 6 kbps codec."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import soundfile as sf
import torch
from src.tts_data import CODEC_HELDOUT_SPEAKERS, phonemize_texts, text_key, text_split
from src.models.laser import LASER
from src.mdctcodec_bitstream import pack_frames, unpack_frames
from archive.scripts.benchmark_mdctcodec_vctk import align_mdct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/workspace/Projects/data/vctk'))
    p.add_argument('--output', type=Path, default=Path('outputs/mdctcodec_tts/cache'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or Path(json.loads(Path('outputs/vctk_mdctcodec_stage1_6kbps/final_evaluation.json').read_text())['checkpoint'])
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    inventory_path = args.output / 'inventory.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        assert inventory['codec_sha256'] == sha
    else:
        def inspect(path):
            speaker = path.parent.name
            if speaker in CODEC_HELDOUT_SPEAKERS:
                return None
            transcript = args.data / 'txt' / speaker / ('_'.join(path.stem.split('_')[:2]) + '.txt')
            if not transcript.exists():
                return None
            text = transcript.read_text().strip()
            info = sf.info(path)
            if info.samplerate != 48000 or info.channels != 1 or not .5 <= info.duration <= 12 or not 2 <= len(text_key(text)) <= 350:
                return None
            return {'path': str(path), 'speaker': speaker, 'text': text,
                    'text_key': text_key(text), 'split': text_split(text),
                    'samples': info.frames, 'seconds': info.duration}
        paths = sorted((args.data / 'wav48_silence_trimmed').glob('*/*_mic2.flac'))
        with ThreadPoolExecutor(max_workers=16) as pool:
            records = [r for r in pool.map(inspect, paths) if r]
        if args.limit:
            # Interleaved speakers retain voice diversity in a small preflight.
            by_speaker = {}
            for r in records:
                by_speaker.setdefault(r['speaker'], []).append(r)
            records = [r for group in zip(*[v for v in by_speaker.values()]) for r in group][:args.limit]
        print('Phonemizing', len(records), 'complete transcripts', flush=True)
        for start in range(0, len(records), 1000):
            batch = records[start:start + 1000]
            for r, phones in zip(batch, phonemize_texts([r['text'] for r in batch])):
                r['phonemes'] = phones
            print('Phonemes', min(start + 1000, len(records)), '/', len(records), flush=True)
        records = [r for r in records if 1 <= len(r['phonemes'].split()) <= 400]
        training = [r for r in records if r['split'] == 'train']
        phone_to_id = {p: i + 3 for i, p in enumerate(sorted({p for r in training for p in r['phonemes'].split()}))}
        speaker_to_id = {s: i for i, s in enumerate(sorted({r['speaker'] for r in training}))}
        records = [r for r in records if r['speaker'] in speaker_to_id]
        inventory = {'codec_checkpoint': str(checkpoint.resolve()), 'codec_sha256': sha,
                     'codec_selection': 'highest stage1 validation ViSQOL; original schedule epoch287',
                     'codec': {'sample_rate': 48000, 'frames_per_second': 150, 'atoms': 8192,
                               'sparsity': 2, 'coefficient_bits': 7, 'quantizer': 'signed127'},
                     'split_policy': 'text-disjoint SHA256, train94/validation3/test3; excludes all eight codec test speakers',
                     'phonemizer': {'backend': 'espeak', 'language': 'en-gb', 'stress': True},
                     'phone_to_id': phone_to_id, 'speaker_to_id': speaker_to_id,
                     'records': records, 'counts': dict(Counter(r['split'] for r in records)),
                     'hours': sum(r['seconds'] for r in records) / 3600}
        inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False))
    print('Inventory', inventory['counts'], inventory['hours'], 'hours', flush=True)
    records = inventory['records']
    sets = {s: {r['text_key'] for r in records if r['split'] == s} for s in ['train', 'validation', 'test']}
    assert not sets['train'] & (sets['validation'] | sets['test'])
    assert not sets['validation'] & sets['test']
    model = LASER.load_from_checkpoint(str(checkpoint), map_location='cpu').to(args.device).eval()
    model.requires_grad_(False)
    assert model.bottleneck.sparsity_level == 2 and model.bottleneck.num_embeddings == 8192
    assert model.bottleneck.coefficient_quantization_bits == 7
    bound = float(model.bottleneck.coefficient_quantization_max)
    inventory['codec']['coefficient_max'] = bound
    started = time.monotonic()
    cache_records = []
    with torch.inference_mode():
        for start in range(0, len(records), 500):
            shard = args.output / f'shard_{start:06d}.pt'
            if shard.exists():
                saved = torch.load(shard, weights_only=True)
                assert saved['codec_sha256'] == sha
                assert [r['path'] for r in saved['records']] == [r['path'] for r in records[start:start+500]]
                cache_records.extend(saved['records']); continue
            batch = []
            for r in records[start:start+500]:
                waveform, rate = sf.read(r['path'], dtype='float32')
                assert rate == 48000 and len(waveform) == r['samples']
                padded, length = align_mdct(torch.from_numpy(waveform).to(args.device)[None, None])
                _, _, sparse = model.encode(padded)
                integers = sparse.values.div(bound / 63).round().long()
                payload = pack_frames(sparse.support.cpu().numpy(), integers.cpu().numpy())
                support, q = unpack_frames(payload)
                codes = torch.empty(len(support), 4, dtype=torch.int16)
                codes[:, 0::2] = torch.from_numpy(support).short()
                codes[:, 1::2] = torch.from_numpy(q + 63).short()
                assert torch.all(codes[:, 0] != codes[:, 2])
                if not cache_records and not batch:
                    z = model.decode_from_atoms_and_coeffs(sparse.support, sparse.values)
                    decoded = model.decode_from_atoms_and_coeffs(torch.from_numpy(support).to(args.device).reshape_as(sparse.support),
                        torch.from_numpy(q).to(args.device).reshape_as(sparse.values).float() * (bound / 63))
                    torch.testing.assert_close(decoded, z, rtol=1e-5, atol=1e-5)
                    sf.write(args.output / 'codec_roundtrip.wav', decoded[0, 0, :length].cpu().numpy(), 48000)
                batch.append({**r, 'codes': codes})
            temporary = shard.with_suffix('.tmp')
            torch.save({'codec_sha256': sha, 'records': batch}, temporary); temporary.replace(shard)
            cache_records.extend(batch)
            print(f'Cached {len(cache_records)}/{len(records)} utterances in {time.monotonic()-started:.1f}s', flush=True)
    payload = {k:v for k,v in inventory.items() if k != 'records'}
    payload['records'] = cache_records
    temporary = args.output / 'tokens.tmp'
    torch.save(payload, temporary); temporary.replace(args.output / 'tokens.pt')
    (args.output / 'complete.json').write_text(json.dumps({'utterances': len(records), 'codec_sha256': sha,
        'token_cache': str((args.output / 'tokens.pt').resolve()), 'counts': inventory['counts']}, indent=2))
    print('Cache complete', args.output / 'tokens.pt', flush=True)


if __name__ == '__main__':
    main()
