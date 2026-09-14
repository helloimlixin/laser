#!/usr/bin/env python3
"""Frozen, paired VCTK TTS evaluation: prepare, generate, score, and publish."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf
import torch

from src.tts_data import text_key, text_split


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def prepare(args):
    from src.tts_data import CODEC_HELDOUT_SPEAKERS
    cache = torch.load(args.cache, map_location='cpu', weights_only=True)
    by_speaker = defaultdict(lambda: defaultdict(list))
    sets = defaultdict(set)
    for r in cache['records']:
        assert text_split(r['text']) == r['split']
        sets[r['split']].add(text_key(r['text']))
        by_speaker[r['speaker']][r['split']].append(r)
    assert not (sets['train'] & sets['test'] or sets['validation'] & sets['test'])
    assert not (set(by_speaker) & CODEC_HELDOUT_SPEAKERS)
    rows = []
    clean = lambda r: {k: v for k, v in r.items() if k != 'codes'}
    for speaker, splits in sorted(by_speaker.items()):
        eligible = [r for r in splits['test'] if 2 <= r['seconds'] <= 8 and 5 <= len(r['text_key'].split()) <= 35]
        assert eligible, speaker
        target = min(eligible, key=lambda r: hashlib.sha256(f'20260912-benchmark:{r["path"]}'.encode()).hexdigest())
        refs = [r for r in splits['train'] if 3 <= r['seconds'] <= 10]
        assert refs, speaker
        enrollment = min(refs, key=lambda r: (abs(r['seconds'] - 8), r['path']))
        assert target['path'] != enrollment['path'] and target['text_key'] != enrollment['text_key']
        row = {'id': Path(target['path']).stem, 'speaker': speaker,
               'target': clean(target), 'enrollment': clean(enrollment), 'seed': 20260912 + len(rows)}
        for field in ['target', 'enrollment']:
            r = row[field]; info = sf.info(r['path'])
            assert info.samplerate == 48000 and info.channels == 1 and info.frames == r['samples']
            r['sha256'] = sha(r['path'])
        rows.append(row)
    manifest = {'version': 1, 'seed': 20260912, 'items': rows, 'cache_sha256': sha(args.cache),
        'codec_sha256': cache['codec_sha256'], 'codec_checkpoint': cache['codec_checkpoint'],
        'selection': 'One fixed hash-selected test utterance per 100 known speakers; 2-8 seconds, 5-35 words; no output selection.',
        'conditioning': 'LASER uses learned speaker ID; F5 and Chatterbox use the same different training utterance (nearest 8 seconds).',
        'limitations': ['System-level comparison, not matched training-data/compute or a zero-shot LASER evaluation.',
            'Prior targets have transcript-disjoint train/validation/test splits. Frozen codec saw these speakers and may have seen target audio.',
            'Released baseline pretraining overlap with VCTK is unknown. No claim of uncontaminated external training.',
            'UTMOS is a learned proxy, not human MOS. ECAPA similarity is not the WavLM SIM used in published F5 tables.',
            'These VCTK scores are not directly comparable to published Seed-TTS/LibriSpeech scores.'],
        'generation': {'seed_per_prompt': '20260912 + manifest index', 'max_seconds': 15,
            'laser': {'temperature': .8, 'top_k': 50, 'max_frames': 2250},
            'f5': {'model': 'F5TTS_v1_Base', 'nfe_step': 32, 'cfg_strength': 2, 'sway_sampling_coef': -1, 'speed': 1},
            'chatterbox': {'model': 'ChatterboxTurboTTS', 'settings': 'Released package 0.1.6 defaults, including watermark'}},
        'metrics': {'asr': 'Systran/faster-whisper-large-v3, float16, beam5, English, temperature0, no VAD or previous-text conditioning',
            'normalizer': 'transformers Whisper EnglishTextNormalizer with empty spelling map',
            'speaker': 'speechbrain/spkrec-ecapa-voxceleb cosine versus enrollment, 16kHz',
            'quality': 'UTMOS22 strong from tarepan/SpeechMOS v1.2.0, original levels resampled to 16kHz',
            'speed': 'Batch1 wall time including text/reference preparation and decoder, CUDA synchronized; model load/warmup/file write excluded; one H200 GPU per system.'}}
    assert len(rows) == 100
    destination = args.root / 'manifest.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == manifest, 'Refusing to replace a frozen benchmark manifest'
    else: write_json(destination, manifest)
    write_json(args.root / 'manifest_sha256.json', {'sha256': sha(destination), 'items': len(rows)})
    print('PREPARED', len(rows), sha(destination), flush=True)


def hf_snapshot(repo, patterns, root):
    from huggingface_hub import HfApi, snapshot_download
    record_path = root / 'models' / (repo.replace('/', '--') + '.json')
    if record_path.exists():
        record = json.loads(record_path.read_text())
    else:
        record = {'repo': repo, 'revision': HfApi().model_info(repo).sha}
        write_json(record_path, record)
    path = snapshot_download(repo, revision=record['revision'], allow_patterns=patterns)
    record.update(path=path, files={str(p.relative_to(path)): sha(p) for p in Path(path).rglob('*') if p.is_file()})
    write_json(record_path, record)
    return Path(path)


def generate(args):
    manifest = json.loads((args.root / 'manifest.json').read_text())
    rows = manifest['items'][:args.limit or None]
    arm_dir = args.root / 'generated' / args.arm; arm_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    random.seed(20260912); np.random.seed(20260912); torch.manual_seed(20260912)
    packages = {}
    for package in ['torch', 'torchaudio', 'f5-tts', 'chatterbox-tts', 'transformers', 'numpy']:
        try: packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    provenance = {'manifest_sha256': sha(args.root / 'manifest.json'), 'arm': args.arm,
        'device': str(device), 'gpu': torch.cuda.get_device_name(device), 'packages': packages}
    cache = None
    is_prior = args.arm.startswith(('laser', 'rvq'))
    is_codec = args.arm == 'codec' or args.arm.startswith('codec_')
    codec_arm = 'rvq' if args.arm.startswith('rvq') or args.arm == 'codec_rvq' else 'laser'
    codec_info = manifest.get('codecs', {}).get(codec_arm, manifest)
    if (is_prior or is_codec) and 'codec_checkpoint' not in codec_info:
        protocol_path = args.root.parent / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        assert protocol['benchmark_manifest_sha256'] == provenance['manifest_sha256']
        codec_info = protocol['arms'][codec_arm]
        provenance['paired_protocol_sha256'] = sha(protocol_path)
    if is_prior or is_codec:
        from src.tts_runtime import CodecDecoder
        decoder = CodecDecoder(codec_info['codec_checkpoint'], str(device))
        assert sha(codec_info['codec_checkpoint']) == codec_info['codec_sha256']
        if is_prior:
            from src.models.laser_tts import LaserTTS, TTSConfig
            assert args.checkpoint and args.checkpoint.is_file()
            state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            assert state['metadata']['codec_sha256'] == codec_info['codec_sha256']
            model = LaserTTS(TTSConfig(**state['model_config'])).to(device).eval()
            model.load_state_dict(state['model'], strict=True)
            metadata = state['metadata']
            provenance.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
                step=state['step'], epoch=state['epoch'], validation_nll=state.get('validation_nll'),
                parameters=sum(p.numel() for p in model.parameters()), precision='bf16 prior, fp32 frozen codec',
                timed_text_preparation='espeak-ng phonemization from raw text')
        else:
            cache = torch.load(args.cache, map_location='cpu', weights_only=True)
            assert sha(args.cache) == codec_info['cache_sha256']
            records = {r['path']: r for r in cache['records']}
    elif args.arm == 'f5':
        from f5_tts.api import F5TTS
        model_path = hf_snapshot('SWivid/F5-TTS', ['F5TTS_v1_Base/*'], args.root)
        vocoder_path = hf_snapshot('charactr/vocos-mel-24khz', ['config.yaml', 'pytorch_model.bin'], args.root)
        model = F5TTS(model='F5TTS_v1_Base', ckpt_file=str(model_path / 'F5TTS_v1_Base/model_1250000.safetensors'),
            vocab_file=str(model_path / 'F5TTS_v1_Base/vocab.txt'), vocoder_local_path=str(vocoder_path), device=str(device))
        provenance['parameters'] = sum(p.numel() for p in model.ema_model.parameters()) + sum(p.numel() for p in model.vocoder.parameters())
    elif args.arm == 'chatterbox':
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        model_path = hf_snapshot('ResembleAI/chatterbox-turbo', ['*.safetensors', '*.json', '*.txt', '*.pt', '*.model'], args.root)
        model = ChatterboxTurboTTS.from_local(model_path, device=str(device))
        provenance['parameters'] = sum(p.numel() for module in [model.t3, model.s3gen, model.ve] for p in module.parameters())
    elif args.arm != 'reference':
        raise ValueError(args.arm)
    provenance_path = arm_dir / 'provenance.json'
    if provenance_path.exists():
        assert json.loads(provenance_path.read_text()) == provenance, 'Generation settings or checkpoint changed'
    else: write_json(provenance_path, provenance)

    @torch.inference_mode()
    def synthesize(row, warmup=False):
        target = row['enrollment'] if warmup else row['target']
        torch.manual_seed(row['seed']); np.random.seed(row['seed']); random.seed(row['seed'])
        info, payload = {}, None
        if is_prior:
            from src.tts_runtime import encode_prompt
            phones, _ = encode_prompt(target['text'], metadata, device)
            speaker = torch.tensor([metadata['speaker_to_id'][row['speaker']]], device=device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens, info = model.generate(phones, speaker, max_frames=2250, temperature=.8, top_k=50,
                    max_seconds=manifest.get('generation',{}).get('max_seconds'),
                    min_seconds=.2 if model.cfg.hard_rate_cap_bps else None)
            audio, payload = decoder.decode(tokens); sr = 48000
        elif is_codec:
            record=records[target['path']]
            audio, payload = decoder.decode(record['codes'],samples=record['samples'] if decoder.hard6k else None); sr = 48000
        elif args.arm == 'reference': audio, sr = sf.read(target['path'], dtype='float32')
        elif args.arm == 'f5':
            audio, sr, _ = model.infer(ref_file=row['enrollment']['path'], ref_text=row['enrollment']['text'],
                gen_text=target['text'], nfe_step=32, cfg_strength=2, sway_sampling_coef=-1,
                speed=1., seed=row['seed'], show_info=lambda *_: None)
        else:
            audio = model.generate(target['text'], audio_prompt_path=row['enrollment']['path']).detach().float().cpu().numpy().reshape(-1)
            sr = model.sr
        return np.asarray(audio, dtype=np.float32), sr, payload, info

    if args.arm != 'reference' and not is_codec:
        synthesize(manifest['items'][0], warmup=True)
        torch.cuda.synchronize(device)
    for index, row in enumerate(rows):
        result_path = arm_dir / (row['id'] + '.json')
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            assert saved['manifest_sha256'] == provenance['manifest_sha256']
            assert sha(saved['audio_path']) == saved['audio_sha256']
            continue
        assert sha(row['target']['path']) == row['target']['sha256']
        assert sha(row['enrollment']['path']) == row['enrollment']['sha256']
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        start = time.perf_counter()
        audio, sr, payload, info = synthesize(row)
        torch.cuda.synchronize(device); elapsed = time.perf_counter() - start
        assert len(audio) > 0 and np.isfinite(audio).all()
        truncated = len(audio) > 15 * sr
        audio = audio[:15 * sr]
        path = arm_dir / (row['id'] + '.wav'); sf.write(path, audio, sr, subtype='FLOAT')
        if payload is not None: path.with_suffix('.bin').write_bytes(payload)
        result = {'id': row['id'], 'speaker': row['speaker'], 'arm': args.arm,
            'manifest_sha256': provenance['manifest_sha256'], 'audio_path': str(path.resolve()),
            'audio_sha256': sha(path), 'sample_rate': sr, 'samples': len(audio), 'seconds': len(audio) / sr,
            'synthesis_seconds': elapsed, 'rtf': elapsed / (len(audio) / sr),
            'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
            'truncated_at_15_seconds': bool(truncated), 'payload_bytes': len(payload) if payload is not None else None,
            **info}
        # Use actual waveform duration (MDCT overlap changes it slightly).
        result['seconds'] = len(audio) / sr
        write_json(result_path, result)
        print('GENERATED', args.arm, index + 1, len(rows), row['id'], round(elapsed, 3), flush=True)
    write_json(arm_dir / 'complete.json', {'items': len(rows), 'manifest_sha256': provenance['manifest_sha256']})


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def score(args):
    import torchaudio
    from faster_whisper import WhisperModel
    from speechbrain.inference.speaker import EncoderClassifier
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    device = torch.device(args.device); torch.set_num_threads(4)
    manifest = json.loads((args.root / 'manifest.json').read_text())
    normalizer = EnglishTextNormalizer({})
    asr_path = hf_snapshot('Systran/faster-whisper-large-v3', ['*.json', '*.bin', '*.txt'], args.root)
    ecapa_path = hf_snapshot('speechbrain/spkrec-ecapa-voxceleb', ['*.ckpt', '*.yaml', '*.txt'], args.root)
    asr = WhisperModel(str(asr_path), device='cuda', device_index=device.index or 0, compute_type='float16')
    speaker_model = EncoderClassifier.from_hparams(source=str(ecapa_path), savedir=str(args.root / 'ecapa_runtime'),
        overrides={'pretrained_path': str(ecapa_path)}, run_opts={'device': str(device)})
    mos = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True).to(device).eval()
    provenance = {'manifest_sha256': sha(args.root / 'manifest.json'), 'metrics': manifest['metrics'],
        'UTMOS_checkpoint_sha256': sha(Path(torch.hub.get_dir()) / 'checkpoints/utmos22_strong_step7459_v1.pt')}
    provenance['packages'] = {p: importlib.metadata.version(p) for p in
        ['torch', 'torchaudio', 'faster-whisper', 'ctranslate2', 'speechbrain', 'transformers']}
    write_json(args.root / 'metric_provenance.json', provenance)
    enrollments = {}

    def load16(path):
        audio, sr = sf.read(path, dtype='float32')
        wave = torch.from_numpy(audio).to(device)[None]
        return torchaudio.functional.resample(wave, sr, 16000)

    with torch.inference_mode():
        for arm in args.arm.split(','):
            dest = args.root / 'scores' / arm; dest.mkdir(parents=True, exist_ok=True)
            for index, row in enumerate(manifest['items'][:args.limit or None]):
                result = json.loads((args.root / 'generated' / arm / (row['id'] + '.json')).read_text())
                assert result['manifest_sha256'] == provenance['manifest_sha256']
                assert sha(result['audio_path']) == result['audio_sha256']
                output = dest / (row['id'] + '.json')
                if output.exists():
                    old = json.loads(output.read_text()); assert old['audio_sha256'] == result['audio_sha256']
                    continue
                wave = load16(result['audio_path'])
                segments, _ = asr.transcribe(wave[0].cpu().numpy(), language='en', beam_size=5,
                    temperature=0, vad_filter=False, condition_on_previous_text=False)
                hypothesis = ' '.join(segment.text.strip() for segment in segments)
                ref, hyp = normalizer(row['target']['text']), normalizer(hypothesis)
                words = ref.split(); errors = edit_distance(words, hyp.split())
                chars = ref.replace(' ', ''); char_errors = edit_distance(chars, hyp.replace(' ', ''))
                if row['speaker'] not in enrollments:
                    enrollments[row['speaker']] = speaker_model.encode_batch(load16(row['enrollment']['path'])).flatten()
                embedding = speaker_model.encode_batch(wave).flatten()
                similarity = float(torch.nn.functional.cosine_similarity(embedding, enrollments[row['speaker']], dim=0))
                utmos = float(mos(wave, 16000)[0])
                result.update(text=row['target']['text'], asr_text=hypothesis, normalized_reference=ref, normalized_hypothesis=hyp,
                    word_errors=errors, words=len(words), wer=errors / max(1, len(words)),
                    char_errors=char_errors, characters=len(chars), cer=char_errors / max(1, len(chars)),
                    speaker_similarity_ecapa=similarity, utmos=utmos)
                write_json(output, result)
                print('SCORED', arm, index + 1, 'WER', round(result['wer'], 4), 'UTMOS', round(utmos, 3), flush=True)


def aggregate(rows):
    return {'items': len(rows), 'wer': sum(r['word_errors'] for r in rows) / sum(r['words'] for r in rows),
        'cer': sum(r['char_errors'] for r in rows) / sum(r['characters'] for r in rows),
        'speaker_similarity_ecapa': float(np.mean([r['speaker_similarity_ecapa'] for r in rows])),
        'utmos': float(np.mean([r['utmos'] for r in rows])),
        'rtf': sum(r['synthesis_seconds'] for r in rows) / sum(r['seconds'] for r in rows),
        'median_latency_seconds': float(np.median([r['synthesis_seconds'] for r in rows])),
        'p95_latency_seconds': float(np.percentile([r['synthesis_seconds'] for r in rows], 95)),
        'peak_gpu_allocated_gib': max(r['peak_gpu_allocated_gib'] for r in rows),
        'cap_fraction': float(np.mean([r['truncated_at_15_seconds'] or r.get('eos_reached') is False for r in rows]))}


def report(args):
    import wandb
    from src.audio_research_media import render_tts_preview
    import torchaudio
    manifest = json.loads((args.root / 'manifest.json').read_text())
    by_arm = {}
    for arm in args.arm.split(','):
        rows = [json.loads((args.root / 'scores' / arm / (r['id'] + '.json')).read_text()) for r in manifest['items']]
        assert len(rows) == 100 and len({r['id'] for r in rows}) == 100
        by_arm[arm] = rows
    summaries = {arm: aggregate(rows) for arm, rows in by_arm.items()}
    # One utterance per speaker; resampling speakers preserves paired comparisons.
    rng = np.random.default_rng(20260912); indices = rng.integers(0, 100, size=(5000, 100))
    bootstrap = {}
    for arm, rows in by_arm.items():
        errors = np.array([r['word_errors'] for r in rows]); words = np.array([r['words'] for r in rows])
        bootstrap[arm] = {'wer': errors[indices].sum(1) / words[indices].sum(1)}
        for key in ['speaker_similarity_ecapa', 'utmos']:
            bootstrap[arm][key] = np.array([r[key] for r in rows])[indices].mean(1)
        summaries[arm]['ci95'] = {key: np.percentile(value, [2.5, 97.5]).tolist() for key, value in bootstrap[arm].items()}
    differences = {}
    for arm in by_arm:
        if not arm.startswith('laser'): continue
        baselines = ['f5', 'chatterbox']
        if arm.startswith('laser_long'):
            baselines.append('laser_extension_last')
        for baseline in baselines:
            if baseline in by_arm:
                differences[f'{arm}-minus-{baseline}'] = {key: {
                    'difference': summaries[arm][key] - summaries[baseline][key],
                    'ci95': np.percentile(bootstrap[arm][key] - bootstrap[baseline][key], [2.5, 97.5]).tolist()}
                    for key in ['wer', 'speaker_similarity_ecapa', 'utmos']}
    result = {'manifest_sha256': sha(args.root / 'manifest.json'), 'summaries': summaries,
              'paired_differences': differences, 'limitations': manifest['limitations']}
    long_selection_path = args.root / 'long_selection.json'
    long_selection = json.loads(long_selection_path.read_text()) if long_selection_path.exists() else None
    if long_selection:
        result['long_selection'] = long_selection
        result['limitations'] = [*result['limitations'],
            'This fixed set has informed earlier experiments; the extension is a regression comparison, not a fresh confirmation test.']
    write_json(args.root / 'results.json', result)
    lines = ['# VCTK pretrained-system reference comparison', '',
        '**Training data, compute, model capacity, representation constraints, and speaker conditioning are unmatched. These results compare the tested complete systems and do not isolate LASER versus RVQ or establish a controlled SOTA ranking.**', '',
        ('100 fixed recordings across 100 known speakers. Long-run checkpoints use validation WER selection and the terminal endpoint.'
         if long_selection else '100 fixed recordings across 100 known speakers. LASER checkpoint selection uses validation NLL only.'), '',
        '| System | WER % ↓ | CER % ↓ | ECAPA similarity ↑ | UTMOS ↑ | RTF ↓ | Cap % ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, r in summaries.items():
        speed = f'{r["rtf"]:.3f}' if arm not in ['reference', 'codec'] else '—'
        lines.append(f'| {arm} | {100*r["wer"]:.2f} | {100*r["cer"]:.2f} | {r["speaker_similarity_ecapa"]:.4f} | {r["utmos"]:.3f} | {speed} | {100*r["cap_fraction"]:.1f} |')
    lines += ['', *result['limitations'], '', 'Full per-file results, paired speaker-bootstrap confidence intervals and model hashes are in the artifact.']
    (args.root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    run_path = args.root / 'run.json'
    previous = json.loads(run_path.read_text())['id'] if run_path.exists() else None
    run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=previous, resume='allow' if previous else None,
        name='mdctcodec-tts-vctk100-unmatched-pretrained-references',
        group='mdctcodec-6kbps-long-20260912' if long_selection else 'mdctcodec-matched-scratch-6kbps-rangefix-20260912',
        job_type='tts-benchmark', config={'manifest': manifest, 'manifest_sha256': result['manifest_sha256']}, dir=str(args.root))
    write_json(run_path, {'id': run.id, 'url': run.url})
    run.summary.update({f'{arm}/{key}': value for arm, r in summaries.items() for key, value in r.items() if key != 'ci95'})
    run.summary.update({'status': 'complete', 'items': 100, 'paired_differences': differences,
                        'comparison_type': 'unmatched_pretrained_system_reference',
                        'matched_training_comparison': False})
    run.use_artifact('helloimlixin-rutgers/laser/model-stage2-tts-13z2ta24-selected-checkpoints:epoch-040')
    if long_selection:
        run.use_artifact(long_selection['training_artifact'])
        run.summary['long_selection'] = long_selection
    if 'laser_post_extension' in by_arm:
        selection = json.loads((args.root / 'extension_selection.json').read_text())
        run.use_artifact(f'helloimlixin-rutgers/laser/model-stage2-tts-13z2ta24-selected-checkpoints:epoch-{selection["completed_epochs"]:03d}')
        run.summary['extension_selection'] = selection
    columns = ['system', 'speaker', 'text', 'asr_text', 'wer', 'speaker_similarity_ecapa', 'utmos', 'rtf']
    run.log({'benchmark/per_utterance': wandb.Table(columns=columns, data=[
        [arm, *[r[key] for key in columns[1:]]] for arm, rows in by_arm.items() for r in rows])})
    run.log({'benchmark/listening': wandb.Table(columns=['system', 'speaker', 'text', 'generated', 'reference', 'enrollment'],
        data=[[arm, r['speaker'], r['text'], wandb.Audio(r['audio_path']),
               wandb.Audio(manifest['items'][i]['target']['path']),
               wandb.Audio(manifest['items'][i]['enrollment']['path'])]
              for arm, rows in by_arm.items() if arm not in ['reference', 'codec'] for i, r in enumerate(rows)])})
    # First eight manifest speakers, fixed before synthesis; never select by score.
    media = {}
    training_curve = args.root / 'figures' / 'training_validation.png'
    if long_selection and training_curve.exists():
        media['training/validation_curve'] = wandb.Image(str(training_curve))
    for arm, rows in by_arm.items():
        for i, r in enumerate(rows[:8]):
            media[f'samples/{i}/{arm}'] = wandb.Audio(r['audio_path'], caption=r['text'])
            if arm != 'reference':
                ref, _ = sf.read(manifest['items'][i]['target']['path'], dtype='float32')
                wave, sr = sf.read(r['audio_path'], dtype='float32')
                if sr != 48000: wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 48000).numpy()
                figures = render_tts_preview(ref, wave, args.root / 'figures' / f'{arm}_{i}', title=f'{arm} | {r["text"]}')
                for kind, p in figures.items(): media[f'samples/{i}/{arm}_{kind}'] = wandb.Image(p)
    run.log(media)
    artifact = wandb.Artifact(f'mdctcodec-tts-benchmark-{run.id}', type='audio-evaluation', metadata=result)
    for name in ['manifest.json', 'manifest_sha256.json', 'results.json', 'comparison.md', 'metric_provenance.json']:
        artifact.add_file(str(args.root / name), name=name)
    for name in ['generated', 'scores', 'models', 'figures']:
        artifact.add_dir(str(args.root / name), name=name)
    artifact.add_file(__file__, name='source/benchmark_mdctcodec_tts.py')
    artifact.add_file('src/tts_runtime.py', name='source/tts_runtime.py')
    artifact.add_file('src/models/laser_tts.py', name='source/laser_tts.py')
    for path in ['scripts/tools/run_mdctcodec_tts_benchmark.py', 'scripts/tools/evaluate_tts_after_extension.py',
                 'docs/mdctcodec-tts-benchmark-2026-09-12.md', 'configs/research/mdctcodec_tts_rangefix_continue60.yaml']:
        if Path(path).is_file(): artifact.add_file(path, name='source/' + path)
    artifact.add_file(str(args.root / 'setup/runtime_freeze.txt'), name='runtime_freeze.txt')
    if (args.root / 'extension_selection.json').exists():
        artifact.add_file(str(args.root / 'extension_selection.json'), name='extension_selection.json')
    if long_selection:
        artifact.add_file(str(long_selection_path), name='long_selection.json')
        for path in ['scripts/tools/evaluate_mdctcodec_long.py', 'configs/research/mdctcodec_tts_long.yaml',
                     'src/tts_generation_validation.py', 'scripts/tools/score_tts_generation_validation.py']:
            artifact.add_file(path, name='source/' + path)
    run.log_artifact(artifact, aliases=['latest', '-'.join(by_arm)]).wait()
    run.finish()
    write_json(args.root / 'complete.json', {'run_url': run.url, 'items': 100, 'arms': list(by_arm)})
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'generate', 'score', 'report'], required=True)
    parser.add_argument('--root', type=Path, default=Path('outputs/mdctcodec_tts_benchmark'))
    parser.add_argument('--cache', type=Path, default=Path('outputs/mdctcodec_tts_rangefix/cache/tokens.pt'))
    parser.add_argument('--arm', default='reference')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(); args.root.mkdir(parents=True, exist_ok=True)
    globals()[args.phase](args)


if __name__ == '__main__': main()
