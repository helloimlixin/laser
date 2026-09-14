#!/usr/bin/env python3
"""Freeze a LASER/RVQ TTS protocol, then verify aligned caches before training."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch
import yaml

from src.models.laser_tts import LaserTTS, TTSConfig
from src.tts_pairing import common_state, file_sha, record_signature, state_sha


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Refusing to change frozen {path}'
    else:
        path.write_text(encoded)


def fresh_targets(records, excluded):
    """Deterministic bipartite matching: one distinct text for every speaker."""
    candidates = defaultdict(list)
    for r in records:
        if r['split'] == 'test' and r['text_key'] not in excluded:
            candidates[r['speaker']].append(r)
    for items in candidates.values():
        items.sort(key=lambda r: hashlib.sha256(f'20260913:paired-test:{r["path"]}'.encode()).hexdigest())
    owners, chosen = {}, {}
    def assign(speaker, seen):
        for record in candidates[speaker]:
            key = record['text_key']
            if key in seen: continue
            seen.add(key)
            if key not in owners or assign(owners[key], seen):
                owners[key] = speaker; chosen[speaker] = record
                return True
        return False
    for speaker in sorted({r['speaker'] for r in records}):
        if not assign(speaker, set()):
            raise ValueError(f'Cannot assign an unused distinct test text to {speaker}')
    return [chosen[speaker] for speaker in sorted(chosen)]


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / 'plan.json'
    if plan_path.exists():
        print('Already frozen:', plan_path); return
    cache_path = Path('outputs/mdctcodec_tts_rangefix/cache/tokens.pt').resolve()
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    original = Path('outputs/mdctcodec_matched_6kbps_rangefix')
    stage1_protocol = json.loads((original/'protocol.json').read_text())
    assert stage1_protocol['generator_updates'] == 200000
    completions = {arm: json.loads((original/arm/'completion.json').read_text()) for arm in ('laser', 'rvq')}
    assert all(c['generator_updates'] == 200000 and c['status'] == 'complete' for c in completions.values())
    old_benchmark = Path('outputs/mdctcodec_tts_benchmark/manifest.json')
    old = json.loads(old_benchmark.read_text())
    excluded = {r['target']['text_key'] for r in old['items']}
    clean = lambda r: {k: v for k, v in r.items() if k != 'codes'}
    rows = []
    for r in fresh_targets(cache['records'], excluded):
        enrollment = min((x for x in cache['records'] if x['split'] == 'train' and
                          x['speaker'] == r['speaker'] and 3 <= x['seconds'] <= 10),
                         key=lambda x: (abs(x['seconds'] - 8), x['path']))
        row = dict(id=Path(r['path']).stem, speaker=r['speaker'], target=clean(r), enrollment=clean(enrollment),
                   seed=20260913 + len(rows))
        for part in ('target', 'enrollment'): row[part]['sha256'] = file_sha(row[part]['path'])
        rows.append(row)
    assert len(rows) == len({r['target']['text_key'] for r in rows}) == 100
    manifest = {'version': 2, 'items': rows, 'seed': 20260913,
        'selection': 'One unique normalized test text per known speaker; all cached 0.5–12 second utterances eligible; excludes all previous benchmark target texts.',
        'excluded_previous_manifest_sha256': file_sha(old_benchmark),
        'conditioning': 'Both priors use the same learned speaker IDs and phonemes. Enrollment audio is used only for ECAPA scoring.',
        'generation': {'temperature': .8, 'top_k': 50, 'max_frames': 2250, 'seeds': '20260913 + index'},
        'metrics': old['metrics'],
        'limitations': ['One paired seed; shared prior backbone and update budget, vocabulary-specific parameters differ.',
            'Known-speaker TTS. The frozen codecs may have seen target audio; stage2 target texts are disjoint from stage2 training and validation.',
            'Fresh evaluation excludes previously benchmarked normalized target texts; historical tuning informed the shared recipe.',
            'UTMOS is a learned proxy, not human MOS. This is a controlled codec/prior comparison, not a general TTS SOTA benchmark.']}
    write(root/'benchmark/manifest.json', manifest)
    model_options = dict(width=512, heads=8, text_layers=3, audio_layers=8, depth_layers=2, dropout=.1)
    arms, shared = {}, None
    torch.set_num_threads(4)
    for arm in ('laser', 'rvq'):
        cfg = TTSConfig(phone_vocab=max(cache['phone_to_id'].values()) + 1,
                        speakers=len(cache['speaker_to_id']), codec=arm, **model_options)
        torch.manual_seed(20260913)
        model = LaserTTS(cfg)
        if shared is None: shared = {k: v.clone() for k, v in common_state(model).items()}
        else:
            incompatible = model.load_state_dict(shared, strict=False)
            assert not incompatible.unexpected_keys
            assert all(k.startswith(('fields.', 'heads.')) for k in incompatible.missing_keys)
        assert state_sha(common_state(model)) == state_sha(shared)
        initial = (root/f'{arm}_initial.pt').resolve(); torch.save(model.state_dict(), initial)
        codec_checkpoint = Path(completions[arm]['best_checkpoint']).resolve()
        if arm == 'laser': assert file_sha(codec_checkpoint) == cache['codec_sha256']
        arms[arm] = {'initialization': str(initial), 'initialization_sha256': file_sha(initial),
            'cache': str(cache_path if arm == 'laser' else (root/'rvq_cache_gpu/tokens.pt').resolve()),
            'codec_checkpoint': str(codec_checkpoint), 'codec_sha256': file_sha(codec_checkpoint),
            'stage1_training_budget': 200000, 'stage1_checkpoint_selection': 'Highest validation ViSQOL within completed matched 200k-update experiment',
            'stage1_validation_visqol': completions[arm]['best_validation_visqol'],
            'model_config': vars(cfg), 'parameters': sum(p.numel() for p in model.parameters()),
            'common_parameters': sum(v.numel() for v in shared.values()),
            'vocab_sizes_without_eos': [8192,127,8192,127] if arm == 'laser' else [1024]*4}
        del model
    train = dict(epochs=160, max_steps=120000, max_hours=16, frame_budget=8192, max_batch=16,
        accumulate=4, learning_rate=.0003, warmup_steps=1000, min_lr_ratio=1/30, weight_decay=.01,
        gradient_clip=1., guided_attention_weight=.2, guided_attention_steps=8000, num_workers=4,
        save_every_steps=250, upload_every_epochs=5, preview_every_epochs=5, preview_count=16,
        preview_max_frames=1500, validation_items=256, evaluate_untrained=False, preview_first_epoch=False,
        generation_validation=dict(items=64, every_epochs=10, max_frames=2250,
            python='/workspace/tts-benchmark-env/bin/python',
            asr_model_record='outputs/mdctcodec_tts_benchmark/models/Systran--faster-whisper-large-v3.json'))
    plan = {'version': 1, 'name': 'mdctcodec-6kbps-paired-rqtransformer-20260913', 'seed': 20260913,
        'arms': arms, 'common_initialization_sha256': state_sha(shared), 'record_signature': record_signature(cache),
        'stage1_protocol_sha256': file_sha(original/'protocol.json'),
        'benchmark_manifest_sha256': file_sha(root/'benchmark/manifest.json'),
        'train': train, 'frame_rate': 150, 'bits_per_frame': 40, 'nominal_kbps': 6.,
        'cache_counts': cache['counts'],
        'cache_encoding': 'Both production token caches use CUDA FP32 encoding, matmul TF32 disabled and cuDNN TF32 enabled. CPU RVQ shards are preflight diagnostics only.',
        'train_audio_hours': sum(r['seconds'] for r in cache['records'] if r['split']=='train')/3600,
        'prior': 'Shared text and temporal transformer; two-layer causal depth transformer over four categorical fields. No codec re-training or bitrate change.',
        'laser_representation': 'atom1, signed-coefficient1+63, atom2, signed-coefficient2+63; OMP selection order. Pair joint ID is atom*127+coefficient_code (1,040,384 legal values). Prediction is factorized, without a million-way head.',
        'selection': 'Primary: lowest validation64 Whisper WER among every-10-completed-epoch candidates, earliest on ties. Secondary: fixed epoch160 endpoint. Never select using test scores or cross-codec token NLL.',
        'checkpoint_upload': 'Online every5 completed epochs: latest optimizer state, best3 validation WER, best3 within-arm NLL; stage1 retains its separate top3 ViSQOL uploads.',
        'compute_budget_gpu_hours': 24, 'budget_scope': 'New paired stage2 campaign only: assigned GPU job wall time, including preparation on GPU, validation, and benchmark; previous codec campaign retains its own ceiling. CPU cache preparation consumes no assigned GPU time.',
        'budget_incomplete_policy': 'Do not publish a matched final result unless both finish160 epochs with exactly matching audited updates and data chains.',
        'limitations': manifest['limitations'], 'source_rqtransformer': 'https://arxiv.org/abs/2203.01941'}
    write(plan_path, plan)
    print(json.dumps({'prepared': str(plan_path), 'parameters': {a:x['parameters'] for a,x in arms.items()},
        'common_parameters': arms['laser']['common_parameters'], 'test_items': len(rows),
        'distinct_test_texts': len({r['target']['text_key'] for r in rows})}, indent=2), flush=True)


def finalize(root):
    plan = json.loads((root/'plan.json').read_text())
    for arm, item in plan['arms'].items():
        cache = torch.load(item['cache'], map_location='cpu', weights_only=True)
        assert record_signature(cache,include_frames=plan.get('record_signature_basis')!='utterance_metadata') == plan['record_signature'], f'{arm}: mismatched records/lengths'
        assert cache['codec_sha256'] == item['codec_sha256']
        item['cache_sha256'] = file_sha(item['cache'])
        del cache
    write(root/'protocol.json', plan)
    for arm, item in plan['arms'].items():
        cfg = {'cache': item['cache'], 'output': str((root/arm).resolve()), 'seed': plan['seed'],
            'device': 'cuda:0', 'evaluation_device': 'cuda:0',
            'model': {k:v for k,v in item['model_config'].items() if k not in ('speakers','phone_vocab')},
            'train': plan['train'], 'media': {'attention_cmap': 'Blues'},
            'paired': {'protocol': str((root/'protocol.json').resolve()), 'protocol_sha256': file_sha(root/'protocol.json')},
            'wandb': {'entity': 'helloimlixin-rutgers', 'project': 'laser', 'group': plan['name'],
                      'name': f'mdctcodec-{arm}-6kbps-paired-rqtransformer-160ep'}}
        destination = root/f'{arm}.yaml'
        if destination.exists(): assert yaml.safe_load(destination.read_text()) == cfg
        else: destination.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print('PAIRED_CACHES_VERIFIED', file_sha(root/'protocol.json'), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/mdctcodec_tts_paired'))
    p.add_argument('--finalize', action='store_true')
    args=p.parse_args()
    (finalize if args.finalize else prepare)(args.root)
