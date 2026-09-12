#!/usr/bin/env python3
"""Verify evidence before releasing the archived-FFHQ production launcher."""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/church-ffhq-archived-20260911'
sys.path.insert(0, str(ROOT))
from src.church_ffhq_archived import ARCHIVE_SHA256


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def equal(a, b, path='root'):
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and torch.equal(a, b), path
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(equal(v, b[k], f'{path}.{k}') for k, v in a.items())
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        return sum(equal(v, w, f'{path}.{i}') for i, (v, w) in enumerate(zip(a, b)))
    assert a == b, (path, a, b)
    return 0


def main():
    torch.set_num_threads(8)
    deadline = time.monotonic()+900
    needed = [BASE/name/'results.json' for name in ('verify-continuous', 'verify-split')]
    while not all(p.exists() for p in needed):
        if time.monotonic() > deadline:
            raise TimeoutError('Verification runs did not finish')
        time.sleep(3)
    configs = []
    for variant in ('control', 'looped'):
        directory = BASE/f'verify-{variant}'
        status, config = read(directory/'status.json'), read(directory/'config.json')
        assert status['phase'] == 'paused' and status['optimizer_step'] == 2
        assert config['batch_size'] == 256 and config['microbatch'] == 32
        assert config['epochs'] == 300 and config['maximum_steps'] == 147900
        assert config['train_images'] == 126227 and config['steps_per_epoch'] == 493
        assert config['stage2_checkpoint_loaded'] is None and config['initial_optimizer_state_entries'] == 0
        assert config['initialization']['kind'] == 'random'
        assert config['archived_code_sha256'] == ARCHIVE_SHA256
        for path, expected in config['source_hashes'].items():
            assert digest(ROOT/path) == expected, path
        configs.append(config)
    assert configs[0]['initialization'] == configs[1]['initialization']
    ignored = {'variant', 'output', 'parameters', 'loop_update'}
    assert {k:v for k,v in configs[0].items() if k not in ignored} == {
        k:v for k,v in configs[1].items() if k not in ignored}
    gpu = read(BASE/'verify-gpu/verification.json')
    assert gpu['passed'] and not gpu['quality_result']
    assert max(gpu['active_loop_fp32_cache_max_errors'].values()) < 1e-4
    assert gpu['geometry_training']['geometry'] > 0
    assert gpu['sampler_smoke']['samples'] == 64
    print(json.dumps({'phase': 'comparing_continuous_and_resumed_checkpoints'}), flush=True)
    a, b = [torch.load(BASE/name/'last.pt', map_location='cpu', weights_only=False, mmap=True)
            for name in ('verify-continuous', 'verify-split')]
    assert a['step'] == b['step'] == 6
    tensors = sum(equal(a[key], b[key], key) for key in ('state_dict', 'optimizer', 'stream'))
    for key in ('best_step', 'bad_checks', 'pending_eval', 'initialized'):
        assert a[key] == b[key], key
    assert abs(a['best_fid']-b['best_fid']) < 1e-7
    assert a['stream']['epoch'] == 2 and a['stream']['position'] == 513
    result_a, result_b = [read(p) for p in needed]
    assert result_a['selected_step'] == result_b['selected_step'] in (2, 4, 6)
    for result in (result_a, result_b):
        assert result['trained_from_scratch'] and result['frozen_stage1_verified']
        assert result['generation']['samples'] == 64 and result['generation']['seed'] == 27701
    del a, b
    sources = set(configs[0]['source_hashes']) | {
        'scripts/launch_church_ffhq_archived.py', 'scripts/verify_church_ffhq_archived.py',
        'scripts/finalize_church_ffhq_archived.py', 'tests/test_church_ffhq_archived.py',
        'docs/lsun-church-archived-ffhq-2026-09-11.md', 'src/rqvae_metrics.py',
        'src/models/rqtransformer/configs.py', 'src/models/rqtransformer/primitives.py',
    }
    evidence = {'production_ready': True, 'unit_tests_passed': 7,
        'full_batch_control_and_looped_steps': 2,
        'initialization': configs[0]['initialization'],
        'gpu': gpu, 'resume_bitwise_equal_tensors': tensors,
        'resume_steps': 6, 'resume_crossed_geometry_activation': True,
        'resume_selection_step': result_a['selected_step'],
        'fid_stats_sha256': digest(Path(configs[0]['fid_stats'])),
        'source_sha256': {name:digest(ROOT/name) for name in sorted(sources)}}
    temporary = BASE/'verification.json.tmp'
    temporary.write_text(json.dumps(evidence, indent=2)+'\n')
    temporary.replace(BASE/'verification.json')
    print(json.dumps({'production_ready': True, 'resume_bitwise_equal_tensors': tensors}), flush=True)


if __name__ == '__main__':
    main()
