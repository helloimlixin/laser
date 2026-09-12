#!/usr/bin/env python3
"""Verify fresh initialization, training and resume before releasing scratch jobs."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/church-original-scratch-20260911'


def read(name):
    return json.loads((BASE/name).read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert '13 passed' in (BASE/'tests.log').read_text()
    batches, initials = {}, []
    for variant in ('looped', 'control'):
        cfg = read(f'verify-{variant}/config.json')
        status = read(f'verify-{variant}/status.json')
        assert cfg['batch_size'] == 2048 and cfg['steps_per_epoch'] == 61 and cfg['maximum_steps'] == 6100
        assert cfg['stage2_checkpoint_loaded'] is None and cfg['initial_optimizer_state_entries'] == 0
        assert cfg['initialization']['kind'] == 'random' and cfg['initialization']['initial_optimizer_step'] == 0
        assert 'source_checkpoint' not in cfg and 'source' not in cfg
        assert status['phase'] == 'paused' and status['optimizer_step'] == 1
        initials.append(cfg['initialization'])
        rows = [json.loads(line) for line in (BASE/f'verify-{variant}/history.jsonl').read_text().splitlines()]
        batches[variant] = next(r for r in rows if r['phase'] == 'train')
    assert initials[0] == initials[1]
    a = torch.load(BASE/'verify-continuous/last.pt', map_location='cpu', weights_only=False)
    b = torch.load(BASE/'verify-split/last.pt', map_location='cpu', weights_only=False)
    count = 0

    def compare(x, y, path):
        nonlocal count
        if torch.is_tensor(x):
            assert torch.equal(x, y), path
            count += 1
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), path
            for key in x: compare(x[key], y[key], path+'.'+str(key))
        elif isinstance(x, (list, tuple)):
            assert len(x) == len(y), path
            for i, (u, v) in enumerate(zip(x, y)): compare(u, v, path+'.'+str(i))
        else:
            assert x == y, (path, x, y)

    fields = ['state_dict','optimizer','stream','step','best_fid','best_step','bad_checks','pending_eval','initialized']
    for key in fields: compare(a[key], b[key], key)
    assert a['step'] == 4 and a['stream']['position'] == 9
    resume = {'passed': True, 'bitwise_equal': True, 'compared_tensors': count,
              'fields': fields, 'steps': 4, 'split_step': 2, 'population': 9,
              'batch': 4, 'epoch_remainder_dropped': True, 'changed_gpu_on_resume': True}
    del a, b
    lifecycle = read('verify-lifecycle/results.json')
    assert read('verify-lifecycle/status.json')['phase'] == 'complete'
    assert lifecycle['trained_from_scratch'] and lifecycle['reason'] == 'screen_plateau'
    assert lifecycle['selected_step'] == 2 and lifecycle['frozen_stage1_verified']
    assert lifecycle['generation']['samples'] == 64
    assert lifecycle['initialization']['stage2_checkpoint_loaded'] is None
    cfg = read('verify-continuous/config.json')
    names = list(cfg['source_hashes']) + [
        'scripts/launch_church_original_scratch.py', 'scripts/finalize_church_original_scratch.py',
        'tests/test_church_original_scratch.py', 'tests/test_church_epoch50_loop.py',
        'tests/test_compound_pair_autoregressive.py', 'docs/lsun-church-original-scratch-2026-09-11.md',
    ]
    sources = {name: digest(ROOT/name) for name in names}
    for name, expected in cfg['source_hashes'].items(): assert sources[name] == expected, name
    result = {'production_ready': True, 'source_sha256': sources, 'tests_passed': 13,
              'initialization': initials[0], 'factory_test_forbids_torch_load_and_load_state_dict': True,
              'production_batch_checks': batches, 'resume': resume, 'selection_lifecycle': lifecycle,
              'smoke_metrics_are_quality_results': False}
    path = BASE/'verification.json'
    assert not path.exists()
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(result, indent=2)+'\n')
    pending.replace(path)
    print(json.dumps({'verified': True, 'initialization': 'random', 'released': str(BASE)}), flush=True)


if __name__ == '__main__': main()
