#!/usr/bin/env python3
"""Release the paused-run supervisor only after restoration and training checks."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/church-epoch50-loop-20260911'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((BASE/name).read_text())


def main():
    assert '9 passed' in (BASE/'tests.log').read_text()
    restored = read('source-restoration.json')
    assert restored['passed'] and restored['matches_archived_fid50k_first128_codes']
    resume = read('resume-verification.json')
    assert resume['passed'] and resume['bitwise_equal']
    lifecycle = read('verify-lifecycle/results.json')
    assert read('verify-lifecycle/status.json')['phase'] == 'complete'
    assert lifecycle['reason'] == 'screen_plateau' and lifecycle['selected_step'] == 0
    assert lifecycle['original_checkpoint_retained'] and lifecycle['frozen_stage1_verified']
    assert lifecycle['generation']['samples'] == 64 and lifecycle['generation']['seed'] == 17701
    saved = torch.load(BASE/'verify-lifecycle/best-screen.pt', map_location='cpu', weights_only=False)
    source = torch.load(ROOT/'outputs/lsun-church-bar-20260910/assets/epoch50-source.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, saved['state_dict'][key]) for key, value in source['state_dict'].items())
    assert not saved['state_dict']['head_transformer.loop_gates'].count_nonzero()
    del saved, source
    batches = {}
    for variant in ('control', 'looped'):
        cfg = read(f'verify-batch-{variant}/config.json')
        status = read(f'verify-batch-{variant}/status.json')
        assert cfg['batch_size'] == 2048 and cfg['microbatch'] == 64
        assert status['phase'] == 'paused' and status['optimizer_step'] == 1
        assert cfg['source_sha256'] == restored['source_sha256']
        rows = [json.loads(line) for line in (BASE/f'verify-batch-{variant}/history.jsonl').read_text().splitlines()]
        batches[variant] = next(row for row in rows if row['phase'] == 'train')
    assert all(abs(v) > 0 for v in batches['looped']['train/loop_gates'])
    names = list(read('verify-continuous/config.json')['source_hashes']) + [
        'scripts/launch_church_epoch50_loop.py', 'scripts/verify_church_epoch50_loop.py',
        'scripts/finalize_church_epoch50_loop_verification.py', 'tests/test_church_epoch50_loop.py',
        'tests/test_compound_pair_autoregressive.py', 'docs/lsun-church-epoch50-loop-2026-09-11.md',
    ]
    sources = {name: digest(ROOT/name) for name in names}
    for name, expected in read('verify-continuous/config.json')['source_hashes'].items():
        assert sources[name] == expected, name
    result = {'production_ready': True, 'source_sha256': sources, 'focused_tests_passed': 9,
        'source_restoration': restored, 'resume': resume, 'production_batch_checks': batches,
        'batch_smoke_followup': 'Only the sampler precision label/comment was corrected afterwards; training operations unchanged.',
        'selection_lifecycle': lifecycle, 'selected_source_tensors_verified': True,
        'smoke_fids_are_quality_results': False}
    path = BASE/'verification.json'
    assert not path.exists()
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(path)
    print(json.dumps({'verified': True, 'released': str(BASE)}), flush=True)


if __name__ == '__main__':
    main()
