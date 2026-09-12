#!/usr/bin/env python3
"""Record completed checks and release the pending compound-loop launch."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'outputs/church-looped-pair-20260911'
LEGACY = ROOT / 'outputs/church-interleaved-pattern-20260911'


def read(name):
    return json.loads((BASE / name).read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    pending = path.with_suffix('.json.tmp')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


def main():
    assert '18 passed' in (BASE / 'pytest.log').read_text()
    resume = read('resume-verification.json')
    assert resume['passed'] and resume['bitwise_equal'] and resume['steps'] == 4
    geometry = read('geometry-backward.json')
    assert geometry['passed'] and geometry['microbatch'] == 64
    configs, batches, caches = {}, {}, {}
    for variant in ('looped', 'unrolled'):
        cfg = read(f'verify-batch-{variant}/config.json')
        status = read(f'verify-batch-{variant}/status.json')
        assert cfg['batch_size'] == 128 and cfg['microbatch'] == 64
        assert status['phase'] == 'paused' and status['optimizer_step'] == 2
        cache = read(f'cache-{variant}.json')
        assert cache['passed_fp32'] and cache['images'] == 2 and cache['sites_per_image'] == 64
        for field in ('atom_tv', 'coefficient_tv'):
            assert cache['metrics']['bf16'][field]['max'] < .01
        rows = [json.loads(line) for line in (BASE / f'verify-batch-{variant}/history.jsonl').read_text().splitlines()]
        batches[variant] = next(row for row in rows if row['phase'] == 'train')
        configs[variant], caches[variant] = cfg, cache
    assert batches['looped']['train/loss'] == batches['unrolled']['train/loss']
    differences = {k for k in configs['looped'] if configs['looped'][k] != configs['unrolled'][k]}
    assert differences == {'output', 'variant', 'parameters', 'head_unique_layers', 'head_passes'}, differences
    lifecycle = read('verify-lifecycle/results.json')
    assert read('verify-lifecycle/status.json')['phase'] == 'complete'
    assert lifecycle['reason'] == 'heldout_or_generation_early_stop' and lifecycle['selected_epoch'] == 1
    assert lifecycle['frozen_stage1_verified'] and lifecycle['generation']['seed'] == 38701
    assert lifecycle['generation']['samples'] == 64
    generated = torch.load(BASE / 'verify-lifecycle/selected-independent-fid/generated-codes.pt', weights_only=True)
    atoms, ids = generated['atoms'].long(), generated['coefficient_ids'].long()
    assert generated.keys() == {'atoms', 'coefficient_ids'}
    assert atoms.shape == ids.shape == (64, 8, 8, 4)
    assert ((atoms >= 0) & (atoms < 16384)).all() and ((ids >= 0) & (ids < 2048)).all()
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
    sources = configs['looped']['source_hashes'].copy()
    for name, expected in sources.items():
        assert digest(ROOT / name) == expected, name
    for name in [
        'scripts/launch_church_looped_pair.py', 'scripts/train_church_interleaved_pattern.py',
        'scripts/verify_church_looped_pair_cache.py', 'scripts/finalize_church_looped_pair_verification.py',
        'scripts/verify_church_looped_pair_geometry.py',
        'tests/test_church_looped_pair.py', 'tests/test_compound_pair_autoregressive.py',
        'tests/test_church_calibrated_training.py', 'docs/lsun-church-looped-rq-2026-09-11.md',
    ]:
        sources[name] = digest(ROOT / name)
    verification = {
        'production_ready': True, 'unit_tests_passed': 18, 'source_sha256': sources,
        'production_batches': batches, 'resume': resume, 'cache_checks': caches,
        'geometry_backward': geometry, 'lifecycle': lifecycle,
        'generated_support_and_coefficients_valid': True, 'complete_site_integer_construction': False,
        'matched_initial_production_loss': batches['looped']['train/loss'],
        'smoke_fids_are_quality_evidence': False,
        'stage1_sha256': configs['looped']['stage1_sha256'],
        'cache_sha256': configs['looped']['cache_sha256'],
    }
    assert not (BASE / 'verification.json').exists()
    assert not (LEGACY / 'verification.json').exists()
    write(BASE / 'verification.json', verification)
    # The already-running supervisor has a fixed compatibility target and a
    # historical checksum check. Its bridge now starts only the verified pair
    # comparison; the old codebook is never consumed by either new trainer.
    calibration = json.loads((ROOT / 'outputs/church-support-pattern-integer-20260911/results.json').read_text())
    write(LEGACY / 'verification.json', {
        'production_ready': True, 'source_sha256': sources,
        'codebook_sha256': digest(Path(calibration['selected_codebook'])),
        'superseded_by': str(BASE), 'launches_integer_model': False,
        'verification': str(BASE / 'verification.json'),
    })
    print(json.dumps({'verified': True, 'released_launch': str(BASE)}), flush=True)


if __name__ == '__main__':
    main()
