#!/usr/bin/env python3
"""Verify joint conditional geometry, early decay and fresh training before launch."""
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/church-joint-geometry-20260912'
sys.path.insert(0, str(ROOT))
from scripts.finalize_church_ffhq_archived import digest, equal


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(8)
    deadline = time.monotonic()+900
    needed = [BASE/name/'results.json' for name in ('verify-continuous', 'verify-split')]
    while not all(p.exists() for p in needed):
        if time.monotonic() > deadline:
            raise TimeoutError('Verification did not finish')
        time.sleep(3)
    suite = ET.parse(BASE/'unit-tests.xml').getroot().find('testsuite')
    assert int(suite.attrib['tests']) == 23
    assert int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0
    calibration_path = BASE/'calibration/calibration.json'
    calibration = read(calibration_path)
    sigma = calibration['sigma_cap']
    relative = calibration['selected_relative_sigma']
    assert calibration['fit'][str(relative)]['passes'] and calibration['confirmation'][str(relative)]['passes']
    assert not set(calibration['protocol']['fit_indices']) & set(calibration['protocol']['confirmation_indices'])
    assert calibration['target_code_sha256'] == digest(ROOT/'src/church_relative_noise.py')
    old_path = ROOT/'outputs/church-relative-noise-20260911/relative'
    original = read(old_path/'config.json')
    config = read(BASE/'verify-full/config.json')
    full_status = read(BASE/'verify-full/status.json')
    assert full_status['phase'] == 'paused' and full_status['optimizer_step'] == 2
    assert config['coefficient_noise_sigma_cap'] == sigma
    assert config['relative_sigma'] == relative and config['truncate'] == 3.
    assert config['calibration_sha256'] == digest(calibration_path)
    assert config['initialization'] == original['initialization']
    assert config['stage2_checkpoint_loaded'] is None and config['initial_optimizer_state_entries'] == 0
    excluded = {'output', 'wandb_id', 'stop_after_step', 'fid_samples', 'confirmation_samples',
                'source_hashes', 'objective', 'coeff_targets', 'calibration', 'calibration_sha256',
                'comparison_control', 'intentional_training_change', 'lr_schedule'}
    common = (original.keys() & config.keys()) - excluded
    assert all(original[k] == config[k] for k in common)
    assert original.keys() == config.keys()
    assert 'first 10 epochs' in config['lr_schedule']
    for checked in (config, original):
        for path, expected in checked['source_hashes'].items():
            assert digest(ROOT/path) == expected, path
    status = read(old_path/'status.json')
    cmd = Path(f"/proc/{status['pid']}/cmdline").read_bytes().split(b'\0')
    assert str(ROOT/'scripts/train_church_relative_noise.py').encode() in cmd
    assert str(old_path).encode() in cmd
    assert status['phase'] not in ('paused', 'failed', 'complete')
    print(json.dumps({'phase': 'comparing_resume_states'}), flush=True)
    a, b = [torch.load(BASE/name/'last.pt', map_location='cpu', weights_only=False, mmap=True)
            for name in ('verify-continuous', 'verify-split')]
    assert a['step'] == b['step'] == 6
    tensors = sum(equal(a[k], b[k], k) for k in ('state_dict', 'optimizer', 'stream'))
    for key in ('best_step', 'bad_checks', 'pending_eval', 'initialized'):
        assert a[key] == b[key]
    assert abs(a['best_fid']-b['best_fid']) < 1e-7
    assert a['stream']['epoch'] == 2 and a['stream']['position'] == 513
    results = [read(p) for p in needed]
    assert results[0]['selected_step'] == results[1]['selected_step'] in (2, 4, 6)
    for result in results:
        assert result['trained_from_scratch'] and result['frozen_stage1_verified']
        assert result['generation']['samples'] == 64 and result['generation']['seed'] == 27701
    del a, b
    sources = set(config['source_hashes']) | {
        'scripts/calibrate_church_relative_noise.py', 'scripts/launch_church_joint_geometry.py',
        'scripts/finalize_church_joint_geometry.py', 'scripts/finalize_church_ffhq_archived.py',
        'tests/test_church_relative_noise.py', 'tests/test_church_joint_geometry.py', 'tests/test_church_ffhq_archived.py',
        'docs/lsun-church-joint-geometry-2026-09-12.md', 'src/rqvae_metrics.py',
        'scripts/audit_church_noise_scale.py', 'scripts/audit_church_compound_path.py', 'src/church_coefficient_noise.py',
        'src/models/rqtransformer/configs.py', 'src/models/rqtransformer/primitives.py',
        str(calibration_path.relative_to(ROOT)),
    }
    result = {'production_ready': True, 'unit_tests_passed': 23,
        'sigma_cap': sigma, 'relative_sigma': relative, 'truncate': 3.,
        'joint_candidate_conditioning': True, 'lr_schedule': config['lr_schedule'],
        'calibration_sha256': digest(calibration_path),
        'source_control_pid': status['pid'], 'control_continues_unchanged': True,
        'matched_initialization': config['initialization'], 'full_batch_size': 256,
        'resume_bitwise_equal_tensors': tensors, 'resume_steps': 6,
        'resume_crossed_geometry_activation': True, 'selection_step': results[0]['selected_step'],
        'source_sha256': {name:digest(ROOT/name) for name in sorted(sources)}}
    temporary = BASE/'verification.json.tmp'
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(BASE/'verification.json')
    print(json.dumps({'production_ready': True, 'resume_bitwise_equal_tensors': tensors}), flush=True)


if __name__ == '__main__':
    main()
