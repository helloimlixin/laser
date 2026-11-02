import copy
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.training.depth_loss_pilot import verify_first_batch


spec = importlib.util.spec_from_file_location('depth_loss_trial',
    Path(__file__).resolve().parents[1] / 'scripts/tools/run_joint20k_depth_loss_epoch1.py')
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def recipes():
    control = dict(run_id='control', name='control', lineage={},
        targets=dict(temperature=[.1] * 5), global_batch=2048, lr_schedule_epochs=100,
        sampling=dict(temperature=1., top_k=2048, top_p=.95),
        loss=dict(depth_weights=[1.] * 5))
    weighted = copy.deepcopy(control)
    weighted['run_id'] = 'weighted'
    weighted['loss']['depth_weights'] = [1.5, 1.25, 1., .75, .5]
    return control, weighted


def test_recipe_pair_only_changes_identity_and_loss_weights():
    control, weighted = recipes()
    trial.assert_recipe_pair(control, weighted)
    for section, key, value in [('targets', 'temperature', [.2] * 5),
                                ('sampling', 'top_k', 1024)]:
        changed = copy.deepcopy(weighted)
        changed[section][key] = value
        with pytest.raises(ValueError):
            trial.assert_recipe_pair(control, changed)
    weighted['lr_schedule_epochs'] = 3
    with pytest.raises(ValueError):
        trial.assert_recipe_pair(control, weighted)


def test_preflight_accepts_normalization_roundoff_but_rejects_changed_weights():
    expected = [1.5, 1.25, 1., .75, .5]
    actual = [1.5, 1.25, .9999999999999999, .75, .49999999999999994]
    trial.assert_depth_weights_match(actual, expected)
    for wrong in ([1.] * 5, actual[:-1], [float('nan'), *actual[1:]],
                  [1.5, 1.25, 1., .75, .50001]):
        with pytest.raises(ValueError):
            trial.assert_depth_weights_match(wrong, expected)


def test_preflight_requires_all_ranks_and_restored_adam_age(tmp_path):
    folder = tmp_path / 'private-smoke'
    folder.mkdir()
    verification = dict(passed=True, global_step=627, optimizer_state_count=780,
        all_adam_steps=627, frozen_tokenizer_unchanged=True)
    path = folder / 'verification.json'
    path.write_text(json.dumps(verification))
    weights = [1.5, 1.25, 1., .75, .5]
    for rank in range(8):
        (folder / f'startup-rank{rank}.json').write_text(json.dumps(dict(
            world_size=8, accumulation_steps=4, global_step=626, depth_loss_weights=weights)))
        (folder / f'loss-pairing-rank{rank}.json').write_text(json.dumps(dict(
            passed=True, cuda=True, training_rng_unchanged=True)))
    trial.validate_preflight(tmp_path, weights)
    verification['all_adam_steps'] = 1
    path.write_text(json.dumps(verification))
    with pytest.raises(AssertionError):
        trial.validate_preflight(tmp_path, weights)
    verification['all_adam_steps'] = 627
    path.write_text(json.dumps(verification))
    (folder / 'startup-rank7.json').unlink()
    with pytest.raises(FileNotFoundError):
        trial.validate_preflight(tmp_path, weights)


def test_first_batch_probe_checks_gradients_without_consuming_rng():
    torch.manual_seed(882)
    q = torch.randn(2, 3, 5, 43).softmax(-1)
    codes = torch.randint(43, (2, 3, 5))
    images = torch.randn(2, 3, 8, 8)
    before = torch.get_rng_state().clone()
    proof = verify_first_batch(q, codes, images)
    assert proof['passed'] and proof['training_rng_unchanged']
    assert torch.equal(before, torch.get_rng_state())
    repeated = verify_first_batch(q, codes, images)
    assert proof == repeated


def fixture_report(tmp_path, monkeypatch):
    monkeypatch.setattr(trial, 'OUT', tmp_path)
    arms = {}
    for arm, offset in [('control', 0.), ('weighted', -1.)]:
        folder = tmp_path / arm
        (folder / 'evaluations').mkdir(parents=True)
        (folder / 'train').mkdir()
        arms[arm] = dict(out=str(folder))
        for epoch, fid in [(2, 70.), (3, 60.)]:
            (folder / f'evaluations/epoch{epoch:03d}.json').write_text(json.dumps({
                'train/global_step': 626 * epoch, 'eval/fid': fid + offset,
                'generated_images': 50000, 'real_images': 1281167,
                'real_split': 'train', 'rejection_sampling': False}))
        (folder / 'evaluations/confirmation-seed261101.json').write_text(json.dumps({
            'global_step': 1878, 'seed': 261101, 'generated_images': 50000, 'fid': 60.5 + offset}))
        (folder / 'completed.json').write_text(json.dumps({'global_step': 1878}))
        for rank in range(8):
            (folder / f'train/loss-pairing-rank{rank}.json').write_text(json.dumps({
                'passed': True, 'cuda': True, 'training_rng_unchanged': True,
                'image_fingerprint': str(rank), 'code_fingerprint': 'code', 'target_fingerprint': 'q'}))
    (tmp_path / 'plan.json').write_text(json.dumps({'arms': arms, 'limits': 'short continuation'}))
    return arms


def test_report_compares_both_epochs_and_two_endpoint_generation_seeds(tmp_path, monkeypatch):
    fixture_report(tmp_path, monkeypatch)
    result = trial.report()
    assert result['completed'] and result['screen_passed']
    assert [row['epoch'] for row in result['rows']] == [2, 3, 3]
    assert all(row['improvement'] == 1. for row in result['rows'])
    assert result['initial_images_codes_and_targets_match_all_eight_ranks']
    assert not result['auto_promoted']


def test_completed_arm_can_be_reused_only_with_all_evaluations(tmp_path, monkeypatch):
    arms = fixture_report(tmp_path, monkeypatch)
    assert trial.arm_completed(arms['control'])
    confirmation = Path(arms['control']['out']) / 'evaluations/confirmation-seed261101.json'
    confirmation.unlink()
    with pytest.raises(FileNotFoundError):
        trial.arm_completed(arms['control'])
    (Path(arms['weighted']['out']) / 'completed.json').unlink()
    assert not trial.arm_completed(arms['weighted'])


def test_report_rejects_a_target_mismatch_between_arms(tmp_path, monkeypatch):
    arms = fixture_report(tmp_path, monkeypatch)
    path = Path(arms['weighted']['out']) / 'train/loss-pairing-rank7.json'
    proof = json.loads(path.read_text())
    proof['target_fingerprint'] = 'different-noise'
    path.write_text(json.dumps(proof))
    with pytest.raises(AssertionError):
        trial.report()
