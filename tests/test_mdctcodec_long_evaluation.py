import json
from pathlib import Path
import pytest

from scripts.tools import evaluate_mdctcodec_long as evaluation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_long_tts_evaluation_uses_validation_wer_and_keeps_terminal_endpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path('outputs/mdctcodec_tts_benchmark')
    for name in ['manifest.json', 'manifest_sha256.json']:
        write(source/name, {'frozen': True})
    for name in ['models', 'setup']:
        (source/name).mkdir()
    for directory in ['generated', 'scores']:
        for arm in ['reference', 'codec', 'laser_extension_last', 'f5', 'chatterbox']:
            write(source/directory/arm/'example.json', {'frozen_baseline': arm})
    training = Path('outputs/mdctcodec_tts_long/stage2')
    for name in ['last.pt', 'low_wer.pt', 'high_wer.pt', 'low_nll.pt']:
        write(training/'checkpoints'/name, {'checkpoint': name})
    write(training/'generation_validation_manifest.json', {'split': 'validation'})
    write(training/'run.json', {'id': 'training_run'})
    write(training/'completion.json', {'status': 'epochs_complete', 'step': 114200,
        'completed_epochs': 160, 'run_url': 'training_url',
        'best': [{'score': 1, 'path': str(training/'checkpoints/low_nll.pt')}],
        'best_generation': [{'score': .3, 'path': str(training/'checkpoints/high_wer.pt')},
                            {'score': .1, 'path': str(training/'checkpoints/low_wer.pt')}]})
    commands = []
    monkeypatch.setattr(evaluation.subprocess, 'run', lambda args, **kwargs: commands.append(args))
    evaluation.tts()
    selection = json.loads(Path('outputs/mdctcodec_tts_long/benchmark/long_selection.json').read_text())
    assert selection['selected_validation_wer'] == .1
    assert selection['checkpoints']['laser_long_best_wer']['source'].endswith('low_wer.pt')
    assert selection['checkpoints']['laser_long_last']['source'].endswith('last.pt')
    assert len(commands) == 5
    assert 'laser_long_best_wer' in commands[-1][-1] and 'laser_long_last' in commands[-1][-1]


def test_codec_evaluation_refuses_unmatched_or_incomplete_budget(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = Path('outputs/mdctcodec_matched_6kbps_long')
    write(root/'laser/completion.json', {'status': 'complete', 'generator_updates': 400000})
    write(root/'rvq/completion.json', {'status': 'paused_budget', 'generator_updates': 390000})
    with pytest.raises(AssertionError):
        evaluation.codec()


def test_queued_evaluator_preserves_already_completed_priority_benchmark(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(Path('outputs/mdctcodec_long_campaign/tts_evaluation_status.json'), {'status': 'complete'})
    write(Path('outputs/mdctcodec_tts_long/benchmark/complete.json'), {'items': 100})
    monkeypatch.setattr(evaluation.sys, 'argv', ['evaluate_mdctcodec_long.py', '--stage', 'tts'])
    def unexpected():
        pytest.fail('An already completed priority benchmark must not run a second time')
    monkeypatch.setattr(evaluation, 'tts', unexpected)
    evaluation.main()
