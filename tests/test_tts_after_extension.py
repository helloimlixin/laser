import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.tools import evaluate_tts_after_extension as evaluation


@pytest.mark.parametrize('identical_selected_model', [False, True])
def test_followup_reports_fixed_endpoints_and_never_selects_using_test_scores(tmp_path, monkeypatch, identical_selected_model):
    monkeypatch.chdir(tmp_path)
    root = Path('outputs/mdctcodec_tts_benchmark')
    training = Path('outputs/mdctcodec_tts_rangefix')
    write = evaluation.write_json
    write(training / 'continue60_media16_launch.json', {'pid': 2147483647})
    (training / 'stage2/checkpoints').mkdir(parents=True)
    (training / 'epoch40_snapshot').mkdir()
    (training / 'stage2/checkpoints/last.pt').write_bytes(b'final optimizer state')
    (training / 'epoch40_snapshot/last.pt').write_bytes(b'epoch40 optimizer state')
    selected = training / 'stage2/checkpoints/best.pt'; selected.write_bytes(b'validation selected model')
    write(training / 'stage2/completion.json', {'status': 'epochs_complete', 'completed_epochs': 60,
        'step': 42819, 'run_url': 'test-run', 'best': [{'score': 3., 'path': str(selected)}]})
    write(root / 'complete.json', {'arms': ['laser_epoch40_selected']})
    write(root / 'results.json', {'keep': 'original measured results'})
    original = root / 'generated/laser_epoch40_selected'
    write(original / 'provenance.json', {'checkpoint_sha256': evaluation.sha(selected) if identical_selected_model else 'older hash'})
    write(original / 'sample.json', {'arm': 'laser_epoch40_selected', 'audio_path': 'unchanged.wav'})
    write(root / 'scores/laser_epoch40_selected/sample.json', {'arm': 'laser_epoch40_selected', 'wer': .4})
    commands = []
    monkeypatch.setattr(evaluation.subprocess, 'run', lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    evaluation.main()
    generated = [command[command.index('--arm') + 1] for command in commands
                 if '--phase' in command and command[command.index('--phase') + 1] == 'generate']
    assert generated == ([] if identical_selected_model else ['laser_post_extension']) + ['laser_epoch40_last', 'laser_extension_last']
    assert 'laser_epoch40_last' in commands[-1][-1] and 'laser_extension_last' in commands[-1][-1]
    assert json.loads((root / 'results_before_extension.json').read_text()) == {'keep': 'original measured results'}
    assert (root / 'checkpoints/last_after_extension.pt').read_bytes() == b'final optimizer state'
    assert json.loads((root / 'after_extension_status.json').read_text())['status'] == 'complete'
