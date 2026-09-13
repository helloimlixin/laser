#!/usr/bin/env python3
"""Evaluate the validation-selected model after the bounded continuation ends."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tools.benchmark_mdctcodec_tts import sha, write_json


def main():
    root = Path('outputs/mdctcodec_tts_benchmark')
    training = Path('outputs/mdctcodec_tts_rangefix')
    launch = json.loads((training / 'continue60_media16_launch.json').read_text())
    status_path = root / 'after_extension_status.json'
    write_json(status_path, {'status': 'waiting_for_training', 'pid': os.getpid(), 'training_pid': launch['pid']})
    while True:
        proc = Path(f'/proc/{launch["pid"]}/stat')
        if not proc.exists() or proc.read_text().split()[2] == 'Z': break
        time.sleep(15)
    complete = json.loads((training / 'stage2/completion.json').read_text())
    if complete['status'] not in ['epochs_complete', 'step_or_time_budget']:
        raise RuntimeError(f'Training did not finish normally: {complete["status"]}')
    while not (root / 'complete.json').exists():
        if (root / 'pipeline_status.json').exists():
            assert json.loads((root / 'pipeline_status.json').read_text())['status'] != 'failed', 'Initial benchmark needs repair'
        time.sleep(15)
    best = min(complete['best'], key=lambda r: r['score'])
    checkpoint = root / 'checkpoints/validation_selected_after_extension.pt'; checkpoint.parent.mkdir(exist_ok=True)
    shutil.copy2(best['path'], checkpoint)
    selection = {'completed_epochs': complete['completed_epochs'], 'final_optimizer_step': complete['step'],
        'selection_metric': 'validation token NLL only', 'score': best['score'], 'source': best['path'],
        'checkpoint': str(checkpoint.resolve()), 'sha256': sha(checkpoint), 'training_run': complete['run_url'],
        'evaluated_utc': datetime.now(timezone.utc).isoformat()}
    write_json(root / 'extension_selection.json', selection)
    old = root / 'generated/laser_epoch40_selected'
    arm = 'laser_post_extension'
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'HF_HOME': '/workspace/tts-benchmark-models', 'HF_HUB_DISABLE_XET': '1'}
    commands = []
    if json.loads((old / 'provenance.json').read_text())['checkpoint_sha256'] == selection['sha256']:
        # Identical selected bytes produce the same paired estimate; no need to
        # rerun inference or pretend the continuation selected a better model.
        for directory in ['generated', 'scores']:
            dest = root / directory / arm; dest.mkdir(parents=True, exist_ok=True)
            for source in (root / directory / 'laser_epoch40_selected').glob('*.json'):
                value = json.loads(source.read_text()); value['arm'] = arm
                value['reused_identical_checkpoint_from'] = 'laser_epoch40_selected'
                write_json(dest / source.name, value)
        selection['reused_identical_checkpoint'] = True
        write_json(root / 'extension_selection.json', selection)
    else:
        commands += [[sys.executable, '-u', 'scripts/tools/benchmark_mdctcodec_tts.py', '--phase', 'generate',
                      '--arm', arm, '--checkpoint', str(checkpoint), '--device', 'cuda:1'],
                     ['/workspace/tts-benchmark-env/bin/python', '-u', 'scripts/tools/benchmark_mdctcodec_tts.py',
                      '--phase', 'score', '--arm', arm, '--device', 'cuda:1']]
    # Fixed-budget endpoints test whether additional training changes free-running
    # speech, even if the best teacher-forced validation checkpoint stays older.
    # Report both endpoints; never choose between them using test scores.
    terminal = root / 'checkpoints/last_after_extension.pt'
    shutil.copy2(training / 'stage2/checkpoints/last.pt', terminal)
    for endpoint_arm, endpoint_path in [
        ('laser_epoch40_last', training / 'epoch40_snapshot/last.pt'),
        ('laser_extension_last', terminal),
    ]:
        commands += [[sys.executable, '-u', 'scripts/tools/benchmark_mdctcodec_tts.py', '--phase', 'generate',
                      '--arm', endpoint_arm, '--checkpoint', str(endpoint_path), '--device', 'cuda:1'],
                     ['/workspace/tts-benchmark-env/bin/python', '-u', 'scripts/tools/benchmark_mdctcodec_tts.py',
                      '--phase', 'score', '--arm', endpoint_arm, '--device', 'cuda:1']]
    shutil.copy2(root / 'results.json', root / 'results_before_extension.json')
    commands.append(['/workspace/tts-benchmark-env/bin/python', '-u', 'scripts/tools/benchmark_mdctcodec_tts.py',
                     '--phase', 'report', '--arm', 'reference,codec,laser_epoch40_selected,laser_epoch40_last,laser_post_extension,laser_extension_last,f5,chatterbox'])
    write_json(status_path, {'status': 'evaluating', 'pid': os.getpid(), 'selection': selection})
    for index, command in enumerate(commands):
        with (root / f'after_extension_{index}.log').open('a') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env)
        if result.returncode: raise RuntimeError(f'Post-extension evaluation failed: {command}')
    write_json(status_path, {'status': 'complete', 'selection': selection})


if __name__ == '__main__':
    try: main()
    except Exception as error:
        write_json('outputs/mdctcodec_tts_benchmark/after_extension_status.json',
                   {'status': 'failed', 'error': str(error), 'pid': os.getpid()})
        raise
