#!/usr/bin/env python3
"""Benchmark the completed larger budgets with frozen prior comparison inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tts():
    source = Path('outputs/mdctcodec_tts_benchmark')
    root = Path('outputs/mdctcodec_tts_long/benchmark'); root.mkdir(parents=True, exist_ok=True)
    training = Path('outputs/mdctcodec_tts_long/stage2')
    completion = json.loads((training/'completion.json').read_text())
    assert completion['status'] in ['epochs_complete', 'step_or_time_budget'], completion
    assert completion['step'] > 42821
    selected = min(completion['best_generation'], key=lambda r: r['score'])
    checkpoints = root / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    choices = {'laser_long_best_wer': Path(selected['path']), 'laser_long_last': training/'checkpoints/last.pt'}
    selection = {'selection_metric': 'Whisper-large-v3 corpus WER on 64 fixed validation prompts only',
        'validation_manifest_sha256': sha(training/'generation_validation_manifest.json'),
        'selected_validation_wer': selected['score'], 'selected_epoch': selected.get('epoch'),
        'selected_step': selected.get('step'), 'completed_epochs': completion['completed_epochs'],
        'terminal_step': completion['step'], 'training_run': completion['run_url'], 'checkpoints': {},
        'training_artifact': 'helloimlixin-rutgers/laser/model-stage2-tts-'+
            json.loads((training/'run.json').read_text())['id']+'-selected-checkpoints:latest'}
    priority_record = Path('outputs/mdctcodec_long_campaign/priority_tts_evaluation.json')
    if priority_record.exists():
        selection['timing_environment'] = json.loads(priority_record.read_text())['timing_scope']
    for arm, path in choices.items():
        target = checkpoints / f'{arm}.pt'
        if target.exists(): assert sha(target) == sha(path)
        else: shutil.copy2(path, target)
        selection['checkpoints'][arm] = {'source': str(path.resolve()), 'snapshot': str(target.resolve()), 'sha256': sha(target)}
    (root/'long_selection.json').write_text(json.dumps(selection, indent=2))
    for name in ['manifest.json', 'manifest_sha256.json']:
        if (root/name).exists(): assert sha(root/name) == sha(source/name)
        else: shutil.copy2(source/name, root/name)
    for name in ['models', 'setup']:
        shutil.copytree(source/name, root/name, dirs_exist_ok=True)
    baselines = ['reference', 'codec', 'laser_extension_last', 'f5', 'chatterbox']
    for directory in ['generated', 'scores']:
        for arm in baselines:
            shutil.copytree(source/directory/arm, root/directory/arm, dirs_exist_ok=True)
    base = ['scripts/tools/benchmark_mdctcodec_tts.py', '--root', str(root), '--device', 'cuda:0']
    metric_python = '/workspace/tts-benchmark-env/bin/python'
    for arm, record in selection['checkpoints'].items():
        subprocess.run([sys.executable, '-u', *base, '--phase', 'generate', '--arm', arm,
                        '--checkpoint', record['snapshot']], check=True)
        subprocess.run([metric_python, '-u', *base, '--phase', 'score', '--arm', arm], check=True)
    subprocess.run([metric_python, '-u', *base, '--phase', 'report', '--arm',
                    ','.join([*baselines, *choices])], check=True)


def codec():
    root = Path('outputs/mdctcodec_matched_6kbps_long')
    for arm in ['laser','rvq']:
        record = json.loads((root/arm/'completion.json').read_text())
        assert record['status'] == 'complete' and record['generator_updates'] == 400000
    subprocess.run([sys.executable, '-u', 'archive/scripts/report_mdctcodec_matched.py',
                    '--root', str(root), '--device', 'cuda:0'], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['tts', 'codec'], required=True)
    args = p.parse_args()
    root = Path('outputs/mdctcodec_long_campaign')
    import fcntl
    lock = (root / f'{args.stage}_evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    status = root / f'{args.stage}_evaluation_status.json'
    completed = (Path('outputs/mdctcodec_tts_long/benchmark/complete.json') if args.stage == 'tts'
                 else Path('outputs/mdctcodec_matched_6kbps_long/comparison_complete.json'))
    if status.exists() and json.loads(status.read_text())['status'] == 'complete' and completed.exists():
        print('Evaluation already complete; preserving the frozen report.', flush=True)
        return
    status.write_text(json.dumps({'status': 'evaluating'}))
    try:
        globals()[args.stage]()
    except Exception as error:
        status.write_text(json.dumps({'status': 'failed', 'error': str(error)}, indent=2))
        raise
    status.write_text(json.dumps({'status': 'complete'}))


if __name__ == '__main__':
    main()
