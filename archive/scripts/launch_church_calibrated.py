#!/usr/bin/env python3
"""Launch matched calibrated-target and hard-target Church experiments."""
import argparse
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/church-calibrated-20260911')
    p.add_argument('--arms', nargs='+', choices=['soft', 'hard'], default=['soft', 'hard'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wandb-key-stdin', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    calibration = output / 'calibration/calibration.json'
    fitted = json.loads(calibration.read_text())
    assert fitted['fit_split'] == 'train' and fitted['selected_sigma'] > 0
    for arm in args.arms:
        directory = output / arm
        if directory.exists() and not args.resume:
            raise FileExistsError(directory)
        if (directory / 'status.json').exists():
            status = json.loads((directory / 'status.json').read_text())
            command = Path(f"/proc/{status['pid']}/cmdline")
            if command.exists() and str(directory).encode() in command.read_bytes():
                raise RuntimeError(f'{arm} is already active')
    environment = os.environ.copy()
    if args.wandb_key_stdin:
        environment['WANDB_API_KEY'] = getpass.getpass('W&B API key: ')
    if not environment.get('WANDB_API_KEY'):
        raise RuntimeError('Supply WANDB_API_KEY or --wandb-key-stdin')
    files = ['scripts/train_church_calibrated.py', 'scripts/launch_church_calibrated.py',
        'scripts/calibrate_church_coeff_targets.py', 'src/church_calibrated_training.py',
        'src/church_ffhq_recipe.py', 'src/coefficient_history_training.py',
        'scripts/train_church_ffhq_recipe.py', 'scripts/train_official_rqtransformer_laser_stage2.py',
        'src/models/rqtransformer/configs.py', 'src/models/rqtransformer/transformers.py',
        'src/models/rqtransformer/attentions.py', 'tests/test_church_calibrated_training.py',
        'src/rqvae_metrics.py']
    hashes = {}
    snapshot = output / 'source-snapshot'
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and args.resume and sha256_file(target) != sha256_file(ROOT / name):
            raise RuntimeError(f'Source changed since launch: {name}')
        shutil.copy2(ROOT / name, target)
        hashes[name] = sha256_file(target)
    (snapshot / 'sha256.json').write_text(json.dumps(hashes, indent=2))
    for arm in args.arms:
        gpu = 0 if arm == 'soft' else 1
        run_id = f'church-calibrated-{arm}-aug219m-20260911'
        command = [sys.executable, '-u', str(ROOT / 'scripts/train_church_calibrated.py'),
                   '--target-mode', arm, '--output', str(output / arm),
                   '--calibration', str(calibration), '--wandb-id', run_id]
        if args.resume:
            command.append('--resume')
        env = {**environment, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '8',
               'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
               'TORCH_HOME': '/workspace/tmp/official-rqvae-eval-cache', 'PYTHONUNBUFFERED': '1'}
        logfile = output / f'{arm}.log'
        with logfile.open('ab') as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        entry = {'arm': arm, 'pid': process.pid, 'gpu': gpu, 'command': command,
                 'started_unix': time.time(), 'log': str(logfile),
                 'wandb_url': f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}'}
        (output / f'launch-{arm}.json').write_text(json.dumps(entry, indent=2))
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
