#!/usr/bin/env python3
"""Durably launch the learned complete sparse-code tokenizer pilot."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/church-learned-site-codec-20260911')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--credential-pid', type=int, help='Reuse only the W&B key from an existing Church training process')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    training = output / 'train'
    launch_path = output / 'launch.json'
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        process = Path(f'/proc/{previous["pid"]}/cmdline')
        if process.exists() and b'train_church_learned_site_codec.py' in process.read_bytes():
            raise RuntimeError('The codec training process is already running')
    if training.exists() and not args.resume:
        raise FileExistsError(training)
    if args.resume and not (training / 'last.pt').exists():
        raise FileNotFoundError(training / 'last.pt')
    credential = os.environ.get('WANDB_API_KEY')
    if not credential and args.credential_pid:
        process = Path(f'/proc/{args.credential_pid}')
        command = (process / 'cmdline').read_bytes()
        if not any(name in command for name in (b'train_church_calibrated.py', b'train_church_learned_site_codec.py')):
            raise ValueError('Credential source must be an existing Church training process')
        for item in (process / 'environ').read_bytes().split(b'\0'):
            if item.startswith(b'WANDB_API_KEY='):
                credential = item.split(b'=', 1)[1].decode()
                break
    if not credential:
        credential = getpass.getpass('W&B API key: ')
    if not credential:
        raise ValueError('A W&B credential is required for the production pilot')
    output.mkdir(parents=True, exist_ok=True)
    files = ['scripts/train_church_learned_site_codec.py', 'scripts/launch_church_learned_site_codec.py',
             'src/learned_sparse_site_codec.py', 'src/complete_sparse_codec.py', 'tests/test_learned_sparse_site_codec.py',
             'src/coefficient_history_training.py', 'src/coefficient_pattern_codec.py',
             'scripts/train_official_rqtransformer_laser_stage2.py']
    if not args.resume:
        hashes = {}
        for name in files:
            target = output / 'source-snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        (output / 'source-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    run_id = 'church-learned-complete16k-20260911'
    command = [sys.executable, '-u', str(ROOT / 'scripts/train_church_learned_site_codec.py'),
               '--output', str(training), '--wandb-id', run_id]
    if args.resume:
        command.append('--resume')
    environment = os.environ.copy()
    environment.update({'WANDB_API_KEY': credential, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
        'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
        'TORCH_HOME': '/workspace/tmp/official-rqvae-eval-cache',
        'LASER_VGG16_WEIGHTS': '/workspace/tmp/laser-vgg/vgg16-397923af.pth'})
    with (output / 'train.log').open('ab') as logfile:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
    metadata = {'pid': child.pid, 'gpu': args.gpu, 'command': command, 'started_unix': time.time(),
                'wandb_url': f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}',
                'log': str(output / 'train.log')}
    launch_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
