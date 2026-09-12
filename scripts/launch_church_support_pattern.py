#!/usr/bin/env python3
"""Launch the verified support-first Church prior with a calibrated pattern table."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'outputs/church-support-pattern-integer-20260911')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--credential-pid', type=int)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    calibration = json.loads((output/'results.json').read_text())
    checked = json.loads((output/'verification.json').read_text())
    if not checked['production_ready']:
        raise RuntimeError('Production verification has not passed')
    book = Path(calibration['selected_codebook'])
    if digest(book) != checked['codebook_sha256']:
        raise RuntimeError('Codebook changed since verification')
    for name, expected in checked['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f'Source changed since verification: {name}')
    training = output/'train'
    if training.exists() and not args.resume:
        raise FileExistsError(training)
    if args.resume and not (training/'last.pt').exists():
        raise FileNotFoundError(training/'last.pt')
    launch_path = output/'launch.json'
    if launch_path.exists():
        previous = json.loads(launch_path.read_text())
        path = Path(f'/proc/{previous["pid"]}/cmdline')
        if path.exists() and b'train_church_support_pattern.py' in path.read_bytes():
            raise RuntimeError('The support-pattern prior is already active')
    credential = os.environ.get('WANDB_API_KEY')
    if not credential and args.credential_pid:
        process = Path(f'/proc/{args.credential_pid}')
        command = (process/'cmdline').read_bytes().split(b'\0')
        allowed = (b'/train_church_calibrated.py', b'/train_church_support_pattern.py')
        if not any(any(part.endswith(name) for name in allowed) for part in command):
            raise RuntimeError('Credential source must be an existing Church training process')
        for item in (process/'environ').read_bytes().split(b'\0'):
            if item.startswith(b'WANDB_API_KEY='):
                credential = item.split(b'=',1)[1].decode()
                break
    if not credential:
        raise RuntimeError('Provide WANDB_API_KEY or an authorized Church credential PID')
    for name in checked['source_sha256']:
        target = output/'source-snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if args.resume and (not target.exists() or digest(target) != checked['source_sha256'][name]):
            raise RuntimeError(f'Snapshot changed since launch: {name}')
        if not args.resume:
            shutil.copy2(ROOT/name,target)
    run_id = f'church-support-pattern{calibration["selected_vocabulary"]}-201m-20260911'
    command = [sys.executable,'-u',str(ROOT/'scripts/train_church_support_pattern.py'),
        '--output',str(training),'--calibration',str(output/'results.json'),
        '--augmentation-check',str(output/'augmentation-check.json'),'--wandb-id',run_id]
    if args.resume:
        command.append('--resume')
    environment = {**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':str(args.gpu),
        'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
        'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache'}
    with (output/'train.log').open('ab') as logfile:
        child = subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
            stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
    metadata = {'pid':child.pid,'gpu':args.gpu,'command':command,'started_unix':time.time(),
        'wandb_url':f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}',
        'log':str(output/'train.log'),'codebook_sha256':digest(book)}
    launch_path.write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
