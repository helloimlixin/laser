#!/usr/bin/env python3
"""Launch the matched Church capacity comparison, one persistent process per GPU."""
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
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911')
    p.add_argument('--arms', nargs='+', choices=['reference', 'balanced'], default=['reference', 'balanced'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--wandb-key-stdin', action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    cache = output / 'continuous-cache.pt'
    if not cache.is_file():
        raise FileNotFoundError(cache)
    for arm in args.arms:
        directory = output / arm
        if directory.exists() and not args.resume:
            raise FileExistsError(directory)
        if (directory / 'status.json').exists():
            status = json.loads((directory / 'status.json').read_text())
            cmdline = Path(f"/proc/{status['pid']}/cmdline")
            if cmdline.exists() and str(directory).encode() in cmdline.read_bytes():
                raise RuntimeError(f'{arm} is already running')
    environment = os.environ.copy()
    if args.wandb_key_stdin:
        environment['WANDB_API_KEY'] = getpass.getpass('W&B API key: ')
    if not environment.get('WANDB_API_KEY'):
        raise RuntimeError('Provide WANDB_API_KEY or --wandb-key-stdin')
    sources = [
        'scripts/train_church_ffhq_recipe.py', 'scripts/launch_church_ffhq_recipe.py',
        'scripts/build_church_ffhq_recipe_cache.py', 'src/church_ffhq_recipe.py',
        'src/coefficient_history_training.py', 'scripts/train_official_rqtransformer_laser_stage2.py',
        'src/models/rqtransformer/configs.py', 'src/models/rqtransformer/transformers.py',
        'src/models/rqtransformer/attentions.py', 'src/rqvae_metrics.py',
        'tests/test_church_ffhq_recipe.py',
    ]
    snapshot = output / 'source-snapshot'
    hashes = {}
    for name in sources:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and args.resume and sha256_file(destination) != sha256_file(ROOT / name):
            raise RuntimeError(f'Source changed since launch: {name}')
        shutil.copy2(ROOT / name, destination)
        hashes[name] = sha256_file(destination)
    (snapshot / 'sha256.json').write_text(json.dumps(hashes, indent=2))
    for arm in args.arms:
        gpu = 0 if arm == 'reference' else 1
        run_id = f'church-ffhq-v4-{arm}-earlydecay-20260911'
        command = [sys.executable, '-u', str(ROOT / 'scripts/train_church_ffhq_recipe.py'),
                   '--architecture', arm, '--output', str(output / arm), '--cache', str(cache),
                   '--epochs', '200', '--batch-size', '128', '--microbatch', '64',
                   '--wandb-id', run_id]
        if args.resume:
            command.append('--resume')
        env = {**environment, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '8',
               'TORCH_HOME': '/workspace/tmp/official-rqvae-eval-cache', 'PYTHONUNBUFFERED': '1'}
        logfile = output / f'{arm}.log'
        with logfile.open('ab') as handle:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        entry = {'architecture': arm, 'gpu': gpu, 'pid': child.pid, 'command': command,
                 'started_unix': time.time(), 'log': str(logfile),
                 'wandb_url': f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}'}
        (output / f'launch-{arm}.json').write_text(json.dumps(entry, indent=2))
        print(json.dumps(entry), flush=True)


if __name__ == '__main__':
    main()
