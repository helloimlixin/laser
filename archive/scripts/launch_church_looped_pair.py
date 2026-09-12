#!/usr/bin/env python3
"""Launch the verified compound RQ loop comparison using an inherited credential."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'outputs/church-looped-pair-20260911'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    verification = json.loads((OUTPUT / 'verification.json').read_text())
    assert verification['production_ready']
    assert os.environ.get('WANDB_API_KEY'), 'Expected inherited W&B authentication'
    for variant in ('looped', 'unrolled'):
        if (OUTPUT / variant).exists():
            raise FileExistsError(OUTPUT / variant)
    for name, expected in verification['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
        target = OUTPUT / 'source-snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    children = []
    for gpu, variant in enumerate(('looped', 'unrolled')):
        run_id = f'church-compound-rq-{variant}-20260911'
        command = [sys.executable, '-u', str(ROOT / 'scripts/train_church_looped_pair.py'),
                   '--output', str(OUTPUT / variant), '--variant', variant, '--wandb-id', run_id]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '8',
               'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
               'TORCH_HOME': '/workspace/tmp/official-rqvae-eval-cache',
               'LASER_VGG16_WEIGHTS': '/workspace/tmp/laser-vgg/vgg16-397923af.pth'}
        with (OUTPUT / f'{variant}.log').open('ab') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append({'variant': variant, 'gpu': gpu, 'pid': child.pid, 'command': command,
                         'wandb_url': f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}'})
        (OUTPUT / 'launch.json').write_text(json.dumps(
            {'started_unix': time.time(), 'runs': children}, indent=2) + '\n')
    print(json.dumps({'phase': 'looped_pair_comparison_launched', 'runs': children}), flush=True)


if __name__ == '__main__':
    main()
