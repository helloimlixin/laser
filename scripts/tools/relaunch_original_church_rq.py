#!/usr/bin/env python3
"""Recover the reviewed new run after a pre-training W&B service startup failure."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/church-rq-baseline-scratch-20260912'


def current_conversation_credential():
    """Read only the user's explicitly supplied key in this exact conversation."""
    thread_id = os.environ['CODEX_THREAD_ID']
    matches = list(Path('/root/.codex/sessions').rglob(f'*{thread_id}.jsonl'))
    assert len(matches) == 1
    keys = set()
    with matches[0].open() as stream:
        first = json.loads(next(stream))
        assert first['payload']['id'] == thread_id
        for line in stream:
            row = json.loads(line)
            payload = row.get('payload', {})
            texts = []
            if row.get('type') == 'event_msg' and payload.get('type') == 'user_message':
                texts.append(payload.get('message', ''))
            if row.get('type') == 'response_item' and payload.get('role') == 'user':
                texts.extend(block.get('text', '') for block in payload.get('content', [])
                             if isinstance(block, dict))
            for text in texts:
                keys.update(re.findall(r'my wandb api key:\s*(wandb_v1_[A-Za-z0-9_-]+)', text))
    assert len(keys) == 1, 'Expected the single user-provided credential in this conversation'
    return keys.pop()


def main():
    credential = current_conversation_credential()
    old = BASE / 'baseline'
    assert old.exists() and not (old / 'initialization.json').exists()
    assert not list(old.glob('*.pt'))
    old.rename(BASE / 'baseline-startup-failed-183405')
    proof = json.loads((BASE / 'verification.json').read_text())
    assert proof['production_ready'] and proof['random_initialization_verified']
    for name, digest in proof['source_hashes'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        dest = BASE / 'source-snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    spec = json.loads((BASE / 'launch-spec.json').read_text())
    command = ['/tmp/laser-sign-venv/bin/python', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=2', str(ROOT / 'scripts/tools/train_church_rq_baseline.py'),
               *spec['arguments']]
    env = {**os.environ, 'WANDB_API_KEY': credential, 'CUDA_VISIBLE_DEVICES': '0,1',
           'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8',
           'TORCH_HOME': '/workspace/tmp/official-rqvae-eval-cache'}
    for name in ('WANDB_SERVICE', '_WANDB_SERVICE'):
        env.pop(name, None)
    with (BASE / 'production.log').open('ab') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'torchrun_pid': child.pid, 'command': command, 'run_id': spec['run_id'],
               'started_unix': time.time(), 'startup_recovery': 'clear inherited W&B service socket'}
    (BASE / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    deadline = time.monotonic() + 180
    while not (old / 'wandb.json').exists():
        if child.poll() is not None:
            raise RuntimeError('Reviewed new launch failed; inspect production.log')
        if time.monotonic() > deadline:
            raise TimeoutError('New W&B service did not initialize')
        time.sleep(2)
    print('new W&B run initialized successfully', flush=True)


if __name__ == '__main__':
    main()
