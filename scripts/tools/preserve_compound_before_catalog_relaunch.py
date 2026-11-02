#!/usr/bin/env python3
"""Preserve the latest committed source checkpoint, then release GPUs for relaunch."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path('/workspace/Projects/laser')
OLD_RUN = 'imagenet-laser-compound512-origlr-scratch100-4h100-20261016-r3'
NEW_RUN = 'imagenet-laser-pairid256k-grid512-scratch100-4h100-20261017'
OLD = ROOT/'outputs'/OLD_RUN
OUT = ROOT/'outputs'/NEW_RUN
BASE = Path('/tmp/laser-compound512-origlr-imagenet-20261016-r3')


def matches(pid, token):
    try:
        return token.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def write(name, **values):
    value = dict(values, time=time.time())
    tmp = OUT/(name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(OUT/name)
    print(json.dumps(value), flush=True)


def main():
    os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')
    sys.path[:0] = [str(ROOT), str(BASE/'upstream')]
    from src import rq_upstream
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    receipts = [json.loads(p.read_text()) for p in (OLD/'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]
    receipt = max(receipts, key=lambda r:r['global_step'])
    write('source-handoff-status.json', phase='verifying_latest_committed_checkpoint', receipt=receipt)
    for key in ['local', 'persistent']:
        path = Path(receipt[key])
        assert path.stat().st_size == receipt['bytes']
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256'], key
        write('source-content-'+key+'-verification.json', passed=True, path=str(path),
            bytes=receipt['bytes'], sha256=receipt['sha256'])
    saved = torch.load(receipt['local'], map_location='cpu', mmap=True, weights_only=False)
    step = receipt['global_step']
    assert saved['run_id'] == OLD_RUN and saved['global_step'] == step
    assert saved['next_epoch'] == step//626 and saved['next_microbatch'] == 0
    assert len(saved['rng_state_by_rank']) == 4
    assert all(int(state['step']) == step for state in saved['optimizer']['state'].values())
    del saved
    current = [json.loads((OLD/'train'/f'progress-rank{r}.json').read_text()) for r in range(4)]
    pids = [r['pid'] for r in current]
    token = str(BASE/'train.py')
    assert all(matches(pid, token) for pid in pids)
    assert matches(167733, str(BASE))
    assert not list((BASE/'checkpoint-staging').glob('*.staging')), 'Do not interrupt a checkpoint serialization'
    observed = max(int(r['train/global_step']) for r in current)
    write('preserved-source-checkpoint.json', passed=True, source_run=OLD_RUN, receipt=receipt,
        local_sha256_verified=True, persistent_sha256_verified=True, all_adam_steps=step,
        all_rank_rng_preserved=True, latest_observed_source_step=observed,
        unfinished_epoch_updates_not_in_checkpoint=observed-step,
        restart_from_scratch_requested=True, old_committed_checkpoint_unchanged=True)
    write('source-handoff-status.json', phase='stopping_previous_experiment', source_pids=pids,
        preserved_step=step, latest_observed_step=observed)
    os.kill(167733, signal.SIGTERM)
    for second in range(40):
        live = [pid for pid in pids if matches(pid, token)]
        if not live:
            break
        if second in [5, 20, 35]:
            for pid in live:
                os.kill(pid, signal.SIGINT if second == 5 else signal.SIGTERM if second == 20 else signal.SIGKILL)
        time.sleep(1)
    assert not any(matches(pid, token) for pid in pids)
    write('source-handoff-status.json', phase='source_stopped_with_verified_checkpoint',
        preserved_step=step, latest_observed_step=observed, source_run=OLD_RUN)


if __name__ == '__main__':
    main()
