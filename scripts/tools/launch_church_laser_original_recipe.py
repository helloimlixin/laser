#!/usr/bin/env python3
"""Launch the verified fresh LASER Church recipe after a checkpointed pause."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from relaunch_original_church_rq import current_conversation_credential

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/church-laser-original-recipe-20260913'
OLD=ROOT/'outputs/church-compact-rq-stage2-20260913'
DRIVER=ROOT/'scripts/tools/train_church_laser_original_recipe.py'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['pause-old','confirm-pause','preflight-512','launch'])
    args=p.parse_args()
    proof=json.loads((BASE/'verification-256.json').read_text())
    assert proof['passed'] and proof['strict_checkpoint_reload']
    assert digest(DRIVER)==proof['driver_sha256']
    if args.phase=='pause-old':
        assert not (BASE/'pause-request.json').exists()
        status=json.loads((OLD/'train/status.json').read_text())
        pid=100611
        proc=Path(f'/proc/{pid}')
        command=proc.joinpath('cmdline').read_bytes().split(b'\0')
        assert str(ROOT/'scripts/tools/train_compact_rq_stage2.py').encode() in command
        assert str(OLD/'train').encode() in command
        assert proc.joinpath('cwd').resolve()==ROOT
        receipt=dict(rank0_pid=pid,requested_unix=time.time(),previous_status=status,
            reason='Replace the overfitted batch256 LASER run with a fresh paper-batch2048 comparison')
        write(BASE/'pause-request.json',receipt)
        os.kill(pid,signal.SIGTERM)
        print(json.dumps(dict(phase='graceful_pause_requested',rank0_pid=pid)),flush=True)
        return
    if args.phase=='confirm-pause':
        request=json.loads((BASE/'pause-request.json').read_text())
        status=json.loads((OLD/'train/status.json').read_text())
        assert status['phase']=='paused',status
        assert not Path(f"/proc/{request['rank0_pid']}").exists()
        checkpoint=OLD/'train/last.pt'
        assert checkpoint.stat().st_mtime>=request['requested_unix']
        assert checkpoint.stat().st_size>4_000_000_000
        sys.path[:0]=[str(OLD/'source-snapshot/outputs/church-rq-baseline-scratch-20260912/upstream-source'),str(ROOT)]
        import torch
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert state['step']==status['optimizer_step']
        assert state['optimizer']['state'] and state['scheduler'] and state['scaler']
        assert len(state['rng_states'])==2
        receipt=dict(checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
            status=status,full_training_state_verified=True,confirmed_unix=time.time())
        write(BASE/'previous-run-preserved.json',receipt)
        print(json.dumps(receipt),flush=True)
        return
    preserved=json.loads((BASE/'previous-run-preserved.json').read_text())
    assert preserved['full_training_state_verified']
    assert not Path('/proc/100611').exists() and not Path('/proc/100612').exists()
    preflight=args.phase=='preflight-512'
    if not preflight:
        proof=json.loads((BASE/'verification-512.json').read_text())
        assert proof['passed'] and proof['strict_checkpoint_reload']
        assert proof['peak_gpu_allocated_gib']<125
        assert digest(DRIVER)==proof['driver_sha256']
    output=BASE/('preflight-512' if preflight else 'train')
    assert not output.exists()
    receipt_path=BASE/('preflight-512-launch.json' if preflight else 'launch.json')
    assert not receipt_path.exists()
    run_id='church-laser-rq32k-original-recipe-scratch-20260913'
    command=['/tmp/laser-sign-venv/bin/python','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=2',str(DRIVER),'--cache',str(OLD/'cache'),
        '--calibration',str(OLD/'temperature-calibration.json'),'--output',str(output),
        '--run-id',run_id,'--batch-size','512']
    if preflight: command+=['--max-updates','2','--offline']
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'8',
        'OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
        'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache'}
    if not preflight: env['WANDB_API_KEY']=current_conversation_credential()
    for name in ('WANDB_SERVICE','_WANDB_SERVICE'): env.pop(name,None)
    with (BASE/('preflight-512.log' if preflight else 'production.log')).open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(torchrun_pid=child.pid,command=command,run_id=run_id,
        started_unix=time.time(),from_scratch=True,preflight_weights_loaded=False,
        driver_sha256=digest(DRIVER),expected_initial_weights_sha256=proof['initial_weights_sha256'])
    write(receipt_path,receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__': main()
