#!/usr/bin/env python3
"""Preserve the larger prior and launch the verified compact prior from scratch."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from relaunch_original_church_rq import current_conversation_credential

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/church-compact-rq-stage2-20260913'
OLD=ROOT/'outputs/church-scaled-atom-stage2-20260913'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    proof=json.loads((BASE/'verification.json').read_text())
    assert proof['preflight_passed'] and proof['strict_checkpoint_reload']
    assert proof['released_sampler_decode_smoke_passed'] and proof['tokenizer_unchanged']
    assert proof['microbatch_per_gpu']==128 and proof['peak_gpu_allocated_gib']<120
    assert proof['parameters']==386882561 and proof['vocab_size']==32769
    assert not (BASE/'train').exists() and not (BASE/'launch.json').exists()
    credential=current_conversation_credential()
    hashes=dict(proof['source_hashes'])
    previous=json.loads((OLD/'source-manifest.json').read_text())
    for name,expected in previous.items():
        assert digest(ROOT/name)==expected,name
        hashes[name]=expected
    for path in (Path(__file__),ROOT/'scripts/tools/relaunch_original_church_rq.py'):
        hashes[str(path.relative_to(ROOT))]=digest(path)
    for name,expected in hashes.items():
        source=ROOT/name
        assert digest(source)==expected,name
        dest=BASE/'source-snapshot'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    write_json(BASE/'source-manifest.json',hashes)
    old_status=json.loads((OLD/'train/status.json').read_text())
    old_pid=old_status['pid']
    proc=Path(f'/proc/{old_pid}')
    cmdline=proc.joinpath('cmdline').read_bytes().split(b'\0')
    assert str(ROOT/'scripts/tools/train_scaled_atom_stage2.py').encode() in cmdline
    assert str(OLD/'train').encode() in cmdline
    assert proc.joinpath('cwd').resolve()==ROOT
    assert old_status['phase']=='training'
    requested=time.time()
    os.kill(old_pid,signal.SIGTERM)
    print(json.dumps(dict(phase='requested_graceful_pause',old_rank0_pid=old_pid,
        previous_optimizer_step=old_status['optimizer_step'])),flush=True)
    deadline=time.monotonic()+180
    while True:
        saved=json.loads((OLD/'train/status.json').read_text())
        if saved['phase']=='paused' and not proc.exists():
            break
        if time.monotonic()>deadline:
            raise TimeoutError('Larger run has not confirmed a checkpointed pause; compact run not launched')
        time.sleep(2)
    checkpoint=OLD/'train/last.pt'
    assert checkpoint.stat().st_mtime>=requested and checkpoint.stat().st_size>5_000_000_000
    pause=dict(old_run='church-scaled-rq8-scratch-20260913',rank0_pid=old_pid,
        checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        checkpoint_bytes=checkpoint.stat().st_size,status=saved,
        reason='Both GPUs reassigned to the user-approved compact vocabulary experiment',
        requested_unix=requested,completed_unix=time.time())
    write_json(BASE/'larger-run-paused.json',pause)
    print(json.dumps(dict(phase='larger_run_checkpointed',optimizer_step=saved['optimizer_step'])),flush=True)
    run_id='church-compact-rq32k-scratch-20260913'
    command=['/tmp/laser-sign-venv/bin/python','-m','torch.distributed.run','--standalone',
        '--nproc_per_node=2',str(ROOT/'scripts/tools/train_compact_rq_stage2.py'),
        '--cache',str(BASE/'cache'),'--calibration',str(BASE/'temperature-calibration.json'),
        '--output',str(BASE/'train'),'--run-id',run_id,'--batch-size','128']
    env={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':'0,1',
        'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
        'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache'}
    for name in ('WANDB_SERVICE','_WANDB_SERVICE'):
        env.pop(name,None)
    with (BASE/'production.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(torchrun_pid=child.pid,command=command,run_id=run_id,started_unix=time.time(),
        expected_initial_weights_sha256=proof['model_initialization_sha256'],
        source_files_verified=len(hashes),production_from_scratch=True,
        preflight_checkpoint_used_for_training=False,larger_run_preserved=str(BASE/'larger-run-paused.json'))
    write_json(BASE/'launch.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
