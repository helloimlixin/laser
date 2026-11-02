#!/usr/bin/env python3
"""Start the fresh integer-token pilot after a verified checkpoint handoff."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=Path('/tmp/laser-packed-pair512-proj128-imagenet-20261017'))
    parser.add_argument('--run',default='imagenet-laser-pairid512-proj128-scratch2-4h100-20261017')
    parser.add_argument('--wait-for-handoff',action='store_true')
    args=parser.parse_args()
    out=ROOT/'outputs'/args.run;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None;panel=None
    def stop(*_):
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
        if panel is not None and panel.poll() is None:panel.terminate()
        raise KeyboardInterrupt('Pilot controller interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def status(phase,**values):write(out/'launch-status.json',dict(phase=phase,pid=os.getpid(),
        run_id=args.run,time=time.time(),**values))
    if args.wait_for_handoff:
        status('waiting_for_verified_source_checkpoint_handoff')
        while True:
            p=out/'handoff-status.json'
            if p.exists() and json.loads(p.read_text())['phase']=='paused_with_verified_checkpoint':break
            time.sleep(2)
    plan=json.loads((out/'prepared.json').read_text())
    assert plan['prepared'] and not plan['training_launched'] and plan['fresh_stage2']
    preserved=json.loads((out/'preserved-source-checkpoint.json').read_text())
    assert preserved['passed'] and preserved['optimizer_updates_lost']==0
    status('waiting_for_gpu_release')
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(2)
    for name,expected in json.loads((out/'reproducibility/code-manifest.json').read_text()).items():
        with (args.base/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected,name
    env=dict(os.environ,**json.loads((args.base/'environment.json').read_text()))
    env['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    sizes=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).splitlines()]
    assert len(sizes)==4 and min(sizes)>75000
    distributed=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4']
    def launch(command,phase):
        nonlocal child
        with (out/f'{phase}.log').open('a') as log:
            child=subprocess.Popen(command,cwd=args.base,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status(phase,child_pid=child.pid)
            code=child.wait()
        if code:
            status(phase+'_failed',returncode=code)
            raise RuntimeError(f'{phase} failed with exit code {code}')
    launch([*distributed,str(args.base/'train.py'),'--microbatch','16','--smoke-updates','2'],'gpu_preflight')
    launch([sys.executable,str(args.base/'verify.py')],'checkpoint_verification')
    assert json.loads((out/'smoke-reload-verification.json').read_text())['passed']
    plan.update(training_launched=True,preflight_passed=True,production_starts_from_zero=True,
        world_size=4,microbatch=16,accumulation_steps=32,launched_at=time.time())
    write(out/'prepared.json',plan)
    with (out/'sample-workspace.log').open('a') as log:
        panel=subprocess.Popen([sys.executable,str(args.base/'configure_samples.py'),'--run',args.run,'--watch'],
            cwd=args.base,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch([*distributed,str(args.base/'train.py'),'--microbatch','16','--epochs','2'],'training')
    status('completed',epochs=2,updates=1252)


if __name__=='__main__':main()
