#!/usr/bin/env python3
"""Pause scratch pilots and launch verified continuations of the successful epoch-50 prior."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credential-pid',type=int,required=True)
    p.add_argument('--pause-pids',type=int,nargs='+',required=True)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/church-epoch50-loop-20260911')
    args=p.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    for variant in ['looped','control']:
        if (output/variant).exists(): raise FileExistsError(output/variant)
    sources={}
    for pid in set(args.pause_pids+[args.credential_pid]):
        process=Path(f'/proc/{pid}')
        command=[v.decode() for v in (process/'cmdline').read_bytes().split(b'\0') if v]
        assert str(ROOT/'scripts/train_church_looped_pair.py') in command
        assert (process/'cwd').resolve()==ROOT
        training=Path(command[command.index('--output')+1])
        sources[pid]={'pid':pid,'output':str(training),'status_before':json.loads((training/'status.json').read_text())}
    entries=Path(f'/proc/{args.credential_pid}/environ').read_bytes().split(b'\0')
    credential=next(v.split(b'=',1)[1].decode() for v in entries if v.startswith(b'WANDB_API_KEY='))
    for pid in args.pause_pids: os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+300
    while any(Path(f'/proc/{pid}/cmdline').exists() and Path(f'/proc/{pid}/cmdline').read_bytes() for pid in args.pause_pids):
        if time.monotonic()>deadline: raise RuntimeError('Graceful pause timed out')
        time.sleep(5)
    import torch
    for pid in args.pause_pids:
        row=sources[pid];training=Path(row['output'])
        status=json.loads((training/'status.json').read_text())
        saved=torch.load(training/'last.pt',map_location='cpu',weights_only=False)
        assert status['phase']=='paused' and status['optimizer_step']==saved['step']
        assert all(k in saved for k in ['state_dict','optimizer','stream'])
        row.update(status_after=status,saved_step=saved['step'],optimizer_and_stream_verified=True)
        del saved
    (output/'previous-scratch-pause.json').write_text(json.dumps(list(sources.values()),indent=2)+'\n')
    (output/'launcher-status.json').write_text(json.dumps({'pid':os.getpid(),'phase':'waiting_for_verification','previous_runs_paused':True})+'\n')
    print(json.dumps({'phase':'previous_runs_saved_and_paused','waiting_for':'verification.json'}),flush=True)
    # Keep the authorized credential only in process memory while model checks
    # run. No credential file is created. Only this fixed trainer may launch.
    deadline=time.monotonic()+7200
    while not (output/'verification.json').exists():
        if time.monotonic()>deadline: raise RuntimeError('Verification did not arrive')
        time.sleep(5)
    checked=json.loads((output/'verification.json').read_text())
    assert checked['production_ready']
    for name,value in checked['source_sha256'].items():
        assert digest(ROOT/name)==value,name
        target=output/'source-snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    launches=[]
    for gpu,variant in enumerate(['looped','control']):
        run_id=f'church-epoch50-{variant}-20260911'
        command=[sys.executable,'-u',str(ROOT/'scripts/train_church_epoch50_loop.py'),
                 '--output',str(output/variant),'--variant',variant,'--wandb-id',run_id]
        env={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'8',
             'OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache',
             'LASER_VGG16_WEIGHTS':'/workspace/tmp/laser-vgg/vgg16-397923af.pth'}
        with (output/f'{variant}.log').open('ab') as f:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        launches.append({'variant':variant,'pid':child.pid,'gpu':gpu,'command':command,
                         'wandb_url':f'https://wandb.ai/helloimlixin-rutgers/laser/runs/{run_id}'})
        result={'started_unix':time.time(),'runs':launches}
        (output/'launch.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'launcher-status.json').write_text(json.dumps({'phase':'launched',**result})+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
