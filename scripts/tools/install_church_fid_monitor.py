#!/usr/bin/env python3
"""Save the live run, await checked monitor code, and resume the same W&B run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/church-joint-geometry-20260912'


def main():
    folder=BASE/'fid-monitor';output=BASE/'joint'
    pid=71810;process=Path('/proc')/str(pid)
    old_command=[x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
    assert str(ROOT/'scripts/tools/train_church_joint_distributed.py') in old_command
    assert str(output) in old_command and (process/'cwd').resolve()==ROOT
    credential=next(x.split(b'=',1)[1].decode() for x in (process/'environ').read_bytes().split(b'\0') if x.startswith(b'WANDB_API_KEY='))
    before=json.loads((output/'status.json').read_text())
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+300
    while (process/'cmdline').exists() and (process/'cmdline').read_bytes():
        if time.monotonic()>deadline:raise TimeoutError('Trainer did not save and pause')
        time.sleep(2)
    import torch
    saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    after=json.loads((output/'status.json').read_text())
    assert after['phase']=='paused' and after['optimizer_step']==saved['step']
    assert {'state_dict','optimizer','stream'}<=saved.keys() and len(saved['optimizer']['state'])>0
    preserved_step=saved['step'];del saved
    for name in ('last.pt','best-screen.pt'):
        target=folder/('pre-monitor-'+name)
        if not target.exists():os.link(output/name,target)
    receipt={'before':before,'after':after,'preserved_step':preserved_step,'optimizer_and_stream_preserved':True}
    (folder/'pause.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'phase':'saved_and_paused','step':preserved_step}),flush=True)
    deadline=time.monotonic()+1200
    while not (folder/'verification.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Monitor verification not supplied')
        time.sleep(2)
    check=json.loads((folder/'verification.json').read_text());assert check['production_ready']
    for name,digest in check['source_hashes'].items():
        source=ROOT/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
        dest=folder/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    env={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'8',
        'OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache',
        'LASER_VGG16_WEIGHTS':'/workspace/tmp/laser-vgg/vgg16-397923af.pth'}
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(ROOT/'scripts/tools/train_church_joint_distributed.py'),'--output',str(output),
        '--microbatch','32','--generation-batch','512','--fid-lr-policy',str(folder/'policy.json'),
        '--wandb-id','church-joint-geometry-20260912']
    with (folder/'production.log').open('ab') as handle:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    launch={'torchrun_pid':child.pid,'command':command,'resumed_step':preserved_step}
    (folder/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    deadline=time.monotonic()+300
    while True:
        if child.poll() is not None:raise RuntimeError('Trainer exited after installing monitor')
        current=json.loads((output/'status.json').read_text())
        if current['phase']=='train' and current['optimizer_step']>=preserved_step+4:break
        if time.monotonic()>deadline:raise TimeoutError('No training progress after resume')
        time.sleep(2)
    with (folder/'health-monitor.log').open('ab') as handle:
        monitor=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/tools/monitor_church_training.py'),
            '--output',str(output),'--reports',str(folder)],cwd=ROOT,stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={**launch,'monitor_pid':monitor.pid,'live_status':current,'installation_complete':True}
    (folder/'installation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
