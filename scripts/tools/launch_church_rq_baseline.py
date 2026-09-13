#!/usr/bin/env python3
"""Cancel the current Church run and launch a verified fresh RQ baseline."""
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
OLD=ROOT/'outputs/church-joint-geometry-20260912'
BASE=ROOT/'outputs/church-rq-baseline-scratch-20260912'


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    pid=74021;process=Path('/proc')/str(pid)
    command=[x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
    assert str(ROOT/'scripts/tools/train_church_joint_distributed.py') in command
    assert str(OLD/'joint') in command and (process/'cwd').resolve()==ROOT
    credential=next(x.split(b'=',1)[1].decode() for x in (process/'environ').read_bytes().split(b'\0') if x.startswith(b'WANDB_API_KEY='))
    before=json.loads((OLD/'joint/status.json').read_text())
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+300
    while (process/'cmdline').exists() and (process/'cmdline').read_bytes():
        if time.monotonic()>deadline:raise TimeoutError('Current run did not save and stop')
        time.sleep(2)
    import torch
    saved=torch.load(OLD/'joint/last.pt',map_location='cpu',weights_only=False,mmap=True)
    after=json.loads((OLD/'joint/status.json').read_text())
    assert after['phase']=='paused' and saved['step']==after['optimizer_step']
    assert len(saved['optimizer']['state'])>0 and 'stream' in saved
    step=saved['step'];del saved
    monitor_pid=74289;monitor=Path('/proc')/str(monitor_pid)
    if (monitor/'cmdline').exists():
        args=(monitor/'cmdline').read_bytes().split(b'\0')
        assert str(ROOT/'scripts/tools/monitor_church_training.py').encode() in args
        assert str(OLD/'joint').encode() in args
        os.kill(monitor_pid,signal.SIGTERM)
    receipt={'cancelled_by_user':True,'saved_step':step,'before':before,'after':after,
        'model_optimizer_stream_preserved':True,'best_checkpoint_preserved':True,'old_health_monitor_stopped':True}
    (BASE/'cancelled-previous-run.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (OLD/'joint/cancellation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    import wandb
    previous=wandb.Api(api_key=credential,timeout=30).run('helloimlixin-rutgers/laser/church-joint-geometry-20260912')
    previous.summary['training_status']='cancelled'
    previous.summary['cancelled_at_optimizer_step']=step
    previous.summary['cancellation_reason']='User requested a new from-scratch baseline RQTransformer run'
    previous.summary.update()
    print(json.dumps({'phase':'previous_run_saved_and_cancelled','step':step}),flush=True)
    (BASE/'launcher-status.json').write_text(json.dumps({'pid':os.getpid(),'phase':'waiting_for_baseline_verification'})+'\n')
    deadline=time.monotonic()+10800
    while not (BASE/'verification.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Baseline verification did not arrive')
        time.sleep(2)
    verified=json.loads((BASE/'verification.json').read_text());assert verified['production_ready']
    assert verified['random_initialization_verified'] and verified['empty_optimizer_verified']
    for name,digest in verified['source_hashes'].items():
        source=ROOT/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
        dest=BASE/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    spec=json.loads((BASE/'launch-spec.json').read_text())
    arguments=spec['arguments']
    assert '--resume' not in arguments and '--source-checkpoint' not in arguments
    assert arguments[arguments.index('--output')+1]==str(BASE/'baseline')
    assert not (BASE/'baseline').exists()
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(ROOT/'scripts/tools/train_church_rq_baseline.py'),*arguments]
    env={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':'0,1','OMP_NUM_THREADS':'8',
        'OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache',
        'LASER_VGG16_WEIGHTS':'/workspace/tmp/laser-vgg/vgg16-397923af.pth'}
    with (BASE/'production.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch={'torchrun_pid':child.pid,'command':command,'run_id':spec['run_id'],'started_unix':time.time()}
    (BASE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    print(json.dumps(launch),flush=True)


if __name__=='__main__':main()
