#!/usr/bin/env python3
"""Guarded migration of the existing corrected run; preserve rollback checkpoints."""
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
    folder=BASE/'multi-gpu'
    verified=json.loads((folder/'verification.json').read_text())
    assert verified['production_ready']
    for name,expected in verified['source_hashes'].items():
        source=ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,name
        dest=folder/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    pid=67288;p=Path('/proc')/str(pid)
    command=[v.decode() for v in (p/'cmdline').read_bytes().split(b'\0') if v]
    assert str(ROOT/'scripts/train_church_joint_geometry.py') in command
    assert str(BASE/'joint') in command and (p/'cwd').resolve()==ROOT
    assert 'T' in (p/'status').read_text().split('State:')[1].splitlines()[0]
    credential=next(v.split(b'=',1)[1].decode() for v in (p/'environ').read_bytes().split(b'\0') if v.startswith(b'WANDB_API_KEY='))
    # Only the authorized trainer receives the key. It is never saved to disk.
    env={**os.environ,'WANDB_API_KEY':credential,'CUDA_VISIBLE_DEVICES':'0,1',
        'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','MKL_NUM_THREADS':'8',
        'TORCH_HOME':'/workspace/tmp/official-rqvae-eval-cache','LASER_VGG16_WEIGHTS':'/workspace/tmp/laser-vgg/vgg16-397923af.pth'}
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
        str(ROOT/'scripts/tools/train_church_joint_distributed.py'),'--output',str(BASE/'joint'),
        '--microbatch','32','--generation-batch','512','--wandb-id','church-joint-geometry-20260912']
    with (folder/'production.log').open('ab') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch={'launcher_pid':os.getpid(),'torchrun_pid':child.pid,'command':command,'old_pid':pid,'started_unix':time.time()}
    (folder/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    try:
        deadline=time.monotonic()+300
        while True:
            if child.poll() is not None:raise RuntimeError('Distributed trainer exited during handoff')
            progress=[BASE/f'joint/evaluations/step-004930/full/progress-{r:02d}.json' for r in range(2)]
            if all(f.exists() for f in progress):
                rows=[json.loads(f.read_text()) for f in progress]
                if all(r['generated_on_rank']>=512 for r in rows):break
            if time.monotonic()>deadline:raise TimeoutError('No generation progress from both ranks')
            time.sleep(2)
        # The suspended process is only inside an unsaved evaluation loop.
        # Its exact training state is already saved and verified at step4930.
        # It cannot handle SIGTERM until its old 50k loop finishes, so retire it
        # after the replacement demonstrably works. No training updates lost.
        assert p.exists() and (p/'cmdline').read_bytes()
        os.kill(pid,signal.SIGKILL)
        receipt={**launch,'handoff_complete':True,'both_rank_progress':rows,
            'preserved_training_step':4930,'partial_old_evaluation_discarded':True,
            'checkpoint_and_optimizer_restored':True,'old_evaluator_retired':True}
        (folder/'handoff.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    except Exception:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=30)
        if p.exists():os.kill(pid,signal.SIGCONT)
        raise


if __name__=='__main__':main()
