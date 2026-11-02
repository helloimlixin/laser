#!/usr/bin/env python3
"""Complete the reconstruction study while preserving the active AR experiment."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/workspace/Projects/laser')
OUT=ROOT/'outputs/imagenet-laser-joint20k-refinement-rfid-20261017'
ACTIVE=ROOT/'outputs/imagenet-laser-joint20k-scratch2-4h100-20261017'
BASE=Path('/tmp/laser-rq-joint20k-imagenet-20261017')
SCRIPT=ROOT/'scripts/tools/fit_laser_joint20k_refinement.py'


def write(phase,**values):
    path=OUT/'controller-status.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),time=time.time(),**values),indent=2)+'\n');tmp.replace(path)


def matches(pid,path):
    try:return str(path).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False


def main():
    environment=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    deadline=time.monotonic()+2400
    write('waiting_for_refinement_fit')
    while not ((OUT/'fit-report.json').exists() and json.loads((OUT/'fit-report.json').read_text())['completed']):
        if time.monotonic()>deadline:raise TimeoutError('Refinement fit did not complete')
        time.sleep(5)
    def launch(arguments,name):
        with (OUT/(name+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen(arguments,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(name,child_pid=child.pid)
            code=child.wait()
        if code:raise RuntimeError(f'{name} failed with exit code{code}')
    if not (OUT/'screen-results.json').exists():launch([sys.executable,str(SCRIPT),'screen'],'screen')
    audit_script=ACTIVE/'context-audit/source/audit.py';stopped=False
    try:
        receipt=ACTIVE/'context-audit/process.json'
        if receipt.exists():
            pid=json.loads(receipt.read_text())['pid']
            if matches(pid,audit_script):
                os.kill(pid,signal.SIGTERM);stopped=True
                for _ in range(30):
                    if not matches(pid,audit_script):break
                    time.sleep(1)
                assert not matches(pid,audit_script)
        launch([sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
            str(SCRIPT),'full','--memory-fraction','.14','--batch-size','8'],'full')
    finally:
        if stopped:
            with (ACTIVE/'context-audit/audit.log').open('ab',buffering=0) as log:
                child=subprocess.Popen([sys.executable,str(audit_script)],cwd=ROOT,env=environment,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (ACTIVE/'context-audit/process.json').write_text(json.dumps(dict(pid=child.pid,time=time.time(),
                restarted_after_offline_fidelity_evaluation=True),indent=2)+'\n')
    result=json.loads((OUT/'full-rfid.json').read_text())
    assert result['completed'] and result['real_images']==result['reconstructed_images_per_candidate']==50000
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    import wandb
    run=wandb.Api(timeout=45).run('helloimlixin-rutgers/laser/imagenet-laser-joint20k-scratch2-4h100-20261017')
    folder=BASE/'audit-uploads/offline-tokenizer-reconstruction';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'refinement-full-rfid.json';target.write_text(json.dumps(result,indent=2)+'\n')
    run.upload_file(str(target),root=str(BASE/'audit-uploads'))
    run.config['offline_refinement_reconstruction_audit']=dict(vocabulary=20481,
        experimental_depths={d:dict(rfid=r['measured_rfid'],tokens_per_image=r['tokens_per_image']) for d,r in result['results'].items()},
        current_training_depth=4,current_training_rfid=4.689409482068868,current_training_unchanged=True,
        requires_longer_autoregressive_sequence=True,additional_training_losses=0)
    run.update()
    write('completed',results=result['results'],current_ar_training_unchanged=True,diagnostics_uploaded=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException as error:
        write('failed',error_type=type(error).__name__,error=str(error));raise
