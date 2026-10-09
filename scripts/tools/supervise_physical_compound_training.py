"""Checkpoint the prior run, validate a bounded migration, then continue to100."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def record(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    for name in ('base','output','parent','parent-output','key-file'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-supervisor',type=int,required=True)
    p.add_argument('--attach-pilot',type=int)
    args=p.parse_args()
    sys.path[:0]=[str(args.base/'source/runtime'),str(args.base/'support')]
    import fcntl
    lock=(args.base/'training-supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run_id=json.loads((args.output/'preparation-complete.json').read_text())['run_id']
    status=args.output/'training-status.json'
    os.environ['WANDB_API_KEY']=args.key_file.read_text().strip()
    os.environ['WANDB_DIR']=str(args.base/'wandb')
    import wandb
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=run_id,resume='allow',
        mode='online',dir=str(args.base),config=json.loads((args.output/'preparation-complete.json').read_text()))
    run.summary['execution/state']='prepared_full_history_pilot'
    run.summary['evaluation/backend']='official RQ-Transformer only'
    for path in (args.base/'entry.py',args.base/'recipe.yaml',args.base/'compound-transfer.json',
                 args.base/'source/runtime/src/models/physical_compound_prior.py',
                 args.base/'source/runtime/src/training/physical_compound_resume.py'):
        run.save(str(path),base_path=str(args.base),policy='now')
    run.finish()
    record(status,dict(state='checkpointing_previous_training',run_id=run_id,time=time.time()))
    proc=Path(f'/proc/{args.parent_supervisor}/cmdline')
    if proc.exists():
        command=proc.read_bytes().replace(b'\0',b' ').decode()
        if command and (str(args.parent) not in command or 'supervise' not in command):
            raise RuntimeError('Previous supervisor identity changed')
        if command:
            os.kill(args.parent_supervisor,signal.SIGTERM)
    deadline=time.monotonic()+900
    parent_pids=[]
    parent_verify=args.parent_output/'verification/rqcos0_20261006_epoch77_0'
    for path in parent_verify.glob('process-rank*.json'):
        parent_pids.append(json.loads(path.read_text())['pid'])
    def parent_rank_alive(pid):
        path=Path(f'/proc/{pid}/cmdline')
        try:
            return str(args.parent).encode() in path.read_bytes()
        except FileNotFoundError:
            return False
    while any(parent_rank_alive(pid) for pid in parent_pids):
        if time.monotonic()>deadline:
            raise RuntimeError('Previous trainer has not released GPUs after its graceful checkpoint')
        record(status,dict(state='waiting_previous_checkpoint_and_gpu_release',run_id=run_id,time=time.time()))
        time.sleep(5)
    saved=json.loads((args.parent_output/'last-local-save.json').read_text())
    ready=json.loads((args.parent_output/'trial-checkpoint-ready.json').read_text())
    if saved['step']!=ready['global_step'] or not Path(saved['target']).is_file():
        raise RuntimeError('Previous training did not commit its final recovery state')
    record(args.output/'previous-training-preserved.json',dict(**saved,ready=ready,pids_released=parent_pids))
    env=dict(os.environ,LASER_RUN_BASE=str(args.base),LASER_PERSISTENT_BASE=str(args.output),
        LASER_ACCUMULATION='4',LASER_CHECKPOINT_STAGING_DIR=str(args.base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base/'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',LASER_COMPILE_BLOCKS='1',LASER_PREVIEW_KEEP_OPTIMIZER='1',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',WANDB_DIR=str(args.base/'wandb'),
        WANDB_CACHE_DIR=str(args.base/'wandb-cache'),WANDB_DATA_DIR=str(args.base/'wandb-data'),
        WANDB_CONFIG_DIR=str(args.base/'wandb-config'),TORCHINDUCTOR_CACHE_DIR=str(args.base/'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='2',TORCH_HOME=str(args.base/'torch-cache'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',NCCL_NVLS_ENABLE='0',
        TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    (args.output/'continuation-20261005').mkdir(exist_ok=True)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
             str(args.base/'entry.py'),'--config',str(args.base/'recipe.yaml')]
    child=None
    def stop(_sig,_frame):
        phase=current_phase
        for path in (args.output/'verification'/phase).glob('process-rank*.json'):
            pid=json.loads(path.read_text())['pid']
            cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and str(args.base).encode() in cmd.read_bytes():
                os.kill(pid,signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(phase):
        nonlocal child,current_phase
        current_phase=phase
        if phase=='compound_pilot' and args.attach_pilot:
            # The native trainer freezes a committed recovery state before
            # draining uploads. Release its GPUs after authenticating that
            # state; the continuing trainer retries the latest upload in its
            # background worker while making new training progress.
            ready=json.loads((args.output/'trial-checkpoint-ready.json').read_text())
            saved=json.loads((args.output/'last-local-save.json').read_text())
            if not ready['no_further_updates'] or ready['global_step']!=48222 or saved['step']!=48222:
                raise RuntimeError('Pilot is not frozen at its verified endpoint')
            verify=args.output/'verification/compound_pilot'
            for rank in range(8):
                if not json.loads((verify/f'step20-rank{rank}.json').read_text())['finite']:
                    raise RuntimeError('Pilot rank failed its finite-state check')
            probe=json.loads((verify/'compound-probe-step20.json').read_text())
            if not probe['finite'] or probe['mean_cross_entropy']>=6.81:
                raise RuntimeError('Pilot likelihood check failed')
            import torch
            from src.training.full_resume_upload import recovery_metadata
            from src.training.k4_checkpoint_io import _checkpoint_upload_source
            os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR']=str(args.base/'checkpoint-upload-cache')
            raw=torch.load(_checkpoint_upload_source(Path(saved['target'])),map_location='cpu',mmap=True,weights_only=False)
            meta=recovery_metadata(raw)
            if meta['global_step']!=48222 or meta['rng_ranks']!=8 or meta['compound_optimizer_ages']['new_adam_step']!=20:
                raise RuntimeError('Pilot recovery state is incomplete')
            record(args.output/'pilot-release-verification.json',dict(
                global_step=meta['global_step'],optimizer_ages=meta['compound_optimizer_ages'],
                rng_ranks=meta['rng_ranks'],scheduler_step=raw['scheduler']['last_epoch'],
                full_checkpoint_committed=True,no_further_updates=True,time=time.time()))
            del raw
            pids=[json.loads(path.read_text())['pid'] for path in verify.glob('process-rank*.json')]
            pids.append(args.attach_pilot)
            for pid in pids:
                command_path=Path(f'/proc/{pid}/cmdline')
                if command_path.exists():
                    command_bytes=command_path.read_bytes()
                    if command_bytes and str(args.base).encode() not in command_bytes:
                        raise RuntimeError('Pilot process ownership changed')
                    if command_bytes:os.kill(pid,signal.SIGKILL)
            time.sleep(3)
            return 0
        stream=(args.output/(phase+'.log')).open('a')
        child=subprocess.Popen(command,cwd=args.base,env=dict(env,LASER_PHASE=phase),
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            record(status,dict(state='pilot_running' if phase=='compound_pilot' else 'training',
                supervisor_pid=os.getpid(),torchrun_pid=child.pid,phase=phase,run_id=run_id,
                target_epoch=100,official_evaluations_only=True,time=time.time()))
            time.sleep(5)
        return child.returncode
    current_phase='compound_pilot'
    code=launch(current_phase)
    checks=[]
    verify=args.output/'verification/compound_pilot'
    for rank in range(8):
        path=verify/f'step20-rank{rank}.json'
        gradient=verify/f'history-gradient-rank{rank}.json'
        checks.append(path.exists() and json.loads(path.read_text()).get('finite') is True)
        checks.append(gradient.exists() and json.loads(gradient.read_text()).get('nonzero') is True)
    probe_path=verify/'compound-probe-step20.json'
    probe=json.loads(probe_path.read_text()) if probe_path.exists() else {}
    passed=code==0 and all(checks) and probe.get('finite') is True and probe.get('mean_cross_entropy',999)<6.81
    report=dict(passed=passed,exit_code=code,rank_checks=checks,probe=probe,time=time.time())
    record(args.output/'pilot-validation.json',report)
    if not passed:
        record(status,dict(state='pilot_failed_restoring_previous_training',run_id=run_id,report=report))
        # Restore the preserved original continuation automatically.
        fallback=[sys.executable,str(args.parent/'support/continue_imagenet_global_cosine.py'),'train-once',
            '--base',str(args.parent),'--evidence',str(args.parent_output),'--key-file',str(args.key_file)]
        with (args.output/'fallback-training.log').open('a') as stream:
            fallback_child=subprocess.Popen(fallback,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        record(status,dict(state='pilot_failed_previous_training_resumed',pid=fallback_child.pid,
            run_id=run_id,report=report,time=time.time()))
        return
    import yaml
    recipe=yaml.safe_load((args.base/'recipe.yaml').read_text())
    recipe['options']['max_optimizer_steps']=0
    (args.base/'recipe.yaml').write_text(yaml.safe_dump(recipe,sort_keys=False))
    (args.output/'trial-checkpoint-ready.json').rename(args.output/'pilot-checkpoint-ready.json')
    run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=run_id,resume='must',
        mode='online',dir=str(args.base))
    run.summary['verification/compound_pilot_passed']=True
    run.summary['execution/state']='continuing_to_epoch100'
    run.finish()
    code=launch('compound_production')
    saved=json.loads((args.output/'last-local-save.json').read_text())
    record(status,dict(state='completed' if code==0 and saved['step']==62600 else 'stopped_resumable' if code==0 else 'failed',
        run_id=run_id,exit_code=code,final_step=saved['step'],target_epoch=100,time=time.time()))


if __name__=='__main__':
    main()
