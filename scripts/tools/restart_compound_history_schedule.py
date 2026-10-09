"""Checkpoint active compound training, retain full Adam/RNG, resume new rates."""
import argparse
import fcntl
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


def command(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('base','output','key-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    transition=(args.base/'schedule-transition.lock').open('w')
    fcntl.flock(transition,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=args.output/'training-status.json'
    previous=json.loads(status.read_text())
    old_supervisor=previous['supervisor_pid']
    old_torchrun=previous['torchrun_pid']
    old_phase=previous['phase']
    run_id=previous['run_id']
    if str(args.base) not in command(old_supervisor) or 'supervise_physical_compound_training' not in command(old_supervisor):
        raise RuntimeError('Active training supervisor identity changed')
    old_ranks=[json.loads(p.read_text())['pid'] for p in
               sorted((args.output/'verification'/old_phase).glob('process-rank*.json'))]
    if len(old_ranks)!=8 or any(str(args.base) not in command(pid) for pid in old_ranks):
        raise RuntimeError('Active eight-rank trainer identity changed')
    review=args.output/'schedule-review'
    record(review/'transition-before.json',dict(previous_status=previous,rank_pids=old_ranks,
        started_unix=time.time(),intent='preserve current model/Adam/RNG and change only LR policy'))
    ready_path=args.output/'trial-checkpoint-ready.json'
    if ready_path.exists():
        raise RuntimeError('Unexpected stale ready marker; inspect before stopping active training')
    os.kill(old_supervisor,signal.SIGTERM)
    deadline=time.monotonic()+900
    while True:
        if ready_path.exists():
            ready=json.loads(ready_path.read_text())
            saved=json.loads((args.output/'last-local-save.json').read_text())
            if ready['no_further_updates'] and ready['global_step']==saved['step']:
                break
        if time.monotonic()>deadline:
            raise RuntimeError('Timed out waiting for durable full recovery checkpoint')
        time.sleep(5)
    sys.path[:0]=[str(args.base/'source/runtime'),str(args.base/'support')]
    import torch
    from src.training.full_resume_upload import recovery_metadata
    from src.training.k4_checkpoint_io import _checkpoint_upload_source,_replace_hard_link
    from src.training.compound_history_schedule import repartition_optimizer_state
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR']=str(args.base/'checkpoint-upload-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES']='1'
    local_source=_checkpoint_upload_source(Path(saved['target']))
    payload=torch.load(local_source,map_location='cpu',mmap=True,weights_only=False)
    meta=recovery_metadata(payload)
    if meta['global_step']!=ready['global_step'] or meta['rng_ranks']!=8 or payload['scheduler']['last_epoch']!=meta['global_step']:
        raise RuntimeError('Recovery checkpoint is incomplete or has an inconsistent cursor')
    transfer=payload['compound_transfer']
    old_optimizer=payload['optimizer']
    adjusted=repartition_optimizer_state(old_optimizer,transfer['parameter_names'],transfer['new_parameter_names'])
    if adjusted['state'] is not old_optimizer['state'] or any(adjusted['state'][i] is not old_optimizer['state'][i] for i in old_optimizer['state']):
        raise RuntimeError('Adam state identity changed while partitioning groups')
    anchor=args.base/f'resume-before-history-schedule-step{meta["global_step"]}.pt'
    _replace_hard_link(local_source,anchor)
    record(review/'pre-restart-verification.json',dict(global_step=meta['global_step'],epoch=meta['epoch'],
        next_microbatch=meta['next_microbatch'],rng_ranks=meta['rng_ranks'],
        optimizer_ages=meta['compound_optimizer_ages'],saved_learning_rates=meta['saved_learning_rates'],
        scheduler_step=payload['scheduler']['last_epoch'],anchor=str(anchor),bytes=anchor.stat().st_size,
        no_further_updates=True,all_adam_tensors_preserved=True,
        group_counts=[dict(kind=g['compound_group'],parameters=len(g['params'])) for g in adjusted['param_groups']],
        previous_ready=ready,time=time.time()))
    del payload,adjusted,old_optimizer
    # Training has committed all states and is only draining cloud uploads.
    # The resumed worker retries those uploads from the same durable files.
    for pid in [*old_ranks,old_torchrun]:
        cmd=command(pid)
        if cmd and str(args.base) not in cmd:
            raise RuntimeError('Frozen training process ownership changed')
        if cmd:os.kill(pid,signal.SIGKILL)
    deadline=time.monotonic()+60
    while command(old_supervisor) or any(command(pid) for pid in old_ranks):
        if time.monotonic()>deadline:raise RuntimeError('Previous supervisor did not release training')
        time.sleep(1)
    lock=(args.base/'training-supervisor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ready_path.rename(review/'pre-restart-checkpoint-ready.json')
    env=dict(os.environ,LASER_RUN_BASE=str(args.base),LASER_PERSISTENT_BASE=str(args.output),
        LASER_ACCUMULATION='4',LASER_PHASE='compound_objective_schedule_v1',
        LASER_CHECKPOINT_STAGING_DIR=str(args.base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(args.base/'checkpoint-upload-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1',LASER_COMPILE_BLOCKS='1',LASER_PREVIEW_KEEP_OPTIMIZER='1',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',WANDB_API_KEY=args.key_file.read_text().strip(),
        WANDB_DIR=str(args.base/'wandb'),WANDB_CACHE_DIR=str(args.base/'wandb-cache'),
        WANDB_DATA_DIR=str(args.base/'wandb-data'),WANDB_CONFIG_DIR=str(args.base/'wandb-config'),
        TORCHINDUCTOR_CACHE_DIR=str(args.base/'inductor-cache'),TORCHINDUCTOR_COMPILE_THREADS='2',
        TORCH_HOME=str(args.base/'torch-cache'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        NCCL_NVLS_ENABLE='0',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
         str(args.base/'entry-objective-schedule.py'),'--config',str(args.base/'recipe.yaml')]
    with (args.output/'compound_objective_schedule.log').open('a') as stream:
        child=subprocess.Popen(cmd,cwd=args.base,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        def stop(sig,frame):
            for path in (args.output/'verification/compound_objective_schedule_v1').glob('process-rank*.json'):
                pid=json.loads(path.read_text())['pid']
                if str(args.base) in command(pid):os.kill(pid,signal.SIGTERM)
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        while child.poll() is None:
            record(status,dict(state='training',supervisor_pid=os.getpid(),torchrun_pid=child.pid,
                phase='compound_objective_schedule_v1',run_id=run_id,target_epoch=100,
                official_evaluations_only=True,schedule_revision='compound-history-discriminative-cosine-v1',
                history_peak_lr=1e-5,history_warmup_updates=200,time=time.time()))
            time.sleep(5)
    saved=json.loads((args.output/'last-local-save.json').read_text())
    record(status,dict(state='completed' if child.returncode==0 and saved['step']==62600
        else 'stopped_resumable' if child.returncode==0 else 'failed',run_id=run_id,
        exit_code=child.returncode,final_step=saved['step'],target_epoch=100,time=time.time()))
    if child.returncode:raise RuntimeError('Rescheduled training failed; checkpoint remains intact')


if __name__=='__main__':main()
