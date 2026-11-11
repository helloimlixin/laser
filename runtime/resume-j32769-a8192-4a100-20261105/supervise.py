"""Benchmark, resume epoch2, and stop the original joint32769 recipe at epoch4."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE=Path('/tmp/laser-j32769-a8192-20261105/frozen')
OUT=BASE.parent/'production-nvme'
PUBLIC=Path('/workspace/Projects/laser/outputs/imnet-j32769-a8192-native-e2to4-4a100-20261105')
RUN='imnet-j32769-a8192-native-e2to4-4a100-20261105'
SOURCE='imnet-j32769-a8192-upstream-native-scratch-2b300-20261031'
CHILD=None
STOP=False


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    p=path.with_suffix('.pending');p.write_text(json.dumps(value,indent=2,default=str)+'\n');p.replace(path)


def status(phase,**extra):
    write(OUT/'status.json',dict(phase=phase,time=time.time(),supervisor_pid=os.getpid(),
        run_id=RUN,source_run=SOURCE,world_size=4,hardware='4 NVIDIA A100-SXM4-80GB',
        global_batch=2048,target_epoch=4,target_step=2504,scheduler_horizon=62600,
        progress=read(OUT/'train/progress-rank0.json'),checkpoint=read(OUT/'checkpoint.json'),
        storage='Local NVMe and verified W&B cloud commits',**extra))


def stop(sig,_):
    global STOP
    STOP=True
    if CHILD is not None and CHILD.poll() is None:os.killpg(CHILD.pid,signal.SIGTERM)
    status('interrupted',signal=sig)
    raise SystemExit(128+sig)


def environment():
    env=dict(os.environ)
    for key in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(key,None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_MODE='online',WANDB_CONSOLE='off',
        CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONUNBUFFERED='1',LASER_STAGE2_OUTPUT=str(OUT),
        TORCH_HOME='/tmp/laser-beam16-a100-20261105/frozen/torch-cache',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        LASER_FUSED_ADAMW='1',LASER_TRAIN_SDPA='0',LASER_COMPILE_NATIVE_LOSS='0',
        NCCL_NVLS_ENABLE='0',PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def command(mb,source,new=False):
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(BASE/'train.py'),'--microbatch',str(mb),'--epochs','4','--resume',str(source),
        '--run-id',RUN]
    if new:cmd.append('--new-wandb-run')
    return cmd


def child(cmd,env,phase,attempt=0):
    global CHILD
    with (OUT/(phase+'.log')).open('ab',buffering=0) as log:
        CHILD=subprocess.Popen(cmd,cwd=BASE,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        started=time.time()
        while CHILD.poll() is None:
            state=phase
            p=OUT/'generation-progress.json';q=OUT/'train/progress-rank0.json'
            if phase=='training' and p.exists() and q.exists() and p.stat().st_mtime>q.stat().st_mtime:
                g=read(p)
                if g['generated_on_rank']<g['samples_on_rank']:state='evaluating'
            status(state,child_pid=CHILD.pid,attempt=attempt,restarts=attempt)
            activity=[OUT/'train/progress-rank0.json',OUT/'generation-progress.json',OUT/(phase+'.log')]
            last=max([started]+[p.stat().st_mtime for p in activity if p.exists()])
            if time.time()-last>1800:
                os.killpg(CHILD.pid,signal.SIGTERM)
                try:CHILD.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL)
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
    return CHILD.returncode


def verify(folder,mb):
    proof=read(folder/'verification.json')
    assert proof['passed'] and proof['global_step']==proof['all_adam_steps']==1255
    assert proof['optimizer_state_count']==780 and proof['frozen_tokenizer_unchanged']
    for rank in range(4):
        x=read(folder/f'startup-rank{rank}.json')
        assert not x['fresh_stage2'] and not x['fresh_optimizer']
        assert x['restored_optimizer_age']==x['global_step']==x['scheduler_step_at_start']==1252
        assert x['optimizer_state_count_at_start']==780 and x['scheduler_horizon']==62600
        assert x['global_batch']==2048 and x['world_size']==4 and x['microbatch']==mb
        assert x['gradient_clip_max_norm']==1. and x['training_autocast_dtype']=='float16'
        assert x['grad_scaler_enabled'] and x['soft_labels'] and x['stochastic_codes']
        t=read(folder/f'targets-rank{rank}.json')
        assert t['vocabulary']==32769 and t['stochastic_prefixes'] and t['soft_targets']
        assert t['temperature']==[0.15819901962280272,0.13549085464477534,0.08794634933471679,0.055382978057861326]
    return proof


def checkpoint():
    import torch
    candidates=sorted((OUT/'persistent-train/checkpoints').glob('step-*.pt'),reverse=True)
    candidates.append(BASE/'inputs/resume-checkpoint.pt')
    for path in candidates:
        try:
            s=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            step=int(s['global_step'])
            assert s['run_id'] in [RUN,SOURCE] and len(s['optimizer']['state'])==780
            assert all(int(v['step'])==step for v in s['optimizer']['state'].values())
            del s
            return path,step
        except Exception as e:write(OUT/'rejected-checkpoints'/f'{path.stem}.json',dict(error=str(e)))
    raise RuntimeError('No valid continuation checkpoint')


def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    assert read(BASE.parent/'checkpoint-verification.json')['passed']
    results=read(OUT/'execution-benchmarks.json') or []
    for mb in [32,64]:
        if any(x['microbatch']==mb for x in results):continue
        phase=f'benchmark-mb{mb}';folder=BASE/phase
        code=child(command(mb,BASE/'inputs/resume-checkpoint.pt')+['--smoke-updates','1255'],
            dict(environment(),LASER_PRIVATE_SMOKE_DIR=str(folder)),phase)
        row=dict(microbatch=mb,returncode=code,log=str(OUT/(phase+'.log')))
        if code==0:
            proof=verify(folder,mb)
            p=read(folder/'progress-rank0.json')
            row.update(images_per_second=p['train/images_per_second'],peak_allocated_gib=p['peak_allocated_gib'],verification=proof)
        else:
            tail=(OUT/(phase+'.log')).read_text()[-12000:]
            row['out_of_memory']='out of memory' in tail.lower() or 'OutOfMemoryError' in tail
            row['error_tail']=tail[-6000:]
            if not row['out_of_memory']:
                write(OUT/'failed-benchmark.json',row)
                raise RuntimeError(f'Unexpected benchmark failure: {phase}')
        results.append(row);write(OUT/'execution-benchmarks.json',results)
        for p in (OUT/'smoke/checkpoints').glob('*.pt'):p.unlink()
    selected=max((x for x in results if x['returncode']==0),key=lambda x:x['images_per_second'])
    mb=selected['microbatch'];write(OUT/'selected-execution.json',selected)
    write(PUBLIC/'selected-execution.json',selected)
    execution=read(BASE/'inputs/execution.json');execution.update(microbatch=mb,accumulation_steps=2048//(4*mb))
    write(BASE/'inputs/execution.json',execution)
    attempt=0;failures=0
    while not STOP:
        source,step=checkpoint()
        new=not (OUT/'wandb.json').exists()
        cmd=command(mb,source,new=new)
        launch=dict(command=cmd,source_checkpoint=str(source),source_step=step,run_id=RUN,
            target_epoch=4,target_step=2504,scheduler_horizon=62600,global_batch=2048,
            world_size=4,microbatch=mb,accumulation_steps=2048//(4*mb),
            original_teacher=True,gradient_clip_max_norm=1.,precision='float16',
            source_epoch1_fid=89.12546967078106,source_epoch2_fid=73.73770937849014,
            started_at=time.time(),attempt=attempt,
            baseline_epoch4_fid=54.1376692178589,baseline_epoch4_inception_score=20.209081649780273,
            url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+RUN)
        write(OUT/'launch.json',launch);write(OUT/'attempts'/f'{attempt:04d}.json',launch)
        started=time.time();code=child(cmd,environment(),'training',attempt)
        complete=read(OUT/'completed.json')
        if code==0 and complete and complete['global_step']==2504:
            evaluation=read(OUT/'evaluations/epoch004.json')
            result=dict(evaluation=evaluation,baseline_fid=54.1376692178589,
                baseline_inception_score=20.209081649780273,
                fid_delta=evaluation['eval/fid']-54.1376692178589,
                inception_delta=evaluation['eval/inception_score']-20.209081649780273,
                stopped_at_budget=True)
            write(OUT/'matched-epoch4-comparison.json',result)
            write(PUBLIC/'matched-epoch4-comparison.json',result)
            status('completed_pilot',comparison=result);return
        p=OUT/'train/progress-rank0.json';progress=read(p) or {}
        moved=p.exists() and p.stat().st_mtime>=started and progress.get('train/global_step',step)>step
        failures=0 if moved else failures+1
        write(OUT/'attempts'/f'{attempt:04d}-exit.json',dict(returncode=code,progress=progress,time=time.time()))
        if failures>=5:raise RuntimeError('Five consecutive startup failures')
        attempt+=1;status('restarting',returncode=code,restarts=attempt)
        time.sleep(min(120,15*max(1,failures)))


if __name__=='__main__':
    try:main()
    except Exception as e:
        status('failed',error=str(e))
        raise
