"""Detached four-GPU continuation with verified preflight and committed-checkpoint recovery."""
import fcntl,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path

BASE=Path('/tmp/laser-resume-h100-20261105/frozen')
ROOT=Path('/workspace/Projects/laser')
RUN='imnet-j16384-best14574-continue100-1b300-20261104'
RUN_OUT=ROOT/'outputs'/RUN
OUT=RUN_OUT/'resume-h100-20261105'
LOCAL=BASE/'run-output'
CHILD=None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def status(phase,**extra):
    value=dict(phase=phase,run_id=RUN,time=time.time(),supervisor_pid=os.getpid(),
        child_pid=CHILD.pid if CHILD else None,target_epoch=100,target_step=62600,
        hardware='4 NVIDIA H100 80GB HBM3',
        latest_progress=read(BASE/'live-training/train/progress-rank0.json'),**extra)
    write(OUT/'status.json',value);write(RUN_OUT/'status.json',value)
    for name in ['checkpoint.json','completed.json','wandb.json','generation-progress.json','source-artifact.json','convergence-state.json']:
        value=read(LOCAL/name)
        if value is not None:
            write(OUT/name,value);write(RUN_OUT/name,value)
    for directory in ['train','evaluations']:
        source=LOCAL/directory
        if source.exists():
            for path in source.glob('*'):
                if path.is_file() and path.suffix in {'.json','.jsonl'}:
                    destination=OUT/directory/path.name
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if not destination.exists() or destination.stat().st_size!=path.stat().st_size:
                        temporary=destination.with_suffix('.pending')
                        shutil.copyfile(path,temporary);temporary.replace(destination)


def environment():
    env=dict(os.environ)
    for key in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(key,None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_MODE='online',WANDB_DISABLE_GIT='true',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        LASER_MAX_RECOVERY_LOSS='8.25',LASER_LR_AMPLITUDE_FACTOR='0.0125',LASER_GENERATION_BATCH='64',
        LASER_EVAL_EVERY_EPOCHS='1',LASER_PRESERVE_INCEPTION_MIN='93.50654602050781',
        LASER_QUALITY_EARLY_STOP='0',LASER_STAGE2_OUTPUT=str(LOCAL),LASER_COMPILE_NATIVE_LOSS='1',
        LASER_FUSED_ADAMW='1',LASER_TRAIN_SDPA='1',LASER_DATA_WORKERS='8',
        PYTORCH_ALLOC_CONF='expandable_segments:True',TORCHINDUCTOR_COMPILE_THREADS='2',
        CUDA_VISIBLE_DEVICES='0,1,2,3',NCCL_DEBUG='WARN')
    return env


def command(checkpoint,micro):
    return [sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(BASE/'train.py'),'--microbatch',str(micro),'--epochs','100','--resume',str(checkpoint)]


def child(cmd,phase):
    global CHILD
    log=BASE/(phase+'.log')
    with log.open('ab',buffering=0) as handle:
        CHILD=subprocess.Popen(cmd,cwd=BASE,env=environment(),stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        while CHILD.poll() is None:
            status(phase,log=str(log))
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
    return CHILD.returncode


def stop(signum,_):
    if CHILD is not None and CHILD.poll() is None:os.killpg(CHILD.pid,signal.SIGINT)
    status('interrupted',signal=signum)
    raise SystemExit(128+signum)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'h100-supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    assert not (OUT/'launch.json').exists(),'Inspect previous launch before relaunch'
    start=time.monotonic()
    while not (BASE/'data/train-ready.json').exists():
        status('preparing_local_training_data',data_status=read(BASE/'data/status.json'))
        if time.monotonic()-start>2400:raise RuntimeError('Data preparation timed out; inspect data.log')
        time.sleep(10)
    reports=[read(BASE.parent/f'benchmark-{i}.json') for i in range(4)]
    assert all(reports),'Inspect incomplete hardware benchmarks'
    candidates=[r for r in reports if r['passed'] and r['sdpa']=='1' and r['peak_allocated_gib']<72]
    assert candidates,'No validated batch size fits the memory margin'
    winner=max(candidates,key=lambda r:r['median_images_per_second'])
    micro=winner['microbatch'];accumulation=2048//(4*micro)
    assert accumulation*4*micro==2048
    write(OUT/'throughput-benchmarks.json',dict(benchmarks=reports,selected=winner,
        benchmark_data='Synthetic images, same frozen tokenizer, full restored model/Adam, native soft objective',
        production_preflight_data='Verified original ImageNet JPEGs with original training transforms'))
    execution=read(BASE/'inputs/execution.json')
    execution.update(microbatch=micro,accumulation_steps=accumulation)
    write(BASE/'inputs/execution.json',execution);write(OUT/'execution.json',execution)
    initial=BASE/'inputs/resume-checkpoint.pt'
    code=child(command(initial,micro)+['--smoke-updates','46253'],'preflight')
    if code:raise RuntimeError(f'Real-data resume preflight failed ({code}); see {BASE}/preflight.log')
    proof=read(LOCAL/'smoke/verification.json');roundtrip=read(LOCAL/'smoke/checkpoint-roundtrip.json')
    assert proof['passed'] and proof['frozen_tokenizer_unchanged']
    assert proof['global_step']==proof['all_adam_steps']==46253 and proof['optimizer_state_count']==780
    init=proof['initialization']
    assert init['global_step']==init['restored_optimizer_age']==46250
    assert init['world_size']==4 and init['global_batch']==2048
    assert init['accumulation_steps']==accumulation and init['microbatch']==micro
    assert not init['fresh_stage2'] and not init['fresh_optimizer'] and init['resume_memory']['source_mmap_released']
    assert init['lr_recovery']['scheduler_step']==46250 and init['lr_recovery']['scheduler_horizon']==62600
    assert abs(init['restored_lr']-9.942649433929667e-7)<1e-16
    assert roundtrip['passed'] and roundtrip['global_step']==46253
    write(OUT/'preflight.json',proof);write(OUT/'checkpoint-roundtrip.json',roundtrip)
    write(BASE/'full-size-smoke-verification.json',proof)
    repro=LOCAL/'reproducibility';repro.mkdir(parents=True,exist_ok=True)
    for path in OUT.glob('*.json'):shutil.copyfile(path,repro/path.name)
    shutil.copyfile(BASE/'h100-runtime.patch',repro/'h100-runtime.patch')
    shutil.copyfile(BASE/'kernel-verification.json',repro/'kernel-verification.json')
    launch=dict(command=command(initial,micro),source_step=46250,target_step=62600,target_epoch=100,
        initial_lr=init['restored_lr'],hardware='4 H100 80GB',world_size=4,microbatch=micro,
        accumulation_steps=accumulation,global_batch=2048,preflight_updates_discarded=True,
        wandb_run_resumed=True,log=str(BASE/'training.log'),local_output=str(LOCAL),time=time.time())
    write(OUT/'launch.json',launch)
    code=child(launch['command'],'training')
    repeated_step=None;retries=0
    while code:
        receipt=read(LOCAL/'checkpoint.json')
        if not receipt or receipt.get('state')!='COMMITTED':raise RuntimeError(f'Trainer exited {code} before a new committed checkpoint')
        checkpoint=Path(receipt['path'])
        assert checkpoint.is_file() and checkpoint.stat().st_size==receipt['bytes']
        step=receipt['global_step'];retries=retries+1 if step==repeated_step else 1;repeated_step=step
        if retries>3:raise RuntimeError('Recovery failed three times at the same checkpoint')
        status('recovering',step=step,retry=retries)
        code=child(command(checkpoint,micro),f'recovery-{step}-{retries}')
    completed=read(LOCAL/'completed.json')
    assert completed and completed['epochs']==100 and completed['global_step']==62600
    status('completed',completion=completed)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error))
        raise
