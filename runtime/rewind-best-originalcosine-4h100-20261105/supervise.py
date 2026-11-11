"""Detached continuation to epoch100, recovering only complete durable checkpoints."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

BASE=Path('/tmp/laser-best-originalcosine-4h100-20261105/frozen')
ROOT=Path('/workspace/Projects/laser')
RUN='imnet-j16384-best14574-originalcosine-continue100-4h100-20261105'
OUT=ROOT/'outputs'/RUN
LOCAL=BASE/'run-output'
CHILD=None
STOPPING=False


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def status(phase,**extra):
    write(OUT/'status.json',dict(phase=phase,run_id=RUN,time=time.time(),
        supervisor_pid=os.getpid(),child_pid=CHILD.pid if CHILD else None,
        source_step=45698,source_epoch=73,target_epoch=100,target_step=62600,
        quality_early_stop=False,quality_pilot_enabled=False,lr_amplitude_factor=1.0,original_unscaled_cosine=True,
        latest_progress=read(BASE/'live-training/train/progress-rank0.json'),
        generation_progress=read(LOCAL/'generation-progress.json'),
        best_checkpoint=read(LOCAL/'persistent-train/checkpoints/best.json'),**extra))
    for name in ['checkpoint.json','matched-baseline.json','completed.json','wandb.json',
                 'source-artifact.json','best-inception.json','original-best.json']:
        value=read(LOCAL/name)
        if value is not None:write(OUT/name,value)
    for rank in range(4):
        value=read(BASE/f'live-training/train/startup-rank{rank}.json')
        path=OUT/f'startup-rank{rank}.json'
        if value is not None and not path.exists():write(path,value)
    evaluations=LOCAL/'evaluations'
    if evaluations.exists():
        for path in evaluations.glob('epoch*.json'):
            target=OUT/'evaluations'/path.name
            if not target.exists():write(target,read(path))


def descendants(pid):
    result=[]
    try:children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:return result
    for child in children:
        result.extend(descendants(int(child)));result.append(int(child))
    return result


def stop(signum,_):
    global STOPPING
    STOPPING=True
    if CHILD is not None:
        for pid in descendants(CHILD.pid):
            try:
                command=Path(f'/proc/{pid}/cmdline').read_bytes()
                if str(BASE/'train.py').encode() in command:os.kill(pid,signal.SIGINT)
            except (FileNotFoundError,ProcessLookupError):pass
        try:os.kill(CHILD.pid,signal.SIGINT)
        except ProcessLookupError:pass
    status('interrupted_by_user',signal=signum)
    raise SystemExit(128+signum)


def environment():
    env=dict(os.environ)
    for key in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(key,None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_MODE='online',WANDB_DISABLE_GIT='true',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        LASER_MAX_RECOVERY_LOSS='8.25',LASER_LR_AMPLITUDE_FACTOR='1.0',LASER_GENERATION_BATCH='64',
        LASER_EVAL_EVERY_EPOCHS='1',LASER_QUALITY_EARLY_STOP='0',
        LASER_PRESERVE_INCEPTION_MIN='92.96241760253906',LASER_STAGE2_OUTPUT=str(LOCAL),
        LASER_COMPILE_NATIVE_LOSS='1',LASER_FUSED_ADAMW='1',LASER_TRAIN_SDPA='0',LASER_DATA_WORKERS='8',
        LASER_INHERIT_SOURCE_EVALUATION='1',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_COMPILE_THREADS='2',CUDA_VISIBLE_DEVICES='0,1,2,3',NCCL_DEBUG='WARN')
    return env


def command(checkpoint,*,new_run=False):
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(BASE/'train.py'),'--microbatch','128','--epochs','100','--resume',str(checkpoint)]
    if new_run:cmd.append('--new-wandb-run')
    return cmd


def child(cmd,phase):
    global CHILD
    log=BASE/(phase+'.log')
    with log.open('ab',buffering=0) as handle:
        CHILD=subprocess.Popen(cmd,cwd=BASE,env=environment(),stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        while CHILD.poll() is None:
            status('training_to_epoch100',log=str(log))
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
    return CHILD.returncode


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Inspect existing launch before relaunch'
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    proof=read(BASE/'continuation-verification.json')
    assert proof['passed'] and proof['source_step']==45698 and proof['all_adam_steps']==45698
    assert proof['target_epoch']==100 and proof['target_step']==62600
    assert proof['lr_amplitude_factor']==1.0 and proof['restored_unscaled_lr']>8.4e-5
    assert proof['quality_pilot_removed'] and proof['quality_stopping_removed']
    execution=read(BASE/'inputs/execution.json')
    assert execution['repair_trial'] is False and execution['quality_guard']['enabled'] is False
    assert environment()['LASER_QUALITY_EARLY_STOP']=='0'
    assert read(BASE/'data/train-ready.json')['training_images']==1281167
    initial=BASE/'inputs/resume-checkpoint.pt'
    repro=LOCAL/'reproducibility';repro.mkdir(parents=True,exist_ok=True)
    for path in OUT.glob('*.json'):(repro/path.name).write_bytes(path.read_bytes())
    (repro/'runtime.patch').write_bytes((BASE/'continuation-runtime.patch').read_bytes())
    write(OUT/'launch.json',dict(command=command(initial,new_run=True),source_step=45698,source_epoch=73,
        source_artifact=proof['source_artifact'],historical_best_fid=14.573915136162327,
        source_lr=proof['saved_lr'],initial_lr=proof['restored_unscaled_lr'],lr_amplitude_factor=1.0,target_epoch=100,target_step=62600,global_batch=2048,
        world_size=4,microbatch=128,accumulation_steps=4,quality_early_stop=False,
        quality_pilot_enabled=False,crash_recovery=True,log=str(BASE/'training.log'),time=time.time()))
    code=child(command(initial,new_run=True),'training')
    repeated_step=None;retries=0
    while code and not STOPPING:
        receipt=read(LOCAL/'checkpoint.json')
        if receipt:
            assert receipt['state']=='COMMITTED'
            assert receipt['artifact'].split('/')[-1].startswith(RUN+'-checkpoint:')
            checkpoint=Path(receipt['path'])
            assert checkpoint.is_file() and checkpoint.stat().st_size==receipt['bytes']
            step=receipt['global_step']
        else:checkpoint=initial;step=45698
        retries=retries+1 if step==repeated_step else 1;repeated_step=step
        if retries>3:raise RuntimeError(f'Three crash recovery attempts failed at step {step}')
        status('recovering_from_committed_checkpoint',recovery_step=step,retry=retries,previous_exit_code=code)
        code=child(command(checkpoint),f'recovery-{step}-{retries}')
    completion=read(LOCAL/'completed.json')
    assert completion and completion['epochs']==100 and completion['global_step']==62600
    status('completed_epoch100',completion=completion)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error));raise
