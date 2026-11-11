"""Run the verified rewind with a baseline gate, eight-update pilot, and quality stops."""
import fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path

BASE=Path('/tmp/laser-best-rewind-4h100-20261105/frozen')
ROOT=Path('/workspace/Projects/laser')
RUN='imnet-j16384-best14574-rewind-gated-4h100-20261105'
OUT=ROOT/'outputs'/RUN
LOCAL=BASE/'run-output'
CHILD=None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def status(phase=None,**extra):
    baseline=read(LOCAL/'matched-baseline.json')
    gate=read(LOCAL/'baseline-reproduction-gate.json')
    pilot=read(LOCAL/'pilot-decision.json')
    progress=read(BASE/'live-training/train/progress-rank0.json')
    rewind=read(LOCAL/'rewound.json')
    generation=read(LOCAL/'generation-progress.json')
    if phase is None:
        if rewind:phase='rewound_to_best_quality_gate'
        elif pilot and pilot['accepted']:phase='quality_gated_training'
        elif gate and gate['accepted']:
            phase='evaluating_eight_update_trial' if progress and progress['train/global_step']>=45706 else 'eight_update_trial'
        elif baseline:phase='checking_restored_best'
        elif generation:phase='evaluating_restored_best_before_any_updates'
        else:phase='restoring_best_checkpoint'
    write(OUT/'status.json',dict(phase=phase,run_id=RUN,time=time.time(),
        supervisor_pid=os.getpid(),child_pid=CHILD.pid if CHILD else None,
        source_step=45698,source_epoch=73,target_epoch=100,
        baseline_verified=gate['accepted'] if gate else False,
        pilot_accepted=pilot['accepted'] if pilot else None,
        latest_progress=progress,generation_progress=generation,
        active_best=read(LOCAL/'persistent-train/checkpoints/best.json'),**extra))
    for name in ['matched-baseline.json','baseline-reproduction-gate.json','pilot-decision.json',
                 'rewound.json','quality-stop.json','quality-stopped.json','checkpoint.json',
                 'completed.json','wandb.json','source-artifact.json','rewind-source.json']:
        value=read(LOCAL/name)
        if value is not None:write(OUT/name,value)
    for rank in range(4):
        value=read(BASE/f'live-training/train/startup-rank{rank}.json')
        path=OUT/f'startup-rank{rank}.json'
        if value is not None and not path.exists():write(path,value)


def descendants(pid):
    result=[]
    try:children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:return result
    for child in children:
        result.extend(descendants(int(child)));result.append(int(child))
    return result


def stop(signum,_):
    if CHILD is not None:
        for pid in descendants(CHILD.pid)+[CHILD.pid]:
            try:os.kill(pid,signal.SIGINT)
            except ProcessLookupError:pass
    status('interrupted',signal=signum)
    raise SystemExit(128+signum)


def environment():
    env=dict(os.environ)
    for key in ['WANDB_SERVICE','_WANDB_SERVICE','LASER_PRESERVE_INCEPTION_MIN']:env.pop(key,None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_MODE='online',WANDB_DISABLE_GIT='true',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',
        LASER_MAX_RECOVERY_LOSS='8.25',LASER_LR_AMPLITUDE_FACTOR='0.15',LASER_GENERATION_BATCH='64',
        LASER_EVAL_EVERY_EPOCHS='1',LASER_QUALITY_EARLY_STOP='1',LASER_STAGE2_OUTPUT=str(LOCAL),
        LASER_COMPILE_NATIVE_LOSS='1',LASER_FUSED_ADAMW='1',LASER_TRAIN_SDPA='0',LASER_DATA_WORKERS='8',
        LASER_INHERIT_SOURCE_EVALUATION='0',PYTORCH_ALLOC_CONF='expandable_segments:True',
        TORCHINDUCTOR_COMPILE_THREADS='2',CUDA_VISIBLE_DEVICES='0,1,2,3',NCCL_DEBUG='WARN')
    return env


def main():
    global CHILD
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Inspect an existing launch before relaunch'
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    proof=read(BASE/'rewind-prepared.json')
    assert proof['passed'] and proof['restored_step']==45698 and proof['all_adam_steps']==45698
    assert proof['source_layout_unchanged'] and proof['source_lr']==proof['restored_lr']
    assert read(BASE/'data/train-ready.json')['training_images']==1281167
    repro=LOCAL/'reproducibility';repro.mkdir(parents=True,exist_ok=True)
    for path in OUT.glob('*.json'):
        if path.name!='launch.json':(repro/path.name).write_bytes(path.read_bytes())
    (repro/'runtime.patch').write_bytes((BASE/'rewind-runtime.patch').read_bytes())
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(BASE/'train.py'),'--microbatch','128','--epochs','100',
        '--resume',str(BASE/'inputs/resume-checkpoint.pt'),'--new-wandb-run']
    log=BASE/'training.log'
    write(OUT/'launch.json',dict(command=command,run_id=RUN,source_step=45698,source_epoch=73,
        source_artifact=proof['source_artifact'],source_lr=proof['source_lr'],global_batch=2048,
        world_size=4,microbatch=128,accumulation_steps=4,baseline_before_any_updates=True,
        pilot_updates=8,long_training_requires_matched_fid_improvement=True,
        immediate_quality_stop_enabled=True,log=str(log),time=time.time()))
    with log.open('ab',buffering=0) as handle:
        CHILD=subprocess.Popen(command,cwd=BASE,env=environment(),stdin=subprocess.DEVNULL,
            stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        while CHILD.poll() is None:
            status()
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
    if CHILD.returncode:
        status('failed_best_preserved',exit_code=CHILD.returncode)
        raise RuntimeError(f'Rewind job exited {CHILD.returncode}; best source remains immutable')
    if (LOCAL/'rewound.json').exists():status('rewound_to_best_quality_gate')
    elif (LOCAL/'completed.json').exists():status('completed')
    else:raise RuntimeError('Trainer exited without a rewind or completion receipt')


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_best_preserved',error=str(error));raise
