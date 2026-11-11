"""Benchmark and validate the aligned CC3M recipe, then supervise the 100-epoch run."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,zipfile
from pathlib import Path
import yaml

ROOT=Path('/workspace/Projects/laser')
BASE=Path('/tmp/laser-cc3m-aligned-20261106')
RUN='cc3m-j16384-beam16-aligned-soft-650m-scratch100-4h100-20261106'
OUT=ROOT/'outputs'/RUN
CONFIG=BASE/'config.yaml'
CHILD=None
STOPPING=False


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.pending');tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)


def descendants(pid):
    result=[]
    try:children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:return result
    for child in children:result.extend(descendants(int(child)));result.append(int(child))
    return result


def stop(sig,_):
    global STOPPING
    STOPPING=True
    if CHILD is not None:
        for pid in descendants(CHILD.pid)+[CHILD.pid]:
            try:
                cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                if b'src.training.cc3m_joint' in cmd:os.kill(pid,signal.SIGTERM)
            except (FileNotFoundError,ProcessLookupError):pass
    status('stopping_by_user',signal=sig)


def status(phase,**extra):
    cfg=yaml.safe_load(CONFIG.read_text())
    value=dict(phase=phase,run_id=RUN,time=time.time(),supervisor_pid=os.getpid(),
        child_pid=CHILD.pid if CHILD else None,epochs=100,target_step=(cfg['train_items']//2048)*100,
        hardware='4 NVIDIA H100 80GB HBM3',global_batch=2048,
        microbatch=cfg['microbatch'],accumulation=cfg['accumulation'],
        image_teacher='beam16 aligned conditional soft posterior',sequence_temperature=2.,
        latest_progress=read(BASE/'live/train/progress-rank0.json'),staging=read(BASE/'staging.json'),
        checkpoint=read(BASE/'checkpoints/last.json'),**extra)
    write(OUT/'status.json',value)
    for name in ['benchmark.json','teacher-audit-rank0.json','wandb.json','completed.json']:
        v=read(BASE/'live'/name)
        if v is not None:write(OUT/name,v)


def environment():
    env=dict(os.environ)
    for key in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(key,None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_MODE='online',WANDB_DISABLE_GIT='true',
        WANDB_CACHE_DIR=str(BASE/'wandb-cache'),WANDB_DATA_DIR=str(BASE/'wandb-data'),
        WANDB_ARTIFACT_DIR=str(BASE/'wandb-artifacts'),TORCH_HOME=str(BASE/'torch-cache'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',RQ_UPSTREAM_ROOT=str(BASE/'upstream'),PYTHONPATH=str(BASE/'source'),
        CUDA_VISIBLE_DEVICES='0,1,2,3',PYTORCH_ALLOC_CONF='expandable_segments:True',
        LASER_FAST_BEAM_TOPK='0',LASER_CC3M_CHECKPOINT_MIRROR='0',NCCL_DEBUG='WARN')
    return env


def execute(phase,extra=(),distributed=True):
    global CHILD
    prefix=[sys.executable,'-u']
    if distributed:prefix+=['-m','torch.distributed.run','--standalone','--nproc-per-node=4']
    command=prefix+['-m','src.training.cc3m_joint','--config',str(CONFIG)]+list(extra)
    with (BASE/'logs'/(phase+'.log')).open('ab',buffering=0) as log:
        CHILD=subprocess.Popen(command,cwd=BASE,env=environment(),stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while CHILD.poll() is None:
            status(phase,log=str(BASE/'logs'/(phase+'.log')))
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
    return CHILD.returncode


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'launch.json').exists(),'Inspect the existing production launch first'
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    assert read(BASE/'gpu-teacher-parity.json')['passed']
    while not (BASE/'initial-shards-ready.json').exists():
        status('staging_initial_shards');time.sleep(10)
    if not (BASE/'live/benchmark.json').exists():
        code=execute('benchmark',['--benchmark'],distributed=False)
        if code:raise RuntimeError(f'Benchmark exited {code}; see logs/benchmark.log')
    cfg=yaml.safe_load(CONFIG.read_text())
    assert cfg['world_size']*cfg['microbatch']*cfg['accumulation']==2048
    assert cfg['image_teacher']=='beam_aligned' and cfg['beam_teacher']['sequence_temperature']==2.
    source=BASE/'source';repro=BASE/'live/reproducibility'
    for name in ['config.yaml','data-manifest.json','gpu-teacher-parity.json']:
        shutil.copyfile(BASE/name,repro/name)
    for name in ['calibration.json','full-results.json','speed-exact-beam-parity.json']:
        shutil.copyfile(BASE/'inputs'/name,repro/name)
    (ROOT/'configs/stage2'/f'{RUN}.yaml').write_text('# @package _global_\n'+CONFIG.read_text())
    manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}
    write(repro/'code-manifest.json',manifest)
    with zipfile.ZipFile(repro/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(source/name,name)
        for p in (BASE/'upstream').rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,'upstream/'+str(p.relative_to(BASE/'upstream')))
    if not (BASE/'live/preflight/passed.json').exists():
        code=execute('preflight',['--smoke'])
        if code:raise RuntimeError(f'Four-GPU preflight exited {code}; see logs/preflight.log')
    proof=read(BASE/'live/preflight/passed.json')
    assert proof['passed'] and proof['steps']==2 and proof['world_size']==4
    assert proof['replicas_equal'] and proof['model_optimizer_roundtrip'] and proof['conditioned_generation']
    assert proof['independent_rank_rng'] and proof['preview_preserves_training_rng']
    for rank in range(4):
        startup=read(BASE/f'live/preflight/startup-rank{rank}.json')
        audit=read(BASE/f'live/teacher-audit-rank{rank}.json')
        assert startup['global_step']==0 and startup['fresh_optimizer'] and not startup['stage2_checkpoint_loaded']
        assert audit['passed'] and audit['beam_width']==16 and audit['sequence_temperature']==2.
        assert audit['ar_joint_probability_verified'] and not audit['native_greedy_teacher_used']
        write(repro/f'preflight-teacher-rank{rank}.json',audit)
    write(repro/'preflight.json',proof);write(OUT/'preflight.json',proof)
    assert not (BASE/'checkpoints/last.pt').exists(),'Production must discard benchmark/preflight weights'
    write(OUT/'launch.json',dict(run_id=RUN,epochs=100,target_step=(cfg['train_items']//2048)*100,
        stage2_initialization='fresh_random',model='upstream CC3M 650M text prefix',
        world_size=4,microbatch=cfg['microbatch'],accumulation=cfg['accumulation'],global_batch=2048,
        beam_width=16,sequence_temperature=2.,teacher='aligned conditional soft posterior',
        source_teacher_artifact=cfg['beam_teacher']['source_artifact'],preflight_passed=True,
        preflight_weights_discarded=True,config=str(CONFIG),log=str(BASE/'logs/training.log'),time=time.time()))
    for attempt in range(1,5):
        if STOPPING:status('stopped_by_user');return
        code=execute('training' if attempt==1 else f'recovery-{attempt}')
        done=read(BASE/'live/completed.json')
        if done:
            assert done['epochs']==100 and done['global_step']==(cfg['train_items']//2048)*100
            status('completed',completion=done);return
        if STOPPING:status('stopped_by_user');return
        status('retry_after_failure',attempt=attempt,exit_code=code)
        if attempt==4:raise RuntimeError('Training failed four attempts; best committed checkpoints preserved')
        time.sleep(5)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error));raise
