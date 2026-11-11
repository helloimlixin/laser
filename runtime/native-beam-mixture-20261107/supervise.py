"""Run a gated mixture pilot and a matched native control, then compare outcomes."""
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from omegaconf import OmegaConf

ROOT=Path('/tmp/laser-native-beam-mixture-20261107');BASE=ROOT/'frozen-mixture'
RUNTIME=Path('/workspace/Projects/laser/runtime/native-beam-mixture-20261107')
RUN='imnet-j16384-a4096-native-beam-mix-scratch4-4a100-20261107'
NATIVE='imnet-j16384-a4096-native-control-scratch4-4a100-20261107'
CHILD=None
CLOUDS={}


def stop(sig,_):
    if CHILD is not None and CHILD.poll() is None:
        try:os.killpg(CHILD.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    write(ROOT/'status.json',dict(phase='interrupted',signal=sig,time=time.time()))
    raise SystemExit(128+sig)


def read(p):return json.loads(p.read_text()) if p.exists() else None


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending')
    t.write_text(json.dumps(v,indent=2,default=str)+'\n');t.replace(p)


def environment(base,out):
    env=dict(os.environ)
    for k in ['WANDB_SERVICE','_WANDB_SERVICE']:env.pop(k,None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com',WANDB_MODE='online',WANDB_CONSOLE='off',
        WANDB_DISABLE_GIT='true',WANDB_CACHE_DIR=str(ROOT/'wandb-cache'),
        LASER_STAGE2_OUTPUT=str(out),CUDA_VISIBLE_DEVICES='0,1,2,3',PYTHONUNBUFFERED='1',
        TORCH_HOME=str(ROOT/'torch-cache'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
        LASER_FUSED_ADAMW='1',LASER_TRAIN_SDPA='1',LASER_SDPA_MIN_SEQ='16',
        LASER_COMPILE_NATIVE_LOSS='1',LASER_COMPILE_LOSS_DYNAMIC='1',
        LASER_EXACT_BEAM_KERNELS='1',LASER_FAST_BEAM_TOPK='0',
        TORCHINDUCTOR_CACHE_DIR='/tmp/laser-speed-aligned-20261105/inductor-cache',
        TORCHINDUCTOR_COMPILE_THREADS='2',NCCL_NVLS_ENABLE='0',PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def command(base,epochs,source=None,smoke=0):
    cmd=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(base/'train.py'),'--microbatch','128','--epochs',str(epochs)]
    if source:cmd+=['--resume',str(source)]
    if smoke:cmd+=['--smoke-updates',str(smoke)]
    return cmd


def child(cmd,env,out,phase):
    global CHILD
    out.mkdir(parents=True,exist_ok=True)
    with (out/(phase+'.log')).open('ab',buffering=0) as log:
        CHILD=subprocess.Popen(cmd,env=env,cwd=BASE,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        started=time.time()
        while CHILD.poll() is None:
            state=dict(phase=phase,child_pid=CHILD.pid,time=time.time(),
                progress=read(out/'train/progress-rank0.json'),checkpoint=read(out/'checkpoint.json'),
                generation=read(out/'generation-progress.json'),target_epochs=int(cmd[cmd.index('--epochs')+1]) if '--epochs' in cmd else None)
            write(out/'status.json',state);write(ROOT/'status.json',dict(state,output=str(out)))
            activity=[out/(phase+'.log'),out/'train/progress-rank0.json',out/'generation-progress.json',out/'checkpoint-storage-wait.json']
            last=max([started]+[p.stat().st_mtime for p in activity if p.exists()])
            if time.time()-last>1800:
                os.killpg(CHILD.pid,signal.SIGTERM)
                try:CHILD.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL)
            try:CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:pass
    return CHILD.returncode


def latest(out,run):
    import torch
    for p in sorted((out/'persistent-train/checkpoints').glob('step-*.pt'),reverse=True):
        try:
            s=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            assert s['run_id']==run and len(s['optimizer']['state'])==780
            step=int(s['global_step']);assert all(int(v['step'])==step for v in s['optimizer']['state'].values())
            del s;return p,step
        except Exception as e:write(out/'rejected-checkpoints'/f'{p.stem}.json',dict(error=str(e)))
    return None,0


def run_job(base,out,run,epochs=4):
    out.mkdir(exist_ok=True)
    uploader=CLOUDS.get(str(out))
    if uploader is None or uploader.poll() is not None:
        with (out/'cloud-worker.log').open('ab',buffering=0) as log:
            uploader=subprocess.Popen([sys.executable,'-u',str(RUNTIME/'cloud_sync.py'),
                '--base',str(base),'--output',str(out)],stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        CLOUDS[str(out)]=uploader
    failures=0;attempt=0
    while True:
        source,step=latest(out,run)
        cmd=command(base,epochs,source)
        write(out/'launch.json',dict(command=cmd,run_id=run,source_step=step,
            fresh=source is None,target_epochs=epochs,target_steps=epochs*626,
            global_batch=2048,microbatch=128,accumulation_steps=4,
            native_weight=float(OmegaConf.load(base/'inputs/recipe.yaml').targets.native_weight),
            url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+run,
            attempt=attempt,time=time.time(),uploader_pid=uploader.pid))
        code=child(cmd,environment(base,out),out,'training')
        done=read(out/'completed.json')
        if code==0 and done and done['global_step']==epochs*626:return done
        progress=read(out/'train/progress-rank0.json') or {}
        failures=0 if progress.get('train/global_step',step)>step else failures+1
        write(out/'last-failure.json',dict(returncode=code,progress=progress,attempt=attempt))
        if failures>=3:raise RuntimeError('Three consecutive pilot startup failures')
        attempt+=1;time.sleep(15*max(1,failures))


def native_control(report):
    target=ROOT/'frozen-native-control'
    if not target.exists():
        target.mkdir()
        for name in ['source','upstream','runtime']:
            shutil.copytree(BASE/name,target/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.caches'))
        (target/'inputs').mkdir()
        for p in (BASE/'inputs').iterdir():
            if p.is_file():(target/'inputs'/p.name).hardlink_to(p)
        recipe=OmegaConf.load(BASE/'inputs/recipe.yaml')
        recipe.run_id=NATIVE;recipe.name='ImageNet4096 | native T0.5 control | matched fresh4'
        recipe.targets.native_weight=1.;recipe.targets.measured_entropy=report['results']['1.0']['teacher_entropy']
        recipe.tokenizer.teacher_sampled_rfid=report['results']['1.0']['rfid']
        # Replace the hardlink before writing a run-specific recipe.
        (target/'inputs/recipe.yaml').unlink();OmegaConf.save(recipe,target/'inputs/recipe.yaml')
        (target/'inputs/execution.json').unlink()
        write(target/'inputs/execution.json',dict(read(BASE/'inputs/execution.json'),native_control=True))
        shutil.copyfile(BASE/'train.py',target/'train.py')
        (target/'data').symlink_to(BASE/'data',target_is_directory=True)
    return target


def main():
    lock=(ROOT/'supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,stop)
    while not (ROOT/'quality/full-results.json').exists():time.sleep(5)
    report=read(ROOT/'quality/full-results.json');assert report['passed'],'No mixture met both reconstruction and entropy gates'
    subprocess.run([sys.executable,'-u',str(RUNTIME/'publish_quality.py')],check=True,
        stdout=(ROOT/'quality/publish.log').open('ab'),stderr=subprocess.STDOUT)
    native=native_control(report)
    # Smoke weights are discarded; both full pilots start from the same seed.
    smoke=ROOT/'smoke';out=ROOT/'production-nvme'
    code=child(command(BASE,4,smoke=3),dict(environment(BASE,out),LASER_PRIVATE_SMOKE_DIR=str(smoke)),out,'smoke')
    assert code==0 and read(smoke/'verification.json')['passed']
    assert read(smoke/'verification.json')['all_adam_steps']==3
    for rank in range(4):
        t=read(smoke/f'targets-rank{rank}.json')
        assert t['prefixes_from_same_teacher_distribution'] and t['full_vocabulary_native_component']
        assert t['entropy']>.2 and not t['near_best_error_mask']
        assert read(smoke/f'generation-interface-rank{rank}.json')['passed']
    for p in (out/'smoke/checkpoints').glob('*.pt'):p.unlink()
    run_job(BASE,out,RUN)
    alpha=report['selected_native_weight']
    if alpha<1:
        run_job(native,ROOT/'native-control',NATIVE)
        initial_mix=read(out/'train/initial-model-sha256.json')['sha256']
        initial_native=read(ROOT/'native-control/train/initial-model-sha256.json')['sha256']
        assert initial_mix==initial_native,'Paired pilots did not share identical initial weights'
    comparison=dict(mixture=read(out/'evaluations/epoch004.json'),
        native_control=read(ROOT/'native-control/evaluations/epoch004.json') if alpha<1 else None,
        old_beam_epoch4_fid=60.662829012099394,earlier_best_epoch4_fid=54.1376692178589,
        selected_native_weight=alpha,teacher_rfid=report['results'][str(alpha)]['rfid'],
        teacher_entropy=report['results'][str(alpha)]['teacher_entropy'],paired_initial_weights_verified=alpha<1)
    fid=comparison['mixture']['eval/fid']
    control_fid=comparison['native_control']['eval/fid'] if alpha<1 else 54.1376692178589
    promote=fid<min(54.1376692178589,control_fid)-.5
    comparison['promoted_to_100_epochs']=promote
    write(ROOT/'experiment-comparison.json',comparison)
    if promote:run_job(BASE,out,RUN,epochs=100)
    write(ROOT/'status.json',dict(phase='completed',comparison=comparison,time=time.time()))


if __name__=='__main__':
    try:main()
    except Exception as error:
        write(ROOT/'failure.json',dict(error=str(error),time=time.time()));raise
