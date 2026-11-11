"""Benchmark disposable continuations, then resume the same run with the winner."""
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE=Path('/tmp/laser-aligned-beam451-20261105/frozen')
PROD=BASE.parent/'production-nvme'
OUT=Path('/tmp/laser-speed-aligned-20261105')
ROOT=Path('/workspace/Projects/laser')
RUNTIME=ROOT/'runtime/accelerate-aligned-20261105'
SUPERVISOR=ROOT/'runtime/aligned-beam451-20261105/supervise.py'
CHILD=None


def read(path):return json.loads(path.read_text()) if path.exists() else None


def write(path,data):
    temp=path.with_suffix('.pending');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def env():
    spec=importlib.util.spec_from_file_location('current_supervisor',SUPERVISOR)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    value=m.environment()
    value.update(TORCHINDUCTOR_CACHE_DIR=str(OUT/'inductor-cache'),TORCHINDUCTOR_COMPILE_THREADS='2',
        LASER_COMPILE_LOSS_DYNAMIC='1',LASER_STAGE2_OUTPUT=str(OUT/'bench-output'))
    return value


def verify(folder,microbatch,step):
    proof=read(folder/'verification.json')
    assert proof['passed'] and proof['global_step']==proof['all_adam_steps']==step+5
    assert proof['frozen_tokenizer_unchanged'] and proof['optimizer_state_count']==780
    for rank in range(4):
        initial=read(folder/f'startup-rank{rank}.json')
        assert initial['global_step']==initial['scheduler_step_at_start']==step
        assert initial['restored_optimizer_age']==step and not initial['fresh_stage2'] and not initial['fresh_optimizer']
        assert initial['global_batch']==2048 and initial['world_size']==4 and initial['microbatch']==microbatch
        assert initial['scheduler_horizon']==62600 and initial['training_autocast_dtype']=='float16'
        assert initial['gradient_clip_max_norm']==1. and initial['teacher_beam_width']==16
        target=read(folder/f'targets-rank{rank}.json')
        assert target['temperature']==2. and target['prefixes_from_same_teacher_distribution']
        assert target['soft_targets'] and not target['native_greedy_soft_teacher_used']
        assert read(folder/f'generation-interface-rank{rank}.json')['passed']
    return proof


def benchmark(name,mb,checkpoint,step,*,sdpa=False,compiled=False,min_seq=1,profile=False):
    global CHILD
    folder=OUT/name;folder.mkdir(exist_ok=True)
    args=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(RUNTIME/'bench_train.py'),'--microbatch',str(mb),'--epochs','100',
        '--resume',str(checkpoint),'--smoke-updates',str(step+5)]
    execution=env()
    execution.update(LASER_TRAIN_SDPA=str(int(sdpa)),LASER_COMPILE_NATIVE_LOSS=str(int(compiled)),
        LASER_SDPA_MIN_SEQ=str(min_seq),LASER_PRIVATE_SMOKE_DIR=str(folder),LASER_PROFILE_TRAINING=str(int(profile)))
    write(OUT/'status.json',dict(phase='benchmark',candidate=name,microbatch=mb,time=time.time()))
    with (OUT/(name+'.log')).open('ab',buffering=0) as log:
        CHILD=subprocess.Popen(args,cwd=BASE,env=execution,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=CHILD.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid,signal.SIGTERM)
            try:CHILD.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
            code=-9
    row=dict(name=name,microbatch=mb,returncode=code,sdpa=sdpa,compiled_loss=compiled,min_sequence_for_sdpa=min_seq)
    if code==0:
        verify(folder,mb,step)
        progresses=[read(folder/f'progress-rank{rank}.json') for rank in range(4)]
        row.update(images_per_second=min(p['train/images_per_second'] for p in progresses),
            peak_allocated_gib=max(p['peak_allocated_gib'] for p in progresses),
            passed=True,teacher_unchanged=True,optimizer_and_scheduler_preserved=True)
        if profile:row['profile']=[read(folder/f'timings-rank{r}.json') for r in range(4)]
    else:
        tail=(OUT/(name+'.log')).read_text()[-10000:]
        row.update(passed=False,out_of_memory='out of memory' in tail.lower(),error_tail=tail[-3000:])
    for p in (OUT/'bench-output/smoke/checkpoints').glob('*.pt'):p.unlink()
    results=read(OUT/'benchmarks.json') or [];results.append(row);write(OUT/'benchmarks.json',results)
    print(json.dumps(row),flush=True)
    return row


def resume(selected):
    code=SUPERVISOR.read_text()
    for key,value in [('LASER_TRAIN_SDPA',str(int(selected['sdpa']))),
                      ('LASER_COMPILE_NATIVE_LOSS',str(int(selected['compiled_loss'])))]:
        old=key+"='0'";new=key+"='"+value+"'"
        assert code.count(old)==1
        code=code.replace(old,new)
    old="LASER_TRAIN_SDPA='"+str(int(selected['sdpa']))+"',"
    new=old+" LASER_SDPA_MIN_SEQ='"+str(selected['min_sequence_for_sdpa'])+"', LASER_COMPILE_LOSS_DYNAMIC='1',"
    code=code.replace(old,new)
    code=code.replace("LASER_FUSED_ADAMW='1',", "LASER_FUSED_ADAMW='1', TORCHINDUCTOR_CACHE_DIR='/tmp/laser-speed-aligned-20261105/inductor-cache', TORCHINDUCTOR_COMPILE_THREADS='2',")
    compile(code,str(SUPERVISOR),'exec');SUPERVISOR.write_text(code)
    shutil.copyfile(SUPERVISOR,BASE/'runtime/supervise.py')
    write(PROD/'selected-execution.json',selected)
    # The supervisor selects from its benchmark ledger on every restart.
    ledger=read(PROD/'execution-benchmarks.json')
    ledger.append(dict(selected,verification=read(OUT/selected['name']/'verification.json')))
    write(PROD/'execution-benchmarks.json',ledger)
    execution=read(BASE/'inputs/execution.json')
    execution.update(microbatch=selected['microbatch'],accumulation_steps=math.ceil(2048/(4*selected['microbatch'])),
        training_sdpa=selected['sdpa'],compiled_native_soft_loss=selected['compiled_loss'],
        sdpa_min_sequence_length=selected['min_sequence_for_sdpa'],
        speed_optimization_report=str(OUT/'benchmarks.json'))
    write(BASE/'inputs/execution.json',execution)
    for name in ['benchmarks.json','parity.json']:
        shutil.copyfile(OUT/name,BASE/'inputs'/('speed-'+name))
    for path in RUNTIME.glob('*.py'):
        target=BASE/'runtime'/'speed';target.mkdir(exist_ok=True);shutil.copyfile(path,target/path.name)
    with (PROD/'supervisor.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(SUPERVISOR)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'resumed.json',dict(supervisor_pid=p.pid,selected=selected,time=time.time(),target_epochs=100))


def main():
    lock=(OUT/'optimization.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (OUT/'stopped.json').exists():time.sleep(2)
    stopped=read(OUT/'stopped.json');checkpoint=Path(stopped['stopped_at_checkpoint']['path'])
    step=stopped['stopped_at_checkpoint']['global_step']
    assert step>=2504
    time.sleep(10) # Allow torchrun's children to release their CUDA contexts.
    fallback=dict(name='native-mb96',microbatch=96,sdpa=False,compiled_loss=False,min_sequence_for_sdpa=1,
        images_per_second=310.,returncode=0,passed=True)
    selected=fallback
    try:
        assert read(OUT/'parity.json')['passed']
        baseline=benchmark('native-mb96',96,checkpoint,step,profile=True)
        assert baseline['passed'];selected=baseline
        # Isolate the two kernel changes before combining them and scaling up.
        candidates=[benchmark('compiled-ce-mb96',96,checkpoint,step,compiled=True),
                    benchmark('sdpa-compiled-mb128',128,checkpoint,step,compiled=True,sdpa=True)]
        combined=candidates[-1]
        if not combined['passed']:
            candidates.append(benchmark('sdpa-compiled-mb112',112,checkpoint,step,compiled=True,sdpa=True))
        elif combined['peak_allocated_gib']<68:
            candidates.append(benchmark('sdpa-compiled-mb192',192,checkpoint,step,compiled=True,sdpa=True))
        elif combined['images_per_second']<=candidates[0].get('images_per_second',0)*1.03:
            candidates.append(benchmark('body-sdpa-compiled-mb128',128,checkpoint,step,compiled=True,sdpa=True,min_seq=16))
        best=max([baseline]+[r for r in candidates if r['passed']],key=lambda r:r['images_per_second'])
        if best['images_per_second']>=baseline['images_per_second']*1.03:selected=best
        write(OUT/'selection.json',dict(selected=selected,baseline=baseline,
            relative_speedup=selected['images_per_second']/baseline['images_per_second']))
    except Exception as error:
        write(OUT/'failure.json',dict(error=str(error),resuming_with=selected,time=time.time()))
    finally:
        # Even a failed optimization must not leave the 100-epoch run stopped.
        resume(selected)


if __name__=='__main__':main()
