"""Profile verified batch sizes and run fresh, recoverable four-GPU pattern training."""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path('/tmp/laser-pattern1024-stage2-20261107')
BASE=ROOT/'frozen'
OUT=ROOT/'production-nvme'
RUN='imagenet-laser-pattern1024-native-scratch100-4a100-20261107'
sys.path.insert(0,str(BASE))
from runtime import launch_support as jobs


def profile(micro):
    folder=ROOT/f'smoke-micro{micro}'
    phase=f'smoke-micro{micro}'
    code=jobs.child(jobs.command(micro,smoke=3),dict(jobs.environment(),LASER_PRIVATE_SMOKE_DIR=str(folder)),phase,folder)
    if code:
        log=(OUT/(phase+'.log')).read_text()
        if 'out of memory' in log.lower():
            return dict(microbatch_limit=micro,passed=False,reason='CUDA out of memory')
        raise RuntimeError(f'{phase} failed; inspect its log')
    proof=jobs.read(folder/'verification.json')
    expected=jobs.read(ROOT/'cpu-verification.json')['parameter_tensors']
    assert proof['passed'] and proof['frozen_tokenizer_unchanged'] and proof['all_adam_steps']==3
    assert proof['optimizer_state_count']==expected
    for rank in range(4):
        initial=jobs.read(folder/f'startup-rank{rank}.json')
        assert initial['fresh_stage2'] and initial['fresh_optimizer'] and initial['global_step']==0
        assert initial['global_batch']==2048 and initial['coefficient_patterns']==1024
        assert initial['optimizer_state_count_at_start']==0 and initial['scheduler_horizon']==62600
        target=jobs.read(folder/f'targets-rank{rank}.json')
        assert target['prefixes_from_same_codec_as_targets'] and target['hard_labels']
        assert target['categorical_events_per_site']==5 and target['native_depth_steps']==4
        assert jobs.read(folder/f'generation-interface-rank{rank}.json')['passed']
    rows=[json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    speed=sum(row['train/images_per_second'] for row in rows[1:])/len(rows[1:])
    progress=jobs.read(folder/'progress-rank0.json')
    accumulation=math.ceil(512/micro)
    result=dict(passed=True,microbatch_limit=micro,largest_actual_microbatch=math.ceil(512/accumulation),
        accumulation_steps=accumulation,global_batch=2048,images_per_second=speed,
        peak_allocated_gib=progress['peak_allocated_gib'],discarded_smoke_weights=True,
        optimizer_state_count=expected,verification=proof)
    for path in (OUT/'smoke/checkpoints').glob('*.pt'):path.unlink()
    return result


def latest():
    import torch
    expected=jobs.read(ROOT/'cpu-verification.json')['parameter_tensors']
    for path in sorted((OUT/'persistent-train/checkpoints').glob('step-*.pt'),reverse=True):
        try:
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            assert state['run_id']==RUN and len(state['optimizer']['state'])==expected
            step=int(state['global_step'])
            assert all(int(v['step'])==step for v in state['optimizer']['state'].values())
            del state
            return path,step
        except Exception as error:
            jobs.write(OUT/'rejected-checkpoints'/f'{path.stem}.json',dict(error=str(error)))
    return None,0


def main():
    lock=(ROOT/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,jobs.stop)
    proof=jobs.read(ROOT/'cpu-verification.json')
    quality=jobs.read(ROOT/'tokenizer-verification.json')
    assert proof['passed'] and quality['passed'] and quality['measured_validation_rfid']<4.73
    assert jobs.read(ROOT/'preserved-baseline.json')['complete_optimizer_state_verified']
    profile_path=ROOT/'throughput-profile.json'
    if not profile_path.exists():
        results=[profile(128)]
        if results[0]['passed'] and results[0]['peak_allocated_gib']<61:
            results.append(profile(192))
        elif not results[0]['passed']:
            results.append(profile(96))
        results.append(profile(64))
        eligible=[row for row in results if row['passed'] and row['peak_allocated_gib']<74]
        assert eligible,'No batch size passed within the production memory margin'
        selected=max(eligible,key=lambda row:row['images_per_second'])
        jobs.write(profile_path,dict(selected_microbatch_limit=selected['microbatch_limit'],
            selected_actual_microbatch_max=selected['largest_actual_microbatch'],results=results,
            global_batch=2048,world_size=4,smoke_weights_and_optimizer_discarded=True))
    micro=jobs.read(profile_path)['selected_microbatch_limit']
    from omegaconf import OmegaConf
    recipe=OmegaConf.load(BASE/'inputs/recipe.yaml')
    recipe.microbatch=micro
    OmegaConf.save(recipe,BASE/'inputs/recipe.yaml')
    shutil.copyfile(BASE/'inputs/recipe.yaml',
        '/workspace/Projects/laser/configs/stage2/imagenet-laser-pattern1024-native-scratch100-4a100-20261107.yaml')
    for name in ['cpu-verification.json','throughput-profile.json','preserved-baseline.json']:
        shutil.copyfile(ROOT/name,BASE/'inputs'/name)
    with (OUT/'cloud-worker.log').open('ab',buffering=0) as log:
        uploader=subprocess.Popen([sys.executable,'-u',str(BASE/'runtime/cloud_sync.py')],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    attempt=failures=0
    while True:
        source,step=latest()
        cmd=jobs.command(micro,resume=source)
        jobs.write(OUT/'launch.json',dict(run_id=RUN,command=cmd,fresh=source is None,source_step=step,
            target_epochs=100,target_steps=62600,world_size=4,global_batch=2048,microbatch_limit=micro,
            accumulation_steps=math.ceil(512/micro),largest_actual_microbatch=math.ceil(512/math.ceil(512/micro)),
            patterns=1024,atoms=16384,categorical_events_per_site=5,native_depth_steps=4,
            measured_tokenizer_rfid=quality['measured_validation_rfid'],attempt=attempt,
            url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+RUN,
            uploader_pid=uploader.pid,time=time.time()))
        code=jobs.child(cmd,jobs.environment(),'training',OUT/'train')
        completed=jobs.read(OUT/'completed.json')
        if code==0 and completed and completed['global_step']==62600:
            jobs.write(ROOT/'status.json',dict(phase='completed',result=completed,time=time.time()))
            return
        progress=jobs.read(OUT/'train/progress-rank0.json') or {}
        failures=0 if progress.get('train/global_step',step)>step else failures+1
        jobs.write(OUT/'last-failure.json',dict(returncode=code,attempt=attempt,progress=progress))
        if failures>=3:raise RuntimeError('Three consecutive startup failures')
        attempt+=1
        time.sleep(15*max(1,failures))


if __name__=='__main__':
    try:main()
    except Exception as error:
        jobs.write(ROOT/'failure.json',dict(error=str(error),time=time.time()))
        raise
