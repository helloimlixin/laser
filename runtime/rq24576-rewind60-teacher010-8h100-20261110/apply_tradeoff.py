"""Apply the measured noise tradeoff before epoch60, preserving all train state."""
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import signal
import subprocess
import sys
import time
from omegaconf import OmegaConf

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=Path('/tmp/laser-rq24576-rewind60-teacher010-8h100-20261110')
OUT=ROOT/'outputs/imagenet-rq24576-rewind60-teacher010-8h100-20261110'
CAL=ROOT/'outputs/imagenet-rq24576-noise-tradeoff-calibration-20261110/report.json'
VAL=ROOT/'outputs/imagenet-rq24576-noisematched-latenterror-val50k-20261110'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def status(phase,**fields):write(OUT/'tradeoff-launch.json',dict(phase=phase,time=time.time(),**fields))


def restore_pause_record():
    held=OUT/'sampler-pause-held-for-calibration.json'
    if held.exists():held.replace(OUT/'sampler-pause.json')


def resume_search_on_failure():
    restore_pause_record();path=OUT/'sampler-pause.json'
    if not path.exists():return
    state=read(path)
    for pid in state['paused_workers']:
        cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and 'rq24576-ema-sampler-search-20261110/worker.py' in cmd.read_bytes().replace(b'\0',b' ').decode():
            os.kill(pid,signal.SIGCONT)
    state.update(resumed=True,resume_reason='tradeoff_launch_failed',resumed_at=time.time());write(path,state)


def main():
    status('waiting_for_training_image_calibration')
    deadline=time.monotonic()+600
    while not CAL.exists():
        assert time.monotonic()<deadline;time.sleep(2)
    cal=read(CAL);assert cal['passed'];temperature=cal['selected_temperature']
    record=read(OUT/'noise-calibration-restart.json');old_pid=record['previous_status']['supervisor_pid']
    while Path(f'/proc/{old_pid}/cmdline').exists() and Path(f'/proc/{old_pid}/cmdline').read_bytes():time.sleep(1)
    restore_pause_record()
    receipt=read(OUT/'checkpoint.json');source=Path(receipt['path']);assert source.stat().st_size==receipt['bytes']
    with source.open('rb') as stream:source_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    assert receipt['global_step']<37560
    plan=read(HERE/'plan.json');plan['teacher']['final']=temperature
    plan['original_requested_temperature']=.1
    plan['name']=f'ImageNet | RQ24576 rewind60 | noise matched T{temperature} | 8H100'
    plan['noise_match_calibration']=cal
    plan.update(validation_source_checkpoint=str(source),validation_start_step=receipt['global_step'],
        validation_source_checkpoint_sha256=source_sha)
    write(HERE/'plan.json',plan);write(OUT/'plan.json',plan);write(BASE/'inputs/late-refinement.json',plan)
    old_recipe=BASE/'inputs/pre-calibration-recipe.yaml'
    if not old_recipe.exists():shutil.copyfile(BASE/'inputs/recipe.yaml',old_recipe)
    recipe=OmegaConf.load(BASE/'inputs/recipe.yaml');recipe.name=plan['name']
    recipe.targets.temperature=temperature;recipe.targets.temperature_schedule=plan['teacher']
    OmegaConf.save(recipe,BASE/'inputs/recipe.yaml')
    execution=read(BASE/'inputs/execution.json');execution['temperature_schedule']=plan['teacher'];write(BASE/'inputs/execution.json',execution)
    shutil.copyfile(CAL,BASE/'inputs/noise-match-training-calibration.json')
    shutil.copyfile(HERE/'native_recipe.py',BASE/'runtime/native_recipe.py')
    training=BASE/'source/src/training/rq_imagenet_shared.py';s=training.read_text()
    old="        run.config.update(rng_metadata,allow_val_change=True)"
    assert old in s
    s=s.replace(old,old+"\n        run.name=PROFILE.name\n        run.config.update({**PROFILE.metadata(),**PROFILE.checkpoint_extra()},allow_val_change=True)",1)
    training.write_text(s)
    check=HERE/'check_refinement.py';s=check.read_text().replace('teacher_temperature(37560,p)==.1',"teacher_temperature(37560,p)==p['final']")
    s=s.replace('teacher_temperature(62600,p)==.1',"teacher_temperature(62600,p)==p['final']");check.write_text(s)
    for file in [training,BASE/'runtime/native_recipe.py',HERE/'validate.py',HERE/'supervise.py',check]:py_compile.compile(str(file),doraise=True)
    subprocess.run([sys.executable,'-u',str(check)],check=True)
    VAL.mkdir(exist_ok=True)
    env=dict(os.environ,LASER_COMPARISON_TEMPERATURE=str(temperature),LASER_COMPARISON_MEMORY_GIB='4.5',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    with (VAL/'coordinator.log').open('ab',buffering=0) as log:
        comparison=subprocess.Popen([sys.executable,'-u',str(HERE/'launch_comparison.py'),'--validation'],
            env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    status('validating_calibrated_resume',temperature=temperature,checkpoint_step=receipt['global_step'],comparison_pid=comparison.pid)
    subprocess.run([sys.executable,'-u',str(HERE/'validate.py')],check=True)
    status('waiting_for_calibrated_full50k_comparison',temperature=temperature)
    assert comparison.wait()==0,'Full50k validation of selected temperature failed'
    validation=read(VAL/'report.json')
    assert validation['passed'] and all(validation['matched_teacher_comparison'].values())
    reference=validation['results']['original_rq_teacher050']['sampled_to_greedy_error_ratio']
    actual=validation['results']['laser_rq24576_teacher_matched']['sampled_to_greedy_error_ratio']
    assert abs(actual-reference)<.0005,'Relative noise agreement did not transfer to validation images'
    shutil.copyfile(VAL/'report.json',BASE/'inputs/noise-matched-full50k-validation.json')
    archive=OUT/'reproducibility';archive.mkdir(exist_ok=True)
    for path in HERE.glob('*.py'):shutil.copyfile(path,archive/path.name)
    for path in (BASE/'inputs').glob('*.yaml'):shutil.copyfile(path,archive/path.name)
    for name in ['late-refinement.json','noise-match-training-calibration.json','noise-matched-full50k-validation.json']:
        shutil.copyfile(BASE/'inputs'/name,archive/name)
    with (OUT/'supervisor.log').open('ab',buffering=0) as log:
        child=subprocess.Popen([sys.executable,'-u',str(HERE/'supervise.py')],stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    status('production_restarted',temperature=temperature,supervisor_pid=child.pid,resume_step=receipt['global_step'])
    print(json.dumps(dict(temperature=temperature,supervisor_pid=child.pid,resume_step=receipt['global_step'])),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        resume_search_on_failure();status('failed',error_type=type(error).__name__,error=str(error));raise
