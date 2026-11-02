#!/usr/bin/env python3
"""Freeze the minimal LASER recipe, then run after the original RQ control."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT=Path('/workspace/Projects/laser')
sys.path.insert(0,str(ROOT))
from runtime.rq_imagenet_entry import freeze_shared_runtime
BASE=Path('/tmp/laser-rq-minimal-imagenet-20261014')
CONTROL_BASE=Path('/tmp/laser-original-rq-imagenet-20261014')
CONTROL=ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014'
RUN='imagenet-laser-rq-minimal-rfid421-scratch20-20261014'
OUT=ROOT/'outputs'/RUN


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare():
    import yaml
    if (OUT/'prepared.json').exists():raise RuntimeError('Already prepared; use run')
    recipe_path=ROOT/'configs/stage2/imagenet-laser-rq-minimal-20ep-20261014.yaml'
    recipe=yaml.safe_load(recipe_path.read_text())
    assert recipe['run_id']==RUN and recipe['auxiliary_loss_terms']==0
    assert json.loads((CONTROL_BASE/'data/ready.json').read_text())['ready']
    BASE.mkdir(parents=True,exist_ok=True);(BASE/'inputs').mkdir(exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc')
    freeze_shared_runtime(BASE,ROOT)
    shutil.copytree(CONTROL_BASE/'upstream',BASE/'upstream',ignore=ignore)
    shutil.copyfile(ROOT/'scripts/tools/train_laser_rq_minimal_imagenet.py',BASE/'train.py')
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_rq_minimal_checkpoint.py',BASE/'verify.py')
    shutil.copyfile(recipe_path,BASE/'inputs/recipe.yaml')
    shutil.copyfile(ROOT/'configs/rq-upstream-core.json',BASE/'inputs/rq-upstream-core.json')
    tokenizer=Path('/tmp/laser-imagenet-duplicate-20261014/inputs/stage1-tokenizer.pt')
    assert digest(tokenizer)==recipe['tokenizer']['sha256']
    os.link(tokenizer,BASE/'inputs/stage1-model.pt')
    for name in ('stage2-training-config.yaml','stage2-archive-config.yaml','imagenet_256_train.npz'):
        shutil.copyfile(CONTROL_BASE/'inputs'/name,BASE/'inputs'/name)
    assert digest(BASE/'inputs/stage2-archive-config.yaml')=='6065c30c936ad534b329f09a652570ea689045c4f8019edc50087f2d81d46d87'
    assert digest(BASE/'inputs/imagenet_256_train.npz')==recipe['evaluation']['reference_sha256']
    for name in ('data','torch-cache'):(BASE/name).symlink_to(CONTROL_BASE/name,target_is_directory=True)
    environment=dict(LASER_RQ_BASE=str(BASE),LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        TORCH_HOME=str(BASE/'torch-cache'),WANDB_DIR=str(BASE/'wandb'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    write(BASE/'environment.json',environment)
    reproducibility=OUT/'reproducibility';reproducibility.mkdir(exist_ok=True)
    for path in (BASE/'inputs').glob('*'):
        shutil.copyfile(path,reproducibility/path.name)
    shutil.copyfile(ROOT/'outputs/stochastic-image-pairs-20261007/calibration.json',
                    reproducibility/'teacher-calibration.json')
    shutil.copyfile(CONTROL_BASE/'data/ready.json',reproducibility/'data-verification.json')
    files=[BASE/'train.py',BASE/'verify.py',BASE/'environment.json']
    files+=list((BASE/'source').rglob('*.py'))+list((BASE/'runtime').glob('*.py'))
    files+=list((BASE/'upstream').rglob('*.py'))+list((BASE/'inputs').glob('*.yaml'))
    files.append(BASE/'inputs/rq-upstream-core.json')
    manifest={str(path.relative_to(BASE)):digest(path) for path in files}
    write(reproducibility/'code-manifest.json',manifest)
    with zipfile.ZipFile(reproducibility/'source.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for path in files:bundle.write(path,path.relative_to(BASE))
        for name in ('launch_laser_rq_minimal_imagenet.py',):
            bundle.write(ROOT/'scripts/tools'/name,'scripts/tools/'+name)
        for name in ('test_laser_rq_minimal.py','test_physical_pair_scalar_prior.py',
                     'test_stochastic_image_pairs.py','test_omp_joint_targets.py'):
            bundle.write(ROOT/'tests'/name,'tests/'+name)
    plan=dict(prepared=True,run_id=RUN,recipe=recipe,waiting_for_control=str(CONTROL),
        fresh_stage2=True,fresh_optimizer=True,train_images=1281167,updates_per_epoch=626,
        stop_updates=12520,cosine_total_updates=62600,
        gpu_preflight_required=True,source_manifest_sha256=digest(reproducibility/'code-manifest.json'),
        prepared_at=time.time())
    write(OUT/'prepared.json',plan)
    print(json.dumps(plan),flush=True)


def run(start_now=False,reuse_verified_preflight=False,resume=False):
    prepared=json.loads((OUT/'prepared.json').read_text());assert prepared['prepared']
    assert json.loads((OUT/'cpu-verification.json').read_text())['passed']
    assert json.loads((OUT/'shared-codebase-verification.json').read_text())['passed']
    calibration=json.loads((OUT/'teacher-calibration-fp32-train.json').read_text())
    assert calibration['passed'] and calibration['encoder_dtype']=='torch.float32'
    assert calibration['linalg_backend']=='cusolver'
    assert calibration['selected_temperature']==prepared['recipe']['teacher']['atom_temperature']
    assert calibration['checkpoint_sha256']==prepared['recipe']['tokenizer']['sha256']
    lock=(BASE/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(phase='waiting_for_free_gpus' if start_now else 'queued_after_control',
        pid=os.getpid(),run_id=RUN,control=str(CONTROL),queued_at=time.time(),
        auxiliary_loss_terms=0,start_now=start_now,resuming=resume)
    write(OUT/'launch-status.json',status)
    child=None
    cancelled=False
    def stop(_signum,_frame):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if start_now:
        control_status=json.loads((CONTROL/'launch-status.json').read_text())
        if control_status['phase']=='paused':
            pause=json.loads((CONTROL/'paused.json').read_text())
            assert pause['paused'] and Path(pause['checkpoint_payload']).is_file()
            status['control_resume_checkpoint']=pause['checkpoint_payload']
        elif control_status['phase']!='completed':
            raise RuntimeError('--start-now requires the original control to be checkpointed and paused')
    while not start_now and not cancelled:
        complete=CONTROL/'completed.json'
        control_status=json.loads((CONTROL/'launch-status.json').read_text())
        if control_status['phase']=='stopped_or_failed':
            status.update(phase='waiting_for_control_recovery',checked_at=time.time())
            write(OUT/'launch-status.json',status)
            time.sleep(30)
            continue
        if complete.exists() and control_status['phase']=='completed':
            result=json.loads(complete.read_text())
            assert result['epochs']==20 and result['global_step']==12520
            break
        status.update(phase='queued_after_control',checked_at=time.time())
        write(OUT/'launch-status.json',status)
        time.sleep(30)
    if cancelled:
        status.update(phase='cancelled_before_start',finished_at=time.time())
        write(OUT/'launch-status.json',status);return 130
    # The queue never kills a GPU process or launches alongside another job.
    for _ in range(120):
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if not active:break
        if cancelled:return 130
        time.sleep(30)
    else:raise RuntimeError('GPUs remained occupied after control completion')
    manifest=json.loads((OUT/'reproducibility/code-manifest.json').read_text())
    for name,expected in manifest.items():assert digest(BASE/name)==expected,name
    assert digest(BASE/'inputs/stage1-model.pt')==prepared['recipe']['tokenizer']['sha256']
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
    sizes=[int(value) for value in memory.splitlines()]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    world=len(sizes) if visible is None else len([value for value in visible.split(',') if value.strip()])
    if world<1:raise RuntimeError('No GPUs available')
    microbatch=64 if min(sizes)>=75000 else 32 if min(sizes)>=45000 else 16
    env=os.environ.copy();env.update(json.loads((BASE/'environment.json').read_text()))
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',
             str(BASE/'train.py'),'--epochs','20','--microbatch',str(microbatch)]
    if resume:
        checkpoint=OUT/'checkpoints/last.pt'
        if not checkpoint.is_file():checkpoint=OUT/'train/checkpoints/last.pt'
        assert checkpoint.is_file()
        assert json.loads((OUT/'performance/resume-checkpoint-audit.json').read_text())['passed']
        assert json.loads((OUT/'performance/speed-fix-verification.json').read_text())['passed']
        command+=['--resume',str(checkpoint)]
        status['resume_checkpoint']=str(checkpoint.resolve())
    status.update(world_size=world,microbatch=microbatch,accumulation_steps=math.ceil(2048/(world*microbatch)),
                  global_batch=2048)
    def launch(arguments,phase):
        nonlocal child
        with (OUT/f'{phase}.log').open('a') as log:
            child=subprocess.Popen(arguments,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT)
            status.update(phase=phase,child_pid=child.pid,started_at=time.time())
            write(OUT/'launch-status.json',status)
            result=child.wait()
        if result:
            status.update(phase='stopped_or_failed',returncode=result,finished_at=time.time())
            write(OUT/'launch-status.json',status)
            raise RuntimeError(f'{phase} returned {result}; inspect {OUT/phase}.log')
    if resume:
        status['verified_teacher_optimization_resume']=True
    elif reuse_verified_preflight:
        assert json.loads((OUT/'smoke/verification.json').read_text())['passed']
        assert json.loads((OUT/'smoke-reload-verification.json').read_text())['passed']
        assert json.loads((OUT/'gpu-generation-verification.json').read_text())['passed']
        assert not list((OUT/'train/checkpoints').glob('*.pt'))
        assert not list((OUT/'train').glob('progress-rank*.json'))
        status['reused_verified_preflight']=True
    else:
        launch([*command,'--smoke-updates','2'],'gpu_preflight')
        launch([sys.executable,str(BASE/'verify.py')],'checkpoint_verification')
    assert json.loads((OUT/'smoke-reload-verification.json').read_text())['passed']
    if cancelled:return 130
    # Production is a fresh process and never loads the private preflight.
    launch(command,'train')
    status.update(phase='completed',returncode=0,finished_at=time.time())
    write(OUT/'launch-status.json',status)
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--start-now',action='store_true',
        help='Start after the original control has been checkpointed and paused')
    parser.add_argument('--reuse-verified-preflight',action='store_true',
        help='Reuse passed GPU checks after a logging-only failure before any production update')
    parser.add_argument('--resume',action='store_true',help='Resume the verified production checkpoint')
    arguments=parser.parse_args()
    if arguments.action=='prepare':prepare()
    else:sys.exit(run(start_now=arguments.start_now,reuse_verified_preflight=arguments.reuse_verified_preflight,
                      resume=arguments.resume))
