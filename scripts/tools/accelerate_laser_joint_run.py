#!/usr/bin/env python3
"""Preserve a completed epoch and resume the same run with aligned GEMMs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))
from runtime.rq_imagenet_entry import freeze_shared_runtime
from src.training.rq_imagenet_runtime import write_json, sha256

RUN = 'imagenet-laser-native-joint-scratch6-4h100-20261015'
ORIGINAL = Path('/tmp/laser-rq-native-joint-imagenet-20261015')
BASE = Path('/tmp/laser-rq-native-joint-aligned-imagenet-20261015')
OUT = ROOT/'outputs'/RUN
PERF = OUT/'performance'


def prepare():
    if (PERF/'aligned-prepared.json').exists():raise RuntimeError('Already prepared')
    (BASE/'inputs').mkdir(parents=True,exist_ok=True)
    freeze_shared_runtime(BASE,ROOT,include_stochastic_teacher=False)
    shutil.copytree(ORIGINAL/'upstream',BASE/'upstream',ignore=shutil.ignore_patterns('__pycache__','.git','*.pyc'))
    for p in (ORIGINAL/'inputs').iterdir():
        if p.name.endswith('.pt'):os.link(p,BASE/'inputs'/p.name)
        else:shutil.copyfile(p,BASE/'inputs'/p.name)
    write_json(BASE/'inputs/execution.json',dict(align_classifier=True,alignment_multiple=64,
        valid_classes=65537,padded_projection_width=65600,extra_softmax_classes=0))
    for name in ['data','torch-cache']:(BASE/name).symlink_to(ORIGINAL/name,target_is_directory=True)
    shutil.copyfile(ORIGINAL/'train.py',BASE/'train.py')
    shutil.copyfile(ROOT/'scripts/tools/verify_laser_joint_alignment_resume.py',BASE/'verify.py')
    environment=json.loads((ORIGINAL/'environment.json').read_text())
    environment.update(LASER_RQ_BASE=str(BASE),WANDB_DIR=str(BASE/'wandb'))
    write_json(BASE/'environment.json',environment)
    files=[BASE/'train.py',BASE/'verify.py',BASE/'environment.json']
    for directory in ['source','runtime','upstream']:files.extend((BASE/directory).rglob('*.py'))
    files.extend((BASE/'inputs').glob('*.yaml'));files.extend((BASE/'inputs').glob('*.json'))
    manifest={str(p.relative_to(BASE)):sha256(p) for p in files}
    write_json(PERF/'aligned-code-manifest.json',manifest)
    with zipfile.ZipFile(PERF/'aligned-source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in files:archive.write(p,p.relative_to(BASE))
        archive.write(Path(__file__),'scripts/tools/'+Path(__file__).name)
        archive.write(ROOT/'tests/test_laser_joint_execution.py','tests/test_laser_joint_execution.py')
    write_json(PERF/'aligned-prepared.json',dict(prepared=True,run_id=RUN,base=str(BASE),
        same_parameters=True,same_objective=True,valid_vocabulary=65537,padded_width=65600,
        trigger='first durable epoch checkpoint',restart_from_scratch=False,time=time.time()))


def run():
    lock=(BASE/'handoff.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((PERF/'aligned-prepared.json').read_text())['prepared']
    assert json.loads((PERF/'alignment-cpu-verification.json').read_text())['passed']
    assert json.loads((PERF/'aligned-linear-autocast.json').read_text())['maximum_fp16_kl']<1.e-5
    for name,expected in json.loads((PERF/'aligned-code-manifest.json').read_text()).items():
        assert sha256(BASE/name)==expected,name
    cancelled=False
    child=None
    def stop(_signal,_frame):
        nonlocal cancelled
        cancelled=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while True:
        if cancelled:return 130
        path=OUT/'checkpoint.json'
        metadata=json.loads(path.read_text()) if path.exists() else {}
        if metadata.get('global_step',0)>=626:break
        write_json(PERF/'alignment-status.json',dict(phase='waiting_for_durable_checkpoint',time=time.time(),
            prior_checkpoint_step=metadata.get('global_step'),controller_pid=os.getpid()))
        time.sleep(10)
    checkpoint=(OUT/'checkpoints/last.pt').resolve()
    assert checkpoint.stat().st_size==metadata['bytes']
    alias=checkpoint.parent.parent/'before-classifier-alignment.pt'
    if not alias.exists():alias.symlink_to(os.path.relpath(checkpoint,alias.parent))
    status=json.loads((OUT/'launch-status.json').read_text())
    old_pid=int(status['pid'])
    command=Path(f'/proc/{old_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert 'launch_laser_native_joint_imagenet.py' in command
    progress=json.loads((OUT/'train/progress-rank0.json').read_text())
    write_json(PERF/'handoff-checkpoint.json',dict(checkpoint=metadata,protected_alias=str(alias),
        previous_progress=progress,previous_controller=old_pid,time=time.time()))
    os.kill(old_pid,signal.SIGTERM)
    for _ in range(120):
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        if not active.strip():break
        if cancelled:return 130
        time.sleep(1)
    else:raise RuntimeError('GPUs did not release after checkpointed handoff')
    environment=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    def launch(arguments,phase):
        nonlocal child
        with (PERF/f'{phase}.log').open('a') as log:
            child=subprocess.Popen(arguments,cwd=BASE,env=environment,stdout=log,stderr=subprocess.STDOUT)
            status=dict(phase=phase,controller_pid=os.getpid(),pid=os.getpid(),child_pid=child.pid,
                run_id=RUN,resuming=True,world_size=4,microbatch=64,accumulation_steps=8,
                global_batch=2048,checkpoint_step=metadata['global_step'],command=arguments,
                execution='aligned_classifier',time=time.time())
            write_json(PERF/'alignment-status.json',status)
            write_json(OUT/'launch-status.json',status)
            code=child.wait()
        if code:raise RuntimeError(f'{phase} exited {code}')
    launch([sys.executable,str(BASE/'verify.py')],'aligned_verification')
    assert json.loads((PERF/'aligned-resume-verification.json').read_text())['passed']
    if cancelled:return 130
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=4',
        str(BASE/'train.py'),'--epochs','6','--microbatch','64','--resume',str(OUT/'checkpoints/last.pt')]
    launch(command,'aligned_training')
    status=dict(phase='completed',pid=os.getpid(),run_id=RUN,resuming=True,execution='aligned_classifier',time=time.time())
    write_json(PERF/'alignment-status.json',status);write_json(OUT/'launch-status.json',status)
    reference=ROOT/'outputs/imagenet-original-rq1400m-rfid473-scratch20-4h100-20261014/evaluations'
    curves={}
    for epoch in [1,2,4,6]:
        filename=f'epoch{epoch:03d}.json'
        joint=json.loads((OUT/'evaluations'/filename).read_text());rq=json.loads((reference/filename).read_text())
        assert joint['generated_images']==rq['generated_images']==50000
        assert joint['real_images']==rq['real_images']==1281167
        curves[str(epoch)]=dict(joint_fid=joint['eval/fid'],rq_fid=rq['eval/fid'],joint_minus_rq=joint['eval/fid']-rq['eval/fid'])
    recipe=json.loads((OUT/'prepared.json').read_text())['recipe']
    write_json(OUT/'comparison.json',dict(completed_epochs=6,global_step=3756,curves=curves,
        joint_sampler=recipe['sampling'],rq_sampler=dict(temperature=1.,top_k=256,top_p=.95),
        continued_beyond_six_epochs=False,classifier_alignment=True))
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:raise SystemExit(run())
