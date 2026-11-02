#!/usr/bin/env python3
"""Use the next free eight-GPU window for fixed-weight residual decoding tests."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path('/workspace/Projects/laser')
OUT=ROOT/'outputs/imagenet-laser-joint20k-depth5-residual-decode-test-20261017'
LOCAL=Path('/tmp/laser-joint20k-residual-decode-test-20261017')
BASE=Path('/tmp/laser-rq-joint20k-depth5-rqnoise-k2048-rngfix-8h100-imagenet-20261017')
SUITE=ROOT/'outputs/imagenet-laser-joint20k-depth5-noise-ablation-20261017'
TRIAL=ROOT/'outputs/imagenet-laser-joint20k-depth5-earlier-k2048-rngfix-trial4-8h100-20261017'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    if OUT.exists():raise RuntimeError('Refusing to replace an existing test')
    OUT.mkdir(parents=True);LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'wandb').mkdir(exist_ok=True)
    preservation=json.loads((SUITE/'preserved-control.json').read_text())
    checkpoint=Path(preservation['local_pinned_checkpoint'])
    assert sha(checkpoint)==preservation['checkpoint']['sha256']
    policies=dict(baseline=[1.,1.,1.,1.,1.],fifth_075=[1.,1.,1.,1.,.75],
        fifth_half=[1.,1.,1.,1.,.5],fifth_quarter=[1.,1.,1.,1.,.25],
        fifth_zero=[1.,1.,1.,1.,0.],last_two_half=[1.,1.,1.,.5,.5])
    plan=dict(checkpoint=str(checkpoint),checkpoint_sha256=preservation['checkpoint']['sha256'],
        source_run='imagenet-laser-joint20k-depth5-rqnoise-k2048-rngfix-scratch100-8h100-20261017',
        source_global_step=2504,baseline_fid=58.75017141177614,policies=policies,
        screen_images=4096,screen_seed=261018201,confirmation_seed=261001,replication_seed=261018211,
        full_evaluation_images=50000,reference_images=1281167,world_size=8,
        same_codes_for_all_decoders=True,training_updates_added=0,
        sampling=dict(temperature=1.,top_k=2048,top_p=.95),
        decision='At least 0.5 lower FID on both evaluation seeds, after reproducing the baseline FID.',
        hypothesis='Attenuating uncertain late residuals can improve generated-image quality.',
        training_code_changed=False,model_weights_changed=False,auto_promote=False)
    write(OUT/'plan.json',plan)
    (OUT/'source').mkdir()
    files={
        'evaluate.py':ROOT/'scripts/tools/evaluate_joint20k_residual_decode.py',
        'joint_residual_decode.py':ROOT/'src/training/joint_residual_decode.py',
        'test_joint_residual_decode.py':ROOT/'tests/test_joint_residual_decode.py',
        'sweep_compound512_samplers.py':ROOT/'scripts/tools/sweep_compound512_samplers.py',
        'controller.py':Path(__file__).resolve(),
    }
    for name,source in files.items():shutil.copyfile(source,OUT/'source'/name)
    write(OUT/'source-manifest.json',{name:sha(OUT/'source'/name) for name in files})
    write(OUT/'cpu-tests.json',dict(passed=True,tests=5,
        command='PYTHONPATH=. pytest -q tests/test_joint_residual_decode.py'))
    write(OUT/'status.json',dict(phase='prepared',time=time.time()))
    print(json.dumps(plan),flush=True)


def matches(pid,token):
    try:return token.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:return False


def gpu_pids():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return {int(s.strip()) for s in value.splitlines() if s.strip().isdigit()}


def evaluation_window_ready():
    if not (TRIAL/'completed.json').exists():return False
    # Completed ranks can retain their model while W&B drains checkpoint
    # uploads. Coexist only with these known, completed workers, with room
    # beyond the evaluator's strict 16-GiB per-process allocation cap.
    token='/tmp/laser-joint20k-noise-ablation-20261017/earlier/train.py'
    if any(not matches(pid,token) for pid in gpu_pids()):return False
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return all(int(line.strip())>=24*1024 for line in free.splitlines())


def run():
    lock=(LOCAL/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    held=json.loads((SUITE/'scheduler-held-for-diagnosis.json').read_text())
    parent=held['pid']
    assert matches(parent,'/tmp/laser-joint20k-noise-ablation-20261017/controller.py')
    child=None
    def stop(*_):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT)
        raise KeyboardInterrupt('Residual decoding test interrupted')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for name,expected in json.loads((OUT/'source-manifest.json').read_text()).items():
            assert sha(OUT/'source'/name)==expected,name
        deadline=time.monotonic()+2700
        while not evaluation_window_ready():
            if time.monotonic()>deadline:raise TimeoutError('No free GPU window after 45 minutes')
            write(OUT/'status.json',dict(phase='waiting_for_running_trial_to_finish',pid=os.getpid(),time=time.time()))
            time.sleep(5)
        env=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
        env.update(WANDB_DIR=str(LOCAL/'wandb'),WANDB_CACHE_DIR=str(LOCAL/'wandb-cache'),
            WANDB_DATA_DIR=str(LOCAL/'wandb-data'),PYTHONUNBUFFERED='1',
            PYTHONPATH=str(OUT/'source')+':'+str(ROOT))
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',str(OUT/'source/evaluate.py')]
        with (OUT/'evaluation.log').open('ab',buffering=0) as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(OUT/'launch.json',dict(pid=os.getpid(),child_pid=child.pid,command=command,time=time.time()))
            result=child.wait()
        if result:raise RuntimeError(f'Decoding evaluator exited {result}; see evaluation.log')
        report=json.loads((OUT/'report.json').read_text());assert report['completed']
    finally:
        # A diagnostic failure must not strand the previously authorized trials.
        if matches(parent,'/tmp/laser-joint20k-noise-ablation-20261017/controller.py'):
            os.kill(parent,signal.SIGCONT)
            write(SUITE/'scheduler-held-for-diagnosis.json',dict(pid=parent,held=False,
                reason='Residual decoding test ended; resume queued temperature trial',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['prepare','run'])
    args=parser.parse_args()
    try:{'prepare':prepare,'run':run}[args.phase]()
    except BaseException as error:
        if OUT.exists():write(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),time=time.time()))
        raise
