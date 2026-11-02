#!/usr/bin/env python3
"""Launch the validated ImageNet joint-code prior from step zero on 2 B300s.

Uses the earlier 86.28 epoch-one FID recipe: depth-specific coefficient tables,
calibrated targets, independent rank RNG, and the original 5e-4 LR schedule.
The bad fully shared-codebook run is preserved as a separate experiment.
"""
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
import zipfile

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-sharedjoint20k-repair-2b300-20261018')
RUN = 'imagenet-laser-joint20k-depth5-rqnoise-rngfix-scratch100-2b300-20261018'
OUT = ROOT/'outputs'/RUN
CHILD = None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+f'.{os.getpid()}.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(path)


def status(phase, **values):
    progress = BASE/'live-training/train/progress-rank0.json'
    latest = json.loads(progress.read_text()) if progress.exists() else {}
    write(OUT/'status.json', dict(phase=phase, run_id=RUN, supervisor_pid=os.getpid(),
        epochs=100, target_updates=62600, world_size=2, hardware='2 NVIDIA B300 SXM6',
        global_batch=2048, latest_progress=latest, time=time.time(), **values))


def stop(signum, _):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    status('interrupted', signal=signum)
    raise SystemExit(128+signum)


def child(command, env, phase):
    global CHILD
    with (OUT/(phase+'.log')).open('ab', buffering=0) as log:
        CHILD = subprocess.Popen(command, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            status(phase, child_pid=CHILD.pid, command=command)
            try:
                code = CHILD.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                pass
    return code


def archive():
    repro=OUT/'reproducibility';repro.mkdir(parents=True,exist_ok=True)
    for source in (OUT/'diagnosis.json',BASE/'data/train-ready.json'):
        if source.exists():shutil.copyfile(source,repro/source.name)
    for p in (BASE/'inputs').iterdir():
        if p.suffix in ('.yaml','.json') or p.name=='joint-codebook.pt':
            shutil.copyfile(p,repro/p.name)
    shutil.copyfile(__file__,repro/Path(__file__).name)
    files=[BASE/'train.py', BASE/'environment.json']
    for name in ('source','runtime','upstream'):
        files.extend(p for p in (BASE/name).rglob('*.py') if '__pycache__' not in p.parts)
    files.extend(p for p in (BASE/'inputs').iterdir() if p.suffix in ('.yaml','.json'))
    hashes={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write(repro/'code-manifest.json',hashes)
    with zipfile.ZipFile(repro/'source.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(BASE))
    requirements=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (repro/'requirements-freeze.txt').write_text(requirements)


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',type=Path,help='Resume this run after an interrupted launch')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(BASE/'repair-controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    env=dict(os.environ,**json.loads((BASE/'environment.json').read_text()))
    for key in ('WANDB_SERVICE','_WANDB_SERVICE'):env.pop(key,None)
    # Credentials are loaded only into memory; archived configuration is public.
    env['WANDB_API_KEY']=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
    ready=BASE/'data/train-ready.json'
    while not ready.exists():
        status('waiting_for_verified_training_images')
        time.sleep(10)
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',str(BASE/'train.py')]
    recipe_path=BASE/'inputs/recipe.yaml'
    recipe=yaml.safe_load(recipe_path.read_text())
    assert recipe['epochs']==recipe['lr_schedule_epochs']==100
    assert recipe['tokenizer']['codebook_sha256']=='922c33037770287facb9ba8f2c477224087155ff14af3b9d6d7390b5a40485a9'
    assert recipe['training_rng']['policy']=='independent_rank_v1'
    if args.resume:
        assert ready.exists()
        microbatch=recipe['microbatch']
        assert args.resume.is_file()
        archive()
        code=child([*command,'--epochs','100','--microbatch',str(microbatch),'--resume',str(args.resume)],env,'training')
    else:
        assert not (OUT/'train/progress-rank0.json').exists(), 'Fresh launch refuses existing training'
        microbatch=None
        for candidate in (512,256):
            smoke=BASE/f'preflight-mb{candidate}'
            assert not smoke.exists(), 'Preflight already exists; inspect it before relaunching'
            phase=f'preflight-mb{candidate}'
            code=child([*command,'--epochs','100','--microbatch',str(candidate),'--smoke-updates','3','--smoke-split','train'],
                dict(env,LASER_PRIVATE_SMOKE_DIR=str(smoke)),phase)
            if not code:
                microbatch=candidate
                break
            log=(OUT/(phase+'.log')).read_text(errors='replace')
            if 'out of memory' not in log.lower():
                raise RuntimeError(f'{phase} failed; inspect {OUT/(phase+".log")}')
        if microbatch is None:raise RuntimeError('Neither 512 nor 256 microbatch fits')
        proof=json.loads((smoke/'verification.json').read_text())
        assert proof['passed'] and proof['global_step']==proof['all_adam_steps']==3
        startup=[json.loads((smoke/f'startup-rank{r}.json').read_text()) for r in range(2)]
        rng=[json.loads((smoke/f'rng-startup-rank{r}.json').read_text()) for r in range(2)]
        assert len({r['dropout_probe_sha256'] for r in rng})==2
        for row in startup:
            assert row['fresh_stage2'] and row['fresh_optimizer'] and row['global_step']==0
            assert row['training_epochs']==row['lr_schedule_epochs']==100
            assert row['global_batch']==2048 and not row['codebook_shared_across_depths']
            assert row['initial_optimizer_lr']==.0005 and row['warmup_epochs']==0
        for r in range(2):
            assert json.loads((smoke/f'preview-verification-rank{r}.json').read_text())['passed']
        initial=json.loads((smoke/'initial-weight-verification.json').read_text())
        assert initial['passed'] and not initial['stage2_checkpoint_loaded']
        recipe['microbatch']=microbatch
        for path in (recipe_path,ROOT/'configs/stage2'/f'{RUN}.yaml'):
            path.write_text(yaml.safe_dump(recipe,sort_keys=False))
        write(OUT/'preflight.json',dict(passed=True,proof=proof,microbatch=microbatch,
            accumulation_steps=2048//(2*microbatch),rank_rng_independent=True,
            probe_weights_discarded=True,production_starts_from_zero=True,initialization=initial))
        archive()
        while not ready.exists():
            status('preflight_passed_waiting_for_verified_training_images',microbatch=microbatch)
            time.sleep(10)
        data=json.loads(ready.read_text())
        assert data['ready'] and data['training_images']==1281167
        code=child([*command,'--epochs','100','--microbatch',str(microbatch)],env,'training')
    if code:
        status('failed',exit_code=code)
        raise RuntimeError(f'Training exited {code}; inspect {OUT/"training.log"}')
    complete=json.loads((OUT/'completed.json').read_text())
    assert complete['epochs']==100 and complete['global_step']==62600
    status('completed',best_fid=complete['best_fid'])


if __name__=='__main__':
    main()
