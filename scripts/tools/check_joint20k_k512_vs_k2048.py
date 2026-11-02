#!/usr/bin/env python3
"""Run the sampler check between training sessions, then resume the same run."""
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

ROOT = Path('/workspace/Projects/laser')
RUN = 'imagenet-laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018'
BASE = Path('/tmp/laser-joint20k-depth5-uniformce-k512-scratch100-4b200-20261018')
TRAIN = ROOT / 'outputs' / RUN
OUT = TRAIN / 'diagnostics/sampler-k512-vs-k2048-epoch1-20261018'
LOCAL = Path('/tmp/laser-k512-vs-k2048-epoch1-20261018')
EVALUATOR = ROOT / 'scripts/tools/evaluate_joint20k_k512_vs_k2048.py'
CHILD = None
STOPPING = False


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def receipts():
    return [json.loads(path.read_text()) for path in
        (TRAIN / 'persistent-train/checkpoints/.checkpoint-data').glob('*.json')]


def status(phase, **fields):
    write(OUT / 'status.json', dict(phase=phase, supervisor_pid=os.getpid(),
        run_id=RUN, world_size=4, time=time.time(), **fields))


def pin(receipt, destination):
    source = Path(receipt['local'])
    assert source.stat().st_size == receipt['bytes']
    assert Path(receipt['persistent']).stat().st_size == receipt['bytes']
    assert digest(source) == receipt['sha256']
    if not destination.exists():
        os.link(source, destination)
    assert destination.stat().st_size == receipt['bytes']


def stop(signum, _frame):
    global STOPPING
    STOPPING = True
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGINT)
    raise SystemExit(128 + signum)


def environment():
    return dict(os.environ, **json.loads((BASE / 'environment.json').read_text()),
        WANDB_API_KEY=Path('/root/.config/laser/wandb_api_key').read_text().strip())


def child(command, log, phase, *, training=False, sampling_top_k=512):
    global CHILD
    with log.open('ab', buffering=0) as stream:
        CHILD = subprocess.Popen(command, cwd=BASE, env=environment(),
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True)
        while True:
            status(phase, child_pid=CHILD.pid, log=str(log))
            if training:
                path = BASE / 'live-training/train/progress-rank0.json'
                progress = json.loads(path.read_text()) if path.exists() else {}
                write(TRAIN / 'status.json', dict(phase='training', run_id=RUN,
                    supervisor_pid=os.getpid(), child_pid=CHILD.pid, epochs=100,
                    world_size=4, hardware='4 NVIDIA B200', microbatch=256,
                    accumulation_steps=2, global_batch=2048,
                    objective='native_soft_target_cross_entropy',
                    sampling=dict(temperature=1., top_k=sampling_top_k, top_p=.95),
                    command=command, latest_progress=progress, time=time.time()))
            try:
                code = CHILD.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                pass
    if code:
        raise RuntimeError(f'{phase} exited {code}; inspect {log}')


def stop_previous(previous, checkpoint):
    controller = previous['supervisor_pid']
    controller_command = Path(f'/proc/{controller}/cmdline').read_bytes()
    assert any(name in controller_command for name in (
        b'launch_joint20k_uniformce_k512_scratch.py', b'check_joint20k_k512_vs_k2048.py'))
    progress = [json.loads((TRAIN / f'train/progress-rank{rank}.json').read_text()) for rank in range(4)]
    for row in progress:
        assert str(BASE / 'train.py').encode() in Path(f'/proc/{row["pid"]}/cmdline').read_bytes()
    write(OUT / 'training-handoff.json', dict(previous_supervisor_pid=controller,
        preserved_checkpoint=checkpoint, latest_observed_step=progress[0]['train/global_step'],
        unsaved_updates_to_replay=progress[0]['train/global_step'] - checkpoint['global_step'],
        original_training_recipe_sha256=digest(BASE / 'inputs/recipe.yaml'), time=time.time()))
    os.kill(controller, signal.SIGTERM)
    for _ in range(45):
        occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], text=True).strip()
        if not occupied:
            break
        time.sleep(1)
    assert not occupied, f'GPU processes remain: {occupied}'
    for _ in range(10):
        path = Path(f'/proc/{controller}/cmdline')
        if not path.exists() or not path.read_bytes():
            break
        time.sleep(1)
    write(TRAIN / 'status.json', dict(phase='evaluating_saved_epoch1_samplers', run_id=RUN,
        supervisor_pid=os.getpid(), preserved_training_step=checkpoint['global_step'],
        original_training_sampler=512, automatic_training_resume=True, world_size=4,
        diagnostics=str(OUT), time=time.time()))


def publish():
    import wandb
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb_api_key').read_text().strip()
    run = wandb.Api(timeout=60).run('helloimlixin-rutgers/laser/' + RUN)
    for path in OUT.glob('*.json'):
        run.upload_file(str(path), root=str(TRAIN))
    for top_k in (512, 2048):
        path = OUT / f'k{top_k}/smoke/samples/step000626-fixed.png'
        run.upload_file(str(path), root=str(TRAIN))
    report = json.loads((OUT / 'report.json').read_text())
    run.summary.update({'sampler_check/epoch1/k512_fid50k':report['results']['512']['fid'],
        'sampler_check/epoch1/k2048_fid50k':report['results']['2048']['fid'],
        'sampler_check/epoch1/fid2048_minus_fid512':report['fid2048_minus_fid512'],
        'sampler_check/epoch1/training_recipe_modified':False})
    write(OUT / 'wandb-publication.json', dict(uploaded=True, run_url=run.url, time=time.time()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    lock = (LOCAL / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'plan.json').exists():
        raise RuntimeError('Sampler check already prepared')
    initial = json.loads((TRAIN / 'status.json').read_text())
    assert initial['phase'] == 'training' and initial['run_id'] == RUN
    source = next(row for row in receipts() if row['global_step'] == 626)
    pin(source, LOCAL / 'epoch001.pt')
    reference = json.loads((TRAIN / 'evaluations/epoch001.json').read_text())
    assert source['fid'] == reference['eval/fid']
    current_step = initial['latest_progress']['train/global_step']
    next_step = ((current_step + 625) // 626) * 626
    plan = dict(run_id=RUN, base=str(BASE), training_output=str(TRAIN), output=str(OUT),
        checkpoint=str(LOCAL / 'epoch001.pt'), checkpoint_sha256=source['sha256'],
        checkpoint_receipt=source, original_fid512=reference['eval/fid'],
        world_size=4, batch_size=128, seed=261001,
        generated_images_per_policy=50000, reference_images=1281167, reference_split='train',
        reference_sha256=digest(BASE / 'inputs/imagenet_256_train.npz'),
        sampling_policies=[dict(temperature=1., top_k=k, top_p=.95) for k in (512, 2048)],
        resume_after_check=True, handoff_minimum_checkpoint_step=next_step,
        evaluator_sha256=digest(EVALUATOR), original_training_sampler=512,
        original_recipe_sha256=digest(BASE / 'inputs/recipe.yaml'), time=time.time())
    write(OUT / 'plan.json', plan)
    shutil.copyfile(EVALUATOR, OUT / EVALUATOR.name)
    shutil.copyfile(__file__, OUT / Path(__file__).name)
    while True:
        recent = max(receipts(), key=lambda row:row['global_step'])
        if recent['global_step'] >= next_step:
            break
        progress = json.loads((TRAIN / 'train/progress-rank0.json').read_text())
        status('waiting_for_training_checkpoint', current_step=progress['train/global_step'],
            current_saved_step=recent['global_step'], target_saved_step=next_step)
        time.sleep(10)
    resume = LOCAL / 'resume-training.pt'
    pin(recent, resume)
    stop_previous(initial, recent)
    error = None
    distributed = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4']
    try:
        child([*distributed, str(EVALUATOR), '--plan', str(OUT / 'plan.json')],
            OUT / 'evaluation.log', 'evaluating')
        assert digest(BASE / 'inputs/recipe.yaml') == plan['original_recipe_sha256']
        assert digest(LOCAL / 'epoch001.pt') == plan['checkpoint_sha256']
        try:
            publish()
        except Exception as publication_error:
            write(OUT / 'wandb-publication.json', dict(uploaded=False, error=str(publication_error)))
    except Exception as evaluation_error:
        error = str(evaluation_error)
        write(OUT / 'evaluation-error.json', dict(error=error, time=time.time()))
    if not STOPPING:
        assert digest(resume) == recent['sha256']
        write(OUT / 'resume-plan.json', dict(checkpoint=str(resume), receipt=recent,
            original_recipe_sha256_unchanged=True, same_optimizer_scheduler_scaler_and_rng=True,
            training_sampler=512, epochs=100, evaluation_error=error, time=time.time()))
        child([*distributed, str(BASE / 'train.py'), '--resume', str(resume),
            '--epochs', '100', '--microbatch', '256'], TRAIN / 'training.log',
            'training_resumed', training=True)
        status('completed')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        main()
    except BaseException as error:
        status('stopped' if STOPPING else 'failed', error_type=type(error).__name__, error=str(error))
        raise
