"""Preserve the replaced run, fit and validate the tokenizer, then train100 epochs."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time
from omegaconf import OmegaConf
from environment import BASE, ROOT, SOURCE, HERE, OUT, QUALITY, RUN, PYTHON, environment
from persistent_copy import copy_file

CHILD = None
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temp.replace(path)


def status(phase, **kw):
    p = BASE / 'live-training/train/progress-rank0.json'
    write(OUT / 'status.json', dict(phase=phase, run_id=RUN, supervisor_pid=os.getpid(),
                                  child_pid=CHILD.pid if CHILD else None, world_size=8,
                                  epochs=100, global_batch=2048, final_codewords=24576,
                                  beam_search=False, mixture_teacher=False,
                                  progress=read(p) if p.exists() else None, time=time.time(), **kw))


def alive(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[0] != 'Z'
    except FileNotFoundError:
        return False


def command_line(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00', ' ')
    except FileNotFoundError:
        return ''


def stop(signum, frame):
    global STOP
    STOP = True
    if CHILD and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    status('stopped_by_signal')
    raise SystemExit(128 + signum)


def preserve_previous(path, oldout):
    import torch
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert payload['run_id'] == oldout.name and len(payload['rng_state_by_rank']) == 8
    metadata = {k: payload[k] for k in ['global_step', 'epoch', 'best_fid']}
    assert payload['scheduler']['after']['last_epoch'] == metadata['global_step']
    del payload
    destination = oldout / 'persistent-train/checkpoints' / path.name
    if not destination.exists() or destination.stat().st_size != path.stat().st_size:
        partial = destination.with_suffix('.native24576-partial')
        copy_file(path, partial)
        assert partial.stat().st_size == path.stat().st_size
        partial.replace(destination)
    receipt = dict(path=str(path), **metadata, bytes=path.stat().st_size, full_state=True,
                   local_state='COMPLETE', persistent_path=str(destination), persistent_state='COMPLETE',
                   saved_at=time.time())
    write(oldout / 'checkpoint.json', receipt)
    write(QUALITY / 'preserved-previous-checkpoint.json', receipt)
    old = read(oldout / 'status.json')
    old.update(phase='stopped_for_native24576', replacement_run=RUN,
               stop_reason='User requested24576 final codewords and native stochastic RQ without beam')
    write(oldout / 'status.json', old)
    try:
        import wandb
        os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
                          WANDB_BASE_URL='https://api.wandb.ai')
        run = wandb.Api(timeout=60).run('helloimlixin-rutgers/laser/' + oldout.name)
        run.summary['training/status'] = 'stopped_for_native24576'
        run.summary['training/replaced_by'] = RUN
        run.summary['training/preserved_checkpoint_step'] = metadata['global_step']
        run.summary.update()
    except Exception as error:
        write(QUALITY / 'previous-wandb-update-error.json', dict(error=str(error)))
    return receipt


def stop_previous():
    oldout = ROOT / 'outputs/imagenet-beam451-native25-bf16-clip01-cont100-8h100-20261108'
    if (QUALITY / 'previous-stopped.json').exists():
        return None
    state = read(oldout / 'status.json')
    pid = state['supervisor_pid']
    assert alive(pid)
    assert 'runtime/beam451-native25-8h100-20261108/supervise.py' in command_line(pid)
    # Preserve the latest complete local state; persistent copying runs beside fitting.
    files = sorted((SOURCE / 'checkpoints').glob('step-*.pt'), key=lambda p: int(p.stem.split('-')[-1]))
    assert files
    parent = state['child_pid']
    children = Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
    tracked = [int(child) for child in children if 'train.py' in command_line(child)]
    write(QUALITY / 'previous-before-stop.json', state)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 90
    while alive(pid) or any(alive(p) for p in tracked):
        if time.monotonic() > deadline:
            raise RuntimeError('Previous training did not exit after the requested replacement')
        time.sleep(2)
    files = sorted((SOURCE / 'checkpoints').glob('step-*.pt'), key=lambda p: int(p.stem.split('-')[-1]))
    path = files[-1]
    write(QUALITY / 'previous-stopped.json', dict(checkpoint=str(path), time=time.time()))
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return pool, pool.submit(preserve_previous, path, oldout)


def distributed(script, args):
    return [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
            str(script), *args]


def launch(command, phase, extra=None):
    global CHILD
    env = environment()
    env.update(extra or {})
    logroot = QUALITY if phase.startswith('quality-') else OUT
    with (logroot / (phase + '.log')).open('ab', buffering=0) as log:
        CHILD = subprocess.Popen(command, env=env, cwd=BASE, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while CHILD.poll() is None:
        status(phase)
        try:
            CHILD.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    return CHILD.returncode


def publish_quality():
    if (QUALITY / 'cloud-verification.json').exists():
        return read(QUALITY / 'cloud-verification.json')
    os.environ.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
                      WANDB_BASE_URL='https://api.wandb.ai', WANDB_CONSOLE='off', WANDB_DISABLE_GIT='true')
    import wandb
    q = read(QUALITY / 'full-validation.json')
    with wandb.init(entity='helloimlixin-rutgers', project='laser', id=QUALITY.name,
                    name='Native RQ24576 | train-fit codebook | full50k validation',
                    job_type='tokenizer-validation', dir=str(BASE / 'quality-wandb'), resume='allow',
                    settings=wandb.Settings(console='off', disable_git=True, x_disable_stats=True),
                    config=dict(final_codewords=24576, depth=4, fitting_images=50000,
                                heldout_training_images=10000, validation_images=50000,
                                source_CNN_frozen=True, native_temperature=.5,
                                beam_search=False, mixture_teacher=False)) as run:
        table = wandb.Table(columns=['candidate', 'heldout_training_rfid', 'latent_mse'])
        for name, row in read(QUALITY / 'screen-results.json')['results'].items():
            table.add_data(name, row['rfid'], row['latent_mse'])
        run.log({'quality/candidates': table, 'quality/greedy_rfid': q['greedy_rfid'],
                 'quality/native_soft05_rfid': q['stochastic_teacher_rfid'],
                 'quality/native_soft05_entropy': q['mean_entropy']})
        artifact = wandb.Artifact('imagenet-native-rq24576-tokenizer-20261108', type='tokenizer', metadata=q)
        for path in [BASE / 'inputs/joint-codebook.pt', *(BASE / 'inputs').glob('vectors-depth*.pt')]:
            artifact.add_file(str(path), name=path.name, policy='immutable', skip_cache=True)
        for path in QUALITY.glob('*.json'):
            artifact.add_file(str(path), name='verification/' + path.name, policy='immutable', skip_cache=True)
        for path in HERE.glob('*.py'):
            artifact.add_file(str(path), name='runtime/' + path.name, policy='immutable', skip_cache=True)
        artifact = run.log_artifact(artifact)
        artifact.wait(timeout=180)
        receipt = dict(artifact=artifact.qualified_name, state=str(artifact.state), url=run.url,
                       greedy_rfid=q['greedy_rfid'], native_soft05_rfid=q['stochastic_teacher_rfid'])
    write(QUALITY / 'cloud-verification.json', receipt)
    write(BASE / 'inputs/paired-tokenizer-artifact.json', receipt)
    return receipt


def benchmark(microbatch):
    folder = BASE / 'benchmarks' / f'mb{microbatch}'
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'verification.json').exists():
        code = launch(distributed(BASE / 'train.py', ['--epochs', '100', '--microbatch', str(microbatch),
                                                      '--smoke-updates', '5']),
                      f'benchmark-mb{microbatch}', dict(LASER_PRIVATE_SMOKE_DIR=str(folder)))
        if code:
            tail = (OUT / f'benchmark-mb{microbatch}.log').read_text()[-16000:]
            if 'out of memory' in tail.lower():
                return dict(passed=False, microbatch=microbatch, reason='GPU memory limit')
            raise RuntimeError('Native RQ benchmark failed: ' + tail[-3500:])
    proof = read(folder / 'verification.json')
    assert proof['passed'] and proof['frozen_tokenizer_unchanged'] and proof['all_adam_steps'] == 5
    speed, memory = [], []
    for rank in range(8):
        start = read(folder / f'startup-rank{rank}.json')
        target = read(folder / f'targets-rank{rank}.json')
        assert start['fresh_stage2'] and start['fresh_optimizer'] and start['world_size'] == 8
        assert start['final_codewords'] == 24576 and start['native_RQBottleneck']
        assert not start['beam_search'] and not start['mixture_teacher']
        assert start['training_autocast_dtype'] == 'bfloat16' and start['gradient_clip_max_norm'] == .1
        assert target['vocabulary'] == 24576 and target['stochastic_prefixes'] and target['soft_targets']
        assert target['native_RQBottleneck'] and target['entropy'] > 0
        rows = [json.loads(line) for line in (folder / f'throughput-rank{rank}.jsonl').read_text().splitlines()]
        speed.append(statistics.median(row['images_per_second'] for row in rows if row['step'] >= 3))
        memory.append(read(folder / f'progress-rank{rank}.json')['peak_allocated_gib'])
    return dict(passed=True, microbatch=microbatch, images_per_second=min(speed),
                peak_allocated_gib=max(memory), parameter_count=start['parameter_count'],
                optimizer_state_count=proof['optimizer_state_count'])


def main():
    lock = (BASE / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, stop)
    preservation = stop_previous()
    phases = [('cache', QUALITY / 'cache-verification.json'),
              ('fit', QUALITY / 'fit-trace.json'), ('screen', QUALITY / 'selected.json'),
              ('full', QUALITY / 'full-validation.json')]
    for phase, receipt in phases:
        done = receipt.exists()
        if done and phase == 'fit':
            done = read(receipt)['trace'][-1]['iteration'] == 24
        if not done:
            code = launch(distributed(BASE / 'runtime/quality.py', [phase]), 'quality-' + phase)
            if code:
                raise RuntimeError(f'Tokenizer phase {phase} failed')
    if preservation:
        pool, future = preservation
        future.result()
        pool.shutdown()
    quality = read(QUALITY / 'full-validation.json')
    assert quality['passed'], 'New greedy codebook did not beat the earlier native16k reconstruction baseline'
    assert quality['vocabulary'] == 24576 and quality['images'] == 50000
    assert quality['mean_entropy'] > 0 and not quality['beam_search'] and not quality['mixture_teacher']
    artifact = publish_quality()
    recipe = OmegaConf.load(BASE / 'inputs/recipe.yaml')
    recipe.tokenizer.codebook_sha256 = quality['codebook_sha256']
    recipe.tokenizer.measured_rfid = quality['greedy_rfid']
    recipe.tokenizer.teacher_sampled_rfid = quality['stochastic_teacher_rfid']
    recipe.targets.measured_entropy = quality['mean_entropy']
    recipe.lineage.tokenizer_artifact = artifact['artifact']
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    candidates = [benchmark(64), benchmark(128)]
    selected = max((row for row in candidates if row['passed']), key=lambda row: row['images_per_second'])
    microbatch = selected['microbatch']
    write(OUT / 'throughput-benchmarks.json', dict(candidates=candidates, selected=selected))
    folder = BASE / 'benchmarks/resume'
    folder.mkdir(exist_ok=True)
    if not (folder / 'verification.json').exists():
        source = read(BASE / 'smoke/checkpoint.json')
        code = launch(distributed(BASE / 'train.py', ['--epochs', '100', '--microbatch', str(microbatch),
                      '--resume', source['path'], '--smoke-updates', '6']), 'resume-preflight',
                      dict(LASER_PRIVATE_SMOKE_DIR=str(folder)))
        if code:
            raise RuntimeError('Native24576 full optimizer/scheduler/RNG restore test failed')
    assert read(folder / 'verification.json')['passed']
    recipe.microbatch = microbatch
    OmegaConf.save(recipe, BASE / 'inputs/recipe.yaml')
    execution = read(BASE / 'inputs/execution.json')
    execution.update(microbatch=microbatch, accumulation_steps=2048 // (8 * microbatch),
                     measured_images_per_second=selected['images_per_second'])
    write(BASE / 'inputs/execution.json', execution)
    write(OUT / 'preflight-verification.json', dict(passed=True, final_codewords=24576,
          fresh_model_and_optimizer=True, full_state_restore=True, native_quantizer=True,
          beam_search=False, mixture_teacher=False, hardware=selected, tokenizer=quality))
    repro = OUT / 'reproducibility'
    repro.mkdir(exist_ok=True)
    for path in (BASE / 'inputs').iterdir():
        if path.is_file() and path.stat().st_size < 50 * 1024**2:
            shutil.copyfile(path, repro / path.name)
    for path in HERE.glob('*.py'):
        shutil.copyfile(path, repro / path.name)
    for path in [QUALITY / 'full-validation.json', QUALITY / 'cache-verification.json']:
        shutil.copyfile(path, repro / path.name)
    shutil.copyfile(BASE / 'inputs/recipe.yaml', ROOT / 'configs/stage2' / (RUN + '.yaml'))
    attempts = {}
    while not STOP:
        step = 0
        args = ['--epochs', '100', '--microbatch', str(microbatch)]
        if (OUT / 'checkpoint.json').exists():
            checkpoint = read(OUT / 'checkpoint.json')
            path = Path(checkpoint['path'])
            if not path.exists():
                path = Path(checkpoint['persistent_path'])
            assert path.stat().st_size == checkpoint['bytes']
            step = checkpoint['global_step']
            args += ['--resume', str(path)]
        attempts[step] = attempts.get(step, 0) + 1
        if attempts[step] > 3:
            raise RuntimeError(f'Repeated native24576 training failure from step{step}')
        command = distributed(BASE / 'train.py', args)
        write(OUT / 'launch.json', dict(command=command, world_size=8, epochs=100,
                                      global_batch=2048, microbatch=microbatch, resume_step=step, time=time.time()))
        code = launch(command, 'training')
        if code == 0:
            assert read(OUT / 'completed.json')['epochs'] == 100
            status('completed')
            return
        status('recovering', exit_code=code, resume_step=step)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
