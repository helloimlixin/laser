"""Run verified four-GPU preflights and supervise the fresh dropout trial.

Copy into the prepared run directory before executing.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

BASE = Path(__file__).resolve().parent
ROOT = Path('/mnt/laser-church/dropout-experiment')
child = None


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def status(phase, **extra):
    row = dict(phase=phase, supervisor_pid=os.getpid(), child_pid=child.pid if child else None,
               updated_unix=time.time(), **extra)
    atomic(BASE / 'status.json', row)
    print(json.dumps(row), flush=True)


def execute(command, name, env, phase):
    global child
    with (BASE / name).open('a') as stream:
        child = subprocess.Popen(command, cwd=BASE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    status(phase)
    while True:
        try:
            code = child.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            status(phase)
    child = None
    if code:
        raise RuntimeError(f'{phase} exited {code}; inspect {name}')


def main():
    lock = Path('/mnt/laser-church/supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NCCL_NVLS_ENABLE='0', PYTHONUNBUFFERED='1',
        TORCH_HOME='/mnt/laser-church/torch-cache', WANDB_MODE='online', WANDB_DISABLE_GIT='true',
        WANDB_ENTITY='helloimlixin-rutgers', WANDB_PROJECT='laser',
        WANDB_API_KEY=Path('/root/.config/laser/wandb-api-key').read_text().strip(),
        WANDB_CACHE_DIR='/mnt/laser-church/wandb-cache', WANDB_DATA_DIR='/mnt/laser-church/wandb-data',
        WANDB_DIR='/mnt/laser-church/wandb', LASER_CHECKPOINT_STAGING_DIR=str(ROOT/'serialization'))
    for name in ['WANDB_SERVICE', '_WANDB_SERVICE', 'WANDB_RUN_ID', 'WANDB_NAME', 'WANDB_RESUME', 'WANDB_RESUME_MODE', 'SMOKE_TEST']:
        env.pop(name, None)
    launcher = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4']
    rows = []
    for accumulation in [4, 2]:
        mode = f'benchmark-a{accumulation}'
        if not (BASE/mode/'complete.json').exists():
            execute(launcher+[str(BASE/'train.py'), '--mode', mode, '--steps', '12'], mode+'.log',
                    dict(env, LASER_ACCUMULATION=str(accumulation)), mode)
        assert read(BASE/mode/'complete.json')['passed']
        reports = [read(BASE/mode/f'performance-rank{rank}.json') for rank in range(4)]
        seconds = max(report['measured_mean_seconds'] for report in reports)
        rows.append(dict(accumulation=accumulation, seconds_per_update=seconds,
                         images_per_second=2048/seconds,
                         peak_allocated_gib=max(r['peak_allocated_gib'] for r in reports)))
    selected = min(rows, key=lambda row: row['seconds_per_update'])
    override = read(BASE/'plan.json').get('accumulation_override')
    if override is not None:
        selected = next(row for row in rows if row['accumulation'] == override)
    atomic(BASE/'benchmark-selection.json', dict(passed=True, rows=rows, selected=selected))
    if not (BASE/'fid-benchmark.json').exists():
        execute(launcher+[str(BASE/'benchmark_fid.py')], 'fid-benchmark.log', env, 'benchmark-fid')
    assert read(BASE/'fid-benchmark.json')['passed']
    assert read(BASE/'validation.json')['passed']
    assert read(BASE/'audit-numerical.json')['passed']
    assert read(BASE/'heldout-probe.json')['passed']
    plan = read(BASE/'plan.json')
    plan.update(accumulation_steps=selected['accumulation'],
                fid_batch_size=read(BASE/'fid-benchmark.json')['selected_batch_size_per_gpu'])
    atomic(BASE/'plan.json', plan)
    if not (BASE/'train/provenance-upload.json').exists():
        # Path.rglob does not follow the runtime symlink. Enumerate it explicitly.
        files = {str(p.relative_to(BASE)): p for p in BASE.rglob('*.py') if '__pycache__' not in p.parts}
        runtime = (BASE/'runtime').resolve()
        files.update({'runtime/'+str(p.relative_to(runtime)): p for p in runtime.rglob('*.py') if '__pycache__' not in p.parts})
        atomic(BASE/'source-manifest.json', {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(files.items())})
        with tarfile.open(BASE/'runtime-source.tar.gz', 'w:gz') as archive:
            for name, path in sorted(files.items()):
                archive.add(path, arcname=name, recursive=False)
            for name in ['audit-numerical.json', 'source-changes.json', 'validation.json']:
                archive.add(BASE/name, arcname=name, recursive=False)
    if not (BASE/'train/request.json').exists():
        assert not (BASE/'train/checkpoints/last.pt').exists(), 'Fresh launch must not consume a preflight checkpoint'
    execute(launcher+[str(BASE/'train.py'), '--mode', 'train'], 'train.log',
            dict(env, LASER_ACCUMULATION=str(selected['accumulation'])), 'training')
    status('completed', result=read(BASE/'train/complete.json'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error=repr(error))
        raise
