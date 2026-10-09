"""Run two matched ImageNet pair-memory arms, then restore the original job."""
from concurrent.futures import ThreadPoolExecutor
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

BASE = Path('/tmp/laser-imagenet-pair-memory-20261007')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-trial-20261007')
PREVIOUS = Path('/workspace/Projects/laser/outputs/imagenet-cross-attention-trial-20261007')
OLD_BASE = Path('/tmp/laser-imagenet-cross-attention-20261007')
ASSETS = Path('/tmp/laser-imagenet-classcond-20261007')
PRODUCTION = Path('/workspace/Projects/laser/outputs/imagenet-rfid421-classcond-8h100-20261007')
sys.path[:0] = [str(BASE/'source'), str(BASE/'source/runtime')]
import torch
from src.training import k4_checkpoint_io as io

lock = (BASE/'launch.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
env = dict(os.environ,
    WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
    WANDB_DIR=str(BASE/'wandb'), WANDB_CACHE_DIR=str(BASE/'wandb-cache'),
    WANDB_DATA_DIR=str(BASE/'wandb-data'), WANDB_ARTIFACT_DIR=str(BASE/'wandb-artifacts'),
    TORCH_HOME=str(ASSETS/'torch-cache'), TORCHINDUCTOR_CACHE_DIR=str(ASSETS/'inductor-cache'),
    TORCHINDUCTOR_COMPILE_THREADS='4', MPLCONFIGDIR=str(BASE/'matplotlib'),
    PYTHONPATH=str(BASE/'source')+':'+str(BASE/'source/runtime'),
    CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', LASER_ACCUMULATION='4',
    LASER_PAIR_MEMORY_BASE=str(BASE), LASER_PAIR_MEMORY_OUTPUT=str(OUT),
    LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
    LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'),
    LASER_CHECKPOINT_IMMUTABLE_FILES='1',
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
    PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0')


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def record(value):
    value.update(time=time.time(), supervisor_pid=os.getpid())
    write(OUT/'launch-status.json', value)
    with (OUT/'launch-history.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    print(json.dumps(value), flush=True)


def alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except (OSError, ProcessLookupError):
        return False


def pin_anchor():
    os.environ['LASER_CHECKPOINT_UPLOAD_CACHE_DIR'] = str(OLD_BASE/'checkpoint-cache')
    os.environ['LASER_CHECKPOINT_IMMUTABLE_FILES'] = '1'
    source = PRODUCTION/'train/checkpoints/last.pt'
    resolved = source.resolve()
    local = io._checkpoint_upload_source(resolved)
    assert local != resolved, 'Anchor requires retained local serialization'
    os.link(local, BASE/'anchor.pt')
    payload = torch.load(BASE/'anchor.pt', map_location='cpu', mmap=True, weights_only=False)
    step = payload['global_step']
    assert len(payload['optimizer']['state']) == 798
    assert {int(value['step']) for value in payload['optimizer']['state'].values()} == {step}
    assert payload['scheduler']['last_epoch'] == step
    assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 8
    assert not any(key.startswith('pair_memory_queries.') for key in payload['state_dict'])
    plan = json.loads((OUT/'plan.json').read_text())
    plan.update(anchor_step=step, endpoint_step=step+plan['updates'],
                anchor_epoch=payload['epoch'], anchor_batch_idx=payload.get('batch_idx'))
    write(OUT/'plan.json', plan)
    report = dict(passed=True, global_step=step, epoch=payload['epoch'], batch_idx=payload.get('batch_idx'),
        world_size=8, old_optimizer_states=798, scheduler_age=step, rng_ranks=8,
        bytes=(BASE/'anchor.pt').stat().st_size, source=str(resolved),
        pinned_checkpoint=str(BASE/'anchor.pt'), sha256=hashlib.file_digest((BASE/'anchor.pt').open('rb'), 'sha256').hexdigest())
    write(OUT/'anchor-verification.json', report)
    return plan, payload


def stop_previous():
    status = json.loads((PREVIOUS/'launch-status.json').read_text())
    assert status['branch'] == 'baseline' and status['phase'] == 'running'
    workers = [json.loads((PREVIOUS/f'verification/baseline/train/architecture-rank{rank}.json').read_text())['pid'] for rank in range(8)]
    pids = [status['supervisor_pid'], status['training_pid'], *workers]
    for pid in pids:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        assert str(OLD_BASE).encode() in command, f'Unrecognized process {pid}'
    progress = json.loads((PREVIOUS/'verification/baseline/train/progress-rank0.json').read_text())
    for pid in pids[:2]:
        if alive(pid):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+40
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(1)
    for pid in pids:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
    write(OUT/'baseline-handoff.json', dict(
        stopped_pids=pids, last_observed_step=progress['global_step'],
        resume_step=json.loads((OUT/'plan.json').read_text())['anchor_step'], time=time.time()))
    production_status = dict(phase='paired_trial', supervisor_pid=os.getpid(),
        trial_status=str(OUT/'launch-status.json'), baseline_resume_checkpoint=str(BASE/'anchor.pt'),
        baseline_will_resume_after_trial=True, time=time.time())
    write(PRODUCTION/'launch-status.json', production_status)


def seed_cache(source, pinned=None):
    source = Path(source).resolve()
    key = hashlib.sha256(str(source).encode()).hexdigest()
    local = BASE/'checkpoint-cache/objects'/f'{key}.pt'
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.is_file():
        if pinned is not None:
            os.link(pinned, local)
        else:
            shutil.copyfile(source, local)
    assert source.stat().st_size == local.stat().st_size
    stat = source.stat()
    identity = {name:getattr(stat,name) for name in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']}
    write(local.with_suffix('.json'), dict(source=str(source), identity=identity))


def run(branch, phase):
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
               str(BASE/'entry.py'), '--config', str(BASE/f'{branch}-{phase}.yaml')]
    with (OUT/f'{branch}-{phase}.log').open('a') as log:
        process = subprocess.Popen(command, cwd=BASE/'source', env=dict(env, LASER_BRANCH=branch, LASER_PHASE=phase),
                                   stdout=log, stderr=subprocess.STDOUT)
        record(dict(phase='running', branch=branch, operation=phase, training_pid=process.pid, command=command))
        if branch == 'baseline':
            write(PRODUCTION/'launch-status.json', dict(phase='running', branch=branch, operation=phase,
                supervisor_pid=os.getpid(), training_pid=process.pid, log=str(OUT/f'{branch}-{phase}.log'),
                resume_step=json.loads((OUT/'plan.json').read_text())['anchor_step'], time=time.time()))
        result = process.wait()
    record(dict(phase='completed' if result == 0 else 'failed', branch=branch, operation=phase, returncode=result))
    if result:
        raise RuntimeError(f'{branch} {phase} failed with exit code {result}')


def verify_endpoint(branch, plan):
    os.environ.update(LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(BASE/'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    path = OUT/branch/'train/checkpoints/last.pt'
    local = io._checkpoint_upload_source(path.resolve())
    assert local != path.resolve()
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    assert payload['global_step'] == payload['scheduler']['last_epoch'] == plan['endpoint_step']
    assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 8
    states = payload['optimizer']['state']
    assert {int(states[index]['step']) for index in range(798)} == {plan['endpoint_step']}
    assert {int(states[index]['step']) for index in range(798,len(states))} == {plan['updates']}
    weights = {key:value for key,value in payload['state_dict'].items() if key.startswith('pair_memory_queries.')}
    assert len(weights) == len(states)-798 and all(torch.isfinite(value).all() for value in weights.values())
    assert sum(value.numel() for value in weights.values()) == plan['added_parameters'][branch]
    assert all(weights[f'pair_memory_queries.{name}_output.weight'].abs().sum() > 0 for name in ['atom','coefficient'])
    write(OUT/f'{branch}-endpoint-verification.json', dict(passed=True, global_step=payload['global_step'],
        epoch=payload['epoch'], batch_idx=payload.get('batch_idx'), old_adam_age=plan['endpoint_step'],
        new_adam_age=plan['updates'], checkpoint=str(path.resolve()), local_serialization=str(local),
        bytes=local.stat().st_size, rng_ranks=8, added_parameters=plan['added_parameters'][branch], time=time.time()))


def compare(plan):
    metrics, seeds = {}, {}
    for branch in ['cross', 'mlp']:
        path = OUT/branch/f'train/evaluations/fid_step_{plan["endpoint_step"]:07d}.json'
        metrics[branch] = json.loads(path.read_text())
        assert metrics[branch]['num_generated_samples'] == plan['fid_samples']
        seeds[branch] = [json.loads((OUT/f'verification/{branch}/eval/fid-sampling-rank{rank}.json').read_text()) for rank in range(8)]
        assert all(value['completed'] and value['training_rng_restored'] for value in seeds[branch])
    assert [value['cuda_rng_sha256'] for value in seeds['cross']] == [value['cuda_rng_sha256'] for value in seeds['mlp']]
    write(OUT/'comparison.json', dict(anchor_step=plan['anchor_step'], endpoint_step=plan['endpoint_step'],
        updates=plan['updates'], fid_samples=plan['fid_samples'], metrics=metrics,
        fid_delta_cross_minus_mlp=metrics['cross']['fid']-metrics['mlp']['fid'],
        matched_seeds_verified=True, no_automatic_promotion=True, time=time.time()))


def main():
    plan, anchor = pin_anchor()
    stop_previous()
    pool = ThreadPoolExecutor(max_workers=2)
    futures = []
    error = None
    try:
        best_sources = {path for field in ['best_fid','best_inception'] for _,path in anchor.get(field,[])}
        futures = [pool.submit(seed_cache, path) for path in best_sources]
        del anchor
        for branch in ['cross','mlp']:
            run(branch,'train')
            verify_endpoint(branch,plan)
            run(branch,'eval')
        compare(plan)
    except Exception as exception:
        error = repr(exception)
        record(dict(phase='trial_failed', error=error, baseline_recovery_pending=True))
    finally:
        for future in futures:
            try:
                future.result()
            except Exception as exception:
                record(dict(phase='best_cache_failed', error=repr(exception)))
        pool.shutdown()
        record(dict(phase='resuming_baseline', baseline_resume_step=plan['anchor_step'], trial_error=error))
        run('baseline','train')
    return 0 if error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
