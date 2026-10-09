"""Compare the frozen pair-memory endpoints with a matched plain continuation.

Keeps the original schedule and all optimizer/RNG state. Evaluations use separate
outputs for each sampling seed; production resumes from the plain endpoint.
"""
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

BASE = Path('/tmp/laser-imagenet-pair-memory-investigation-20261007')
OUT = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-investigation-20261007')
PAIR_BASE = Path('/tmp/laser-imagenet-pair-memory-20261007')
PAIR_OUT = Path('/workspace/Projects/laser/outputs/imagenet-pair-memory-trial-20261007')
ASSETS = Path('/tmp/laser-imagenet-classcond-20261007')
PRODUCTION = Path('/workspace/Projects/laser/outputs/imagenet-rfid421-classcond-8h100-20261007')
ENDPOINT = 3798
# Separate by more than world_size: seed+rank streams must not overlap across
# the two evaluations (adjacent base seeds would reuse seven of eight streams).
SEEDS = (261001, 271001)

METRIC_AUDIT = '''
# Read-only metric audit: native computation finishes before moments are saved.
from src import rqvae_metrics as audited_metrics
native_metric_compute = audited_metrics.DistributedOriginalRQVAEMetrics.compute
def audited_metric_compute(metric, *args, **kwargs):
    result = native_metric_compute(metric, *args, **kwargs)
    if PHASE.startswith('seed-') and int(os.environ['RANK']) == 0:
        import numpy as np
        count = int(metric.fake_count.item())
        assert count == 50000
        mu, covariance = audited_metrics._mean_covariance(metric.fake_sum, metric.fake_cross, count)
        real_mu, real_covariance = audited_metrics.load_reference_statistics(metric.reference_stats_path)
        mean_term = float(np.square(real_mu-mu).sum())
        folder = OUT / 'verification' / BRANCH / PHASE
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / 'fid-feature-moments.npz', mu=mu, sigma=covariance, samples=count)
        record('fid-decomposition.json', dict(fid=result[0], mean_term=mean_term,
            covariance_term=result[0]-mean_term, generated_samples=count,
            real_covariance_trace=float(np.trace(real_covariance)),
            generated_covariance_trace=float(np.trace(covariance)),
            native_metric_unchanged=True, moments_file=str(folder / 'fid-feature-moments.npz')))
    return result
audited_metrics.DistributedOriginalRQVAEMetrics.compute = audited_metric_compute
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def record(**value):
    value.update(time=time.time(), supervisor_pid=os.getpid())
    write(OUT / 'launch-status.json', value)
    with (OUT / 'launch-history.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def prepare():
    import yaml
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (BASE / 'source').exists():
        shutil.copytree(PAIR_BASE / 'source', BASE / 'source', copy_function=shutil.copyfile,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for mode in ('cross', 'mlp'):
        shutil.copyfile(PAIR_BASE / f'initial-{mode}.pt', BASE / f'initial-{mode}.pt')
    helper = Path(__file__).parent / 'pair_memory_ablation.py'
    if helper.resolve() != (BASE / 'pair_memory_ablation.py').resolve():
        shutil.copyfile(helper, BASE / 'pair_memory_ablation.py')
    shutil.copyfile(BASE / 'pair_memory_ablation.py', OUT / 'pair_memory_ablation.py')
    helper = Path(__file__).parent / 'restore_evaluated_plain.py'
    if helper.resolve() != (BASE / 'restore_evaluated_plain.py').resolve():
        shutil.copyfile(helper, BASE / 'restore_evaluated_plain.py')
    shutil.copyfile(BASE / 'restore_evaluated_plain.py', OUT / 'restore_evaluated_plain.py')
    entry = (PAIR_BASE / 'entry.py').read_text()
    entry = entry.replace("ANCHOR_STEP = PLAN['anchor_step']",
                          "ANCHOR_STEP = int(os.environ.get('LASER_RESUME_STEP', PLAN['anchor_step'] if BRANCH == 'baseline' else PLAN['original_architecture_anchor_step']))")
    entry = entry.replace("kwargs['resume'] = 'must' if BRANCH == 'baseline' else 'allow'",
                          "kwargs['resume'] = os.environ.get('LASER_WANDB_RESUME', 'allow')")
    entry = entry.replace("        new_state = {'pair_memory_queries.'",
        "        if PHASE.endswith('memory-bos'):\n            from pair_memory_ablation import use_bos_only_memory\n            use_bos_only_memory(model.pair_memory_queries)\n        new_state = {'pair_memory_queries.'")
    entry = entry.replace('        value.pair_memory_query_mode = BRANCH',
        "        value.pair_memory_ablation = 'bos' if PHASE.endswith('memory-bos') else None\n        value.pair_memory_query_mode = BRANCH")
    entry = entry.replace('    return payload\n',
        '    from restore_evaluated_plain import restore_plain_best\n    return restore_plain_best(payload,path,OUT,record)\n')
    entry = entry.replace('    wb = native_wandb_init(*args, **kwargs)',
        '    wb = native_wandb_init(*args, **kwargs)\n    from restore_evaluated_plain import log_restored_plain_metrics\n    log_restored_plain_metrics(wb)')
    entry = entry.replace("if __name__ == '__main__':", METRIC_AUDIT + "\nif __name__ == '__main__':")
    (BASE / 'entry.py').write_text(entry)
    shutil.copyfile(BASE / 'entry.py', OUT / 'entry.py')
    plain = yaml.safe_load((PAIR_BASE / 'baseline-train.yaml').read_text())
    plain['options'].update(
        output=str(OUT / 'baseline/train'), checkpoint_dir=str(OUT / 'baseline/train/checkpoints'),
        resume_checkpoint=str(BASE / 'anchor.pt'), fid_every=0, fid_early_epochs=0,
        sample_grid_every=0, sample_grid_on_start=False,
        upload_checkpoints=False, save_step_freq=512, save_ckpt_freq=100,
        wandb_id='imagenet-rfid421-plain-matched-3798-20261007',
        wandb_name='ImageNet rFID4.21 | plain matched step 3798 | investigation')
    # Finalized after pinning the newest pre-FID plain checkpoint.
    (BASE / 'baseline-train.yaml').write_text(yaml.safe_dump(plain, sort_keys=False))
    for branch in ('baseline', 'cross', 'mlp'):
        for seed in SEEDS:
            if branch != 'baseline' and seed == SEEDS[0]:
                continue  # Already evaluated with this exact frozen protocol.
            config = yaml.safe_load((PAIR_BASE / 'cross-eval.yaml').read_text())
            checkpoint = (OUT / 'baseline/train/checkpoints/last.pt' if branch == 'baseline'
                          else PAIR_OUT / branch / 'train/checkpoints/last.pt')
            output = OUT / branch / f'seed-{seed}'
            config['options'].update(output=str(output), checkpoint_dir=str(output / 'checkpoints'),
                resume_checkpoint=str(checkpoint), fid_seed=seed,
                sample_grid_on_start=seed == SEEDS[0], sample_grid_every=0,
                wandb_id=f'imagenet-rfid421-investigate-{branch}-{seed}-20261007',
                wandb_name=f'ImageNet rFID4.21 | {branch} step3798 | sampling seed {seed}')
            (BASE / f'{branch}-seed-{seed}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    config = yaml.safe_load((BASE / f'cross-seed-{SEEDS[1]}.yaml').read_text())
    phase = f'seed-{SEEDS[1]}-memory-bos'
    output = OUT / 'cross' / phase
    config['options'].update(output=str(output), checkpoint_dir=str(output/'checkpoints'),
        sample_grid_on_start=True,
        wandb_id=f'imagenet-rfid421-cross-memory-bos-{SEEDS[1]}-20261007',
        wandb_name=f'ImageNet rFID4.21 | cross step3798 | BOS memory ablation | seed {SEEDS[1]}')
    (BASE / f'cross-{phase}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(PAIR_BASE / 'baseline-train.yaml', BASE / 'production-train.yaml')
    # Native resume rebases best-FID filenames into checkpoint_dir. Retain
    # references when moving the same plain training state to comparison output.
    for phase in ('train', *(f'seed-{seed}' for seed in SEEDS)):
        folder = OUT / 'baseline' / phase / 'checkpoints'
        folder.mkdir(parents=True, exist_ok=True)
        for source in (PRODUCTION / 'train/checkpoints').glob('best*.pt'):
            target = folder / source.name
            if not target.exists():
                target.symlink_to(source)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def handoff(io, torch):
    import yaml
    status = json.loads((PAIR_OUT / 'launch-status.json').read_text())
    assert status['branch'] == 'baseline' and status['phase'] == 'running'
    workers = [json.loads((PAIR_OUT / f'verification/baseline/train/architecture-rank{r}.json').read_text())['pid']
               for r in range(8)]
    pids = [status['supervisor_pid'], status['training_pid'], *workers]
    for pid in pids:
        assert str(PAIR_BASE).encode() in Path(f'/proc/{pid}/cmdline').read_bytes(), pid
    source = (PRODUCTION / 'train/checkpoints/last.pt').resolve()
    local = io._checkpoint_upload_source(source)
    assert local != source
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    step = payload['global_step']
    if step >= 2504:  # The production evaluation at 2504 consumes RNG.
        local = BASE / 'baseline-anchor.pt'  # Previously pinned at 2000.
        payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
        step = payload['global_step']
    assert 1750 <= step < 2504
    assert payload['scheduler']['last_epoch'] == step
    assert len(payload['optimizer']['state']) == 798
    assert {int(v['step']) for v in payload['optimizer']['state'].values()} == {step}
    assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 8
    anchor = BASE / 'anchor.pt'
    assert not anchor.exists(), 'A handoff is single-use; inspect the existing launch before retrying'
    os.link(local, anchor)
    plan = json.loads((PAIR_OUT / 'plan.json').read_text())
    plan.update(anchor_step=step, endpoint_step=ENDPOINT, baseline_remaining_updates=ENDPOINT-step,
                original_architecture_anchor_step=1750, seeds=list(SEEDS),
                original_architecture_anchor_epoch=plan['anchor_epoch'],
                original_architecture_anchor_batch_idx=plan['anchor_batch_idx'],
                anchor_epoch=payload['epoch'],anchor_batch_idx=payload.get('batch_idx'),
                plain_continuation_no_intermediate_fid=True,
                sampling_replication_only=True, production_resume_after_investigation=True)
    write(OUT / 'plan.json', plan)
    config = yaml.safe_load((BASE / 'baseline-train.yaml').read_text())
    config['options']['max_optimizer_steps'] = ENDPOINT-step
    (BASE / 'baseline-train.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for path in BASE.glob('*.yaml'):
        shutil.copyfile(path, OUT / path.name)
    for pid in pids[:2]:
        if alive(pid):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+40
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(1)
    for pid in pids:
        if alive(pid):
            os.kill(pid, signal.SIGKILL)
    write(OUT / 'baseline-handoff.json', dict(stopped_pids=pids, resume_step=step,
          checkpoint=str(anchor), original_persistent_source=str(source), time=time.time()))
    write(PAIR_OUT / 'launch-status.json', dict(phase='handed_off', investigation=str(OUT), time=time.time()))
    write(PRODUCTION / 'launch-status.json', dict(phase='investigation', supervisor_pid=os.getpid(),
          investigation=str(OUT), baseline_resume_after_investigation=True, time=time.time()))
    record(phase='handoff_completed', baseline_anchor_step=step, endpoint_step=ENDPOINT)


def run(env, branch, phase, *, resume_step=None, production=False):
    config = BASE / ('production-train.yaml' if production else f'{branch}-{phase}.yaml')
    child_env = dict(env, LASER_BRANCH=branch, LASER_PHASE=phase)
    if resume_step is not None:
        child_env['LASER_RESUME_STEP'] = str(resume_step)
    if production:
        child_env['LASER_WANDB_RESUME'] = 'must'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
               str(BASE / 'entry.py'), '--config', str(config)]
    log_path = OUT / ('production-train.log' if production else f'{branch}-{phase}.log')
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=BASE / 'source', env=child_env,
                                   stdout=log, stderr=subprocess.STDOUT)
        record(phase='running', branch=branch, operation=phase, training_pid=process.pid,
               production=production, log=str(log_path))
        if production:
            write(PRODUCTION / 'launch-status.json', dict(phase='running', branch=branch,
                  supervisor_pid=os.getpid(), training_pid=process.pid, log=str(log_path),
                  resume_step=resume_step, time=time.time()))
        result = process.wait()
    record(phase='completed' if result == 0 else 'failed', branch=branch,
           operation=phase, returncode=result)
    if result:
        raise RuntimeError(f'{branch} {phase} exited {result}')


def verify_plain(io, torch):
    checkpoint = OUT / 'baseline/train/checkpoints/last.pt'
    local = io._checkpoint_upload_source(checkpoint.resolve())
    assert local != checkpoint.resolve()
    payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
    assert payload['global_step'] == payload['scheduler']['last_epoch'] == ENDPOINT
    assert len(payload['optimizer']['state']) == 798
    assert {int(v['step']) for v in payload['optimizer']['state'].values()} == {ENDPOINT}
    assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 8
    assert not any(k.startswith('pair_memory_queries.') for k in payload['state_dict'])
    write(OUT / 'baseline-endpoint-verification.json', dict(passed=True, global_step=ENDPOINT,
          epoch=payload['epoch'], batch_idx=payload.get('batch_idx'), optimizer_states=798,
          old_adam_age=ENDPOINT, scheduler_age=ENDPOINT, rng_ranks=8,
          checkpoint=str(checkpoint.resolve()), local_serialization=str(local)))


def compare():
    metrics, seeds = {}, {}
    for branch in ('baseline', 'cross', 'mlp'):
        metrics[branch], seeds[branch] = {}, {}
        for seed in SEEDS:
            old = branch != 'baseline' and seed == SEEDS[0]
            folder = PAIR_OUT / branch / 'train' if old else OUT / branch / f'seed-{seed}'
            report = json.loads((folder / f'evaluations/fid_step_{ENDPOINT:07d}.json').read_text())
            assert report['num_generated_samples'] == 50000 and report['global_step'] == ENDPOINT
            assert report['fid_seed'] == seed and report['metric_backend'] == 'original-rqvae'
            verification = (PAIR_OUT / 'verification' / branch / 'eval' if old
                            else OUT / 'verification' / branch / f'seed-{seed}')
            ranks = [json.loads((verification / f'fid-sampling-rank{r}.json').read_text()) for r in range(8)]
            assert all(r['completed'] and r['training_rng_restored'] for r in ranks)
            metrics[branch][str(seed)] = report
            seeds[branch][str(seed)] = [r['cuda_rng_sha256'] for r in ranks]
    for seed in SEEDS:
        assert seeds['baseline'][str(seed)] == seeds['cross'][str(seed)] == seeds['mlp'][str(seed)]
    phase = f'seed-{SEEDS[1]}-memory-bos'
    bos = json.loads((OUT / 'cross' / phase / f'evaluations/fid_step_{ENDPOINT:07d}.json').read_text())
    ranks = [json.loads((OUT / 'verification/cross' / phase / f'fid-sampling-rank{r}.json').read_text()) for r in range(8)]
    assert all(r['completed'] and r['training_rng_restored'] for r in ranks)
    assert [r['cuda_rng_sha256'] for r in ranks] == seeds['cross'][str(SEEDS[1])]
    assert bos['num_generated_samples'] == 50000 and bos['global_step'] == ENDPOINT and bos['fid_seed'] == SEEDS[1]
    write(OUT / 'comparison.json', dict(endpoint_step=ENDPOINT, fid_samples=50000, metrics=metrics,
          bos_memory_inference_ablation=bos,
          bos_memory_FID_minus_full=bos['fid']-metrics['cross'][str(SEEDS[1])]['fid'],
          matched_sampling_rng_verified=True, independent_training_replicates=False,
          deltas={str(seed): {control: metrics['cross'][str(seed)]['fid']-metrics[control][str(seed)]['fid']
                  for control in ('baseline', 'mlp')} for seed in SEEDS}, time=time.time()))


def main():
    import yaml
    attached = json.loads((OUT / 'launch-status.json').read_text()) if '--attach-baseline' in sys.argv else None
    prepare()
    lock = (BASE / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path[:0] = [str(BASE / 'source'), str(BASE / 'source/runtime')]
    import torch
    from src.training import k4_checkpoint_io as io
    env = dict(os.environ,
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        WANDB_DIR=str(BASE / 'wandb'), WANDB_CACHE_DIR=str(BASE / 'wandb-cache'),
        WANDB_DATA_DIR=str(BASE / 'wandb-data'), WANDB_ARTIFACT_DIR=str(BASE / 'wandb-artifacts'),
        TORCH_HOME=str(ASSETS / 'torch-cache'), TORCHINDUCTOR_CACHE_DIR=str(ASSETS / 'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='4', MPLCONFIGDIR=str(BASE / 'matplotlib'),
        PYTHONPATH=str(BASE / 'source')+':'+str(BASE / 'source/runtime'),
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', LASER_ACCUMULATION='4',
        LASER_PAIR_MEMORY_BASE=str(BASE), LASER_PAIR_MEMORY_OUTPUT=str(OUT),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(PAIR_BASE / 'checkpoint-cache'),
        LASER_CHECKPOINT_IMMUTABLE_FILES='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0')
    os.environ.update({k:v for k,v in env.items() if k.startswith('LASER_CHECKPOINT_')})
    if '--resume-handoff' in sys.argv or attached is not None:
        plan = json.loads((OUT / 'plan.json').read_text())
        assert (BASE / 'anchor.pt').is_file()
        assert attached is not None or not (OUT / 'baseline/train/checkpoints/last.pt').exists()
        plan['seeds'] = list(SEEDS)
        write(OUT / 'plan.json', plan)
        config = yaml.safe_load((BASE / 'baseline-train.yaml').read_text())
        config['options']['max_optimizer_steps'] = plan['baseline_remaining_updates']
        (BASE / 'baseline-train.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copyfile(BASE / 'baseline-train.yaml', OUT / 'baseline-train.yaml')
    else:
        handoff(io, torch)
    resume = BASE / 'anchor.pt'
    error = None
    try:
        if attached is None:
            run(env, 'baseline', 'train')
        else:
            assert attached['branch'] == 'baseline' and attached['operation'] == 'train' and not attached['production']
            pid = attached['training_pid']
            assert str(BASE).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
            record(phase='running', branch='baseline', operation='train', training_pid=pid,
                   production=False, adopted_existing_training=True)
            write(PRODUCTION / 'launch-status.json', dict(phase='investigation',
                  supervisor_pid=os.getpid(),training_pid=pid,investigation=str(OUT),
                  baseline_resume_after_investigation=True,time=time.time()))
            while alive(pid):
                time.sleep(5)
        verify_plain(io, torch)
        resume = OUT / 'baseline/train/checkpoints/last.pt'
        for branch, seed in (('baseline', SEEDS[0]), ('cross', SEEDS[1]),
                             ('mlp', SEEDS[1]), ('baseline', SEEDS[1])):
            run(env, branch, f'seed-{seed}')
        run(env, 'cross', f'seed-{SEEDS[1]}-memory-bos')
        compare()
    except Exception as exception:
        error = repr(exception)
        record(phase='investigation_failed', error=error, baseline_recovery_pending=True)
    finally:
        # If continuation failed, recover its most recent verified full snapshot.
        candidate = OUT / 'baseline/train/checkpoints/last.pt'
        if candidate.is_file():
            local = io._checkpoint_upload_source(candidate.resolve())
            payload = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
            assert len(payload['optimizer']['state']) == 798
            resume, step = candidate, payload['global_step']
        else:
            payload = torch.load(resume, map_location='cpu', mmap=True, weights_only=False)
            step = payload['global_step']
        config = yaml.safe_load((BASE / 'production-train.yaml').read_text())
        config['options'].update(resume_checkpoint=str(resume), max_optimizer_steps=0)
        (BASE / 'production-train.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copyfile(BASE / 'production-train.yaml', OUT / 'production-train.yaml')
        record(phase='resuming_production', resume_step=step, investigation_error=error)
        run(env, 'baseline', 'train', resume_step=step, production=True)


if __name__ == '__main__':
    if '--prepare-only' in sys.argv:
        prepare()
    else:
        main()
