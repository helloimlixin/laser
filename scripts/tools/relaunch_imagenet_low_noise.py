"""Prepare and supervise a calibrated-noise continuation of the plain baseline."""
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

ROOT = Path('/workspace/Projects/laser')
OLD_BASE = Path('/tmp/laser-imagenet-pair-memory-investigation-20261007')
PRODUCTION = ROOT / 'outputs/imagenet-rfid421-classcond-8h100-20261007'
BASE = Path('/tmp/laser-imagenet-low-noise-20261008')
OUT = ROOT / 'outputs/imagenet-rfid421-low-noise-8h100-20261008'
CACHE = Path('/tmp/laser-imagenet-pair-memory-20261007/checkpoint-cache')
ASSETS = Path('/tmp/laser-imagenet-classcond-20261007')
RUN_ID = 'imagenet-rfid421-low-noise-8h100-20261008'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def record(**value):
    value.update(time=time.time(), supervisor_pid=os.getpid())
    write(OUT / 'launch-status.json', value)
    with (OUT / 'launch-history.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)


def prepare():
    import yaml
    calibration = json.loads((OUT / 'noise-calibration.json').read_text())
    assert calibration['passed']
    BASE.mkdir(parents=True, exist_ok=True)
    assert not (BASE / 'source').exists(), 'Preparation is single-use'
    shutil.copytree(OLD_BASE / 'source', BASE / 'source', copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('wandb', 'checkpoint-staging', 'matplotlib'):
        (BASE / name).mkdir(exist_ok=True)
    entry = (OLD_BASE / 'entry.py').read_text()
    old = "    from restore_evaluated_plain import restore_plain_best\n    return restore_plain_best(payload,path,OUT,record)"
    new = """    if isinstance(path, (str, Path)) and Path(path).resolve() == (BASE / 'anchor.pt').resolve():
        protected = ('state_dict', 'optimizer', 'scheduler', 'rng_state_by_rank', 'config')
        updated = dict(payload, best_fid=[], best_inception=[])
        assert all(updated[key] is payload[key] for key in protected)
        payload = updated
    return payload"""
    assert entry.count(old) == 1
    entry = entry.replace(old, new)
    old = "    from restore_evaluated_plain import log_restored_plain_metrics\n    log_restored_plain_metrics(wb)"
    new = """    calibration = json.loads((OUT / 'noise-calibration.json').read_text())
    wb.config.update({'source_run': 'helloimlixin-rutgers/laser/imagenet-rfid421-classcond-8h100-20261007',
        'source_checkpoint_step': ANCHOR_STEP, 'noise_calibration': calibration,
        'continuation_changed_option': 'coeff_target_temperature',
        'optimizer_scheduler_sampler_rng_restored': True}, allow_val_change=True)
    wb.summary.update({'continuation/source_step': ANCHOR_STEP,
        'noise/temperature': calibration['selected']['temperature'],
        'noise/expected_latent_energy_fraction': calibration['independent_verification']['expected_added_latent_energy_fraction'],
        'noise/k2_bin_sd_fraction': 0.25})
    for name in ('noise-calibration.json', 'plan.json', 'source-manifest.json', 'anchor-verification.json'):
        wb.save(str(OUT / name), base_path=str(OUT), policy='now')"""
    assert entry.count(old) == 1
    entry = entry.replace(old, new)
    # Check the actual probability distribution used by the first live batch.
    hook = """
native_coeff_ids = training.LaserAux.compound_coeff_ids
coefficient_probe_done = False
calibrated_temperature = json.loads((OUT / 'noise-calibration.json').read_text())['selected']['temperature']
def calibrated_coeff_ids(aux, coeffs, *, stochastic=True, temp=0.5, hard=False):
    global coefficient_probe_done
    assert not aux.soft_target_physical and temp == calibrated_temperature and stochastic and not hard
    ids, probabilities = native_coeff_ids(aux, coeffs, stochastic=stochastic, temp=temp, hard=hard)
    if not coefficient_probe_done:
        c = coeffs.reshape(-1, 4)[:32].float().cpu()
        actual = probabilities.reshape(-1, 4, aux.coeff_vocab_size)[:32].float().cpu()
        bins = aux.coeff_bins.detach().float().cpu()
        expected_probs = (-(c[..., None] - bins).square() / temp).softmax(-1)
        torch.testing.assert_close(actual, expected_probs, rtol=2e-5, atol=2e-7)
        rms = ((actual * (c[..., None] - bins).square()).sum(-1).mean(0)).sqrt()
        record('target-noise-rank' + os.environ['RANK'] + '.json', dict(passed=True,
            temperature=temp, target_space='normalized', native_probability_parity=True,
            noise_rms_bins_per_depth=(rms / (bins[1] - bins[0])).tolist()))
        coefficient_probe_done = True
    return ids, probabilities
training.LaserAux.compound_coeff_ids = calibrated_coeff_ids

native_restore_rng = training.restore_rank_rng_state
def verified_restore_rng(payload, device, seed):
    restored = native_restore_rng(payload, device, seed)
    assert restored is True
    expected = payload['rng_state_by_rank'][int(os.environ['RANK'])]
    assert torch.equal(torch.get_rng_state(), expected['torch_cpu'])
    assert torch.equal(torch.cuda.get_rng_state(device), expected['torch_cuda'])
    record('restored-rng-rank' + os.environ['RANK'] + '.json', dict(passed=True,
        cpu_rng_exact=True, cuda_rng_exact=True, global_step=payload['global_step']))
    return restored
training.restore_rank_rng_state = verified_restore_rng

"""
    assert entry.count("if __name__ == '__main__':") == 1
    entry = entry.replace("if __name__ == '__main__':", hook + "if __name__ == '__main__':")
    (BASE / 'entry.py').write_text(entry)
    shutil.copyfile(BASE / 'entry.py', OUT / 'entry.py')
    config = yaml.safe_load((OLD_BASE / 'production-train.yaml').read_text())
    config['options'].update(output=str(OUT / 'train'), checkpoint_dir=str(OUT / 'train/checkpoints'),
        resume_checkpoint=str(BASE / 'anchor.pt'), max_optimizer_steps=0,
        coeff_target_temperature=calibration['selected']['temperature'],
        wandb_id=RUN_ID, wandb_name='ImageNet rFID4.21 | baseline low noise, quarter K2 bin SD | 8 H100')
    text = yaml.safe_dump(config, sort_keys=False)
    for p in [BASE / 'train.yaml', OUT / 'train.yaml', ROOT / 'configs/stage2/imagenet-rfid421-low-noise-8h100-20261008.yaml']:
        p.write_text(text)
    copy = ROOT / 'scripts/tools/relaunch_imagenet_low_noise.py'
    shutil.copyfile(copy, BASE / copy.name)
    shutil.copyfile(copy, OUT / copy.name)
    shutil.copyfile(ROOT / 'scripts/tools/calibrate_imagenet_low_noise.py', OUT / 'calibrate_imagenet_low_noise.py')
    # All native sources are copied from the running baseline without edits.
    hashes = {}
    for relative in ['src/training/rqtransformer.py', 'src/training/k4_checkpoint_io.py',
                     'src/models/rqtransformer/transformers.py']:
        before = hashlib.sha256((OLD_BASE / 'source' / relative).read_bytes()).hexdigest()
        after = hashlib.sha256((BASE / 'source' / relative).read_bytes()).hexdigest()
        assert before == after
        hashes[relative] = after
    write(OUT / 'source-manifest.json', dict(native_source_unchanged=True,
        source=str(OLD_BASE / 'source'), sha256=hashes,
        entry_sha256=hashlib.sha256(entry.encode()).hexdigest()))
    write(OUT / 'plan.json', dict(status='prepared', anchor_step=None,
        updates=None, epochs=100, world_size=8, global_batch=2048,
        architecture='plain compound pair autoregressive baseline',
        changed_training_option='coeff_target_temperature',
        previous_temperature=0.5, temperature=calibration['selected']['temperature'],
        source_wandb_run='helloimlixin-rutgers/laser/imagenet-rfid421-classcond-8h100-20261007',
        wandb_id=RUN_ID, full_optimizer_scheduler_sampler_rng_resume=True))
    record(phase='prepared')


def alive(pid):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2] != 'Z'


def environment():
    return dict(os.environ,
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        WANDB_DIR=str(BASE / 'wandb'), WANDB_CACHE_DIR=str(BASE / 'wandb-cache'),
        WANDB_DATA_DIR=str(BASE / 'wandb-data'), WANDB_ARTIFACT_DIR=str(BASE / 'wandb-artifacts'),
        TORCH_HOME=str(ASSETS / 'torch-cache'), TORCHINDUCTOR_CACHE_DIR=str(ASSETS / 'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='4', MPLCONFIGDIR=str(BASE / 'matplotlib'),
        PYTHONPATH=str(BASE / 'source') + ':' + str(BASE / 'source/runtime'),
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', LASER_ACCUMULATION='4',
        LASER_PAIR_MEMORY_BASE=str(BASE), LASER_PAIR_MEMORY_OUTPUT=str(OUT),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(CACHE), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        LASER_BRANCH='baseline', LASER_PHASE='train', LASER_WANDB_RESUME='allow',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0')


def main():
    import torch
    torch.set_num_threads(4)
    sys.path[:0] = [str(BASE / 'source'), str(BASE / 'source/runtime')]
    from src.training import k4_checkpoint_io as io
    lock = (BASE / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((OUT / 'preflight-verification.json').read_text())['passed']
    env = environment()
    os.environ.update({k: v for k, v in env.items() if k.startswith('LASER_CHECKPOINT_')})
    status = json.loads((PRODUCTION / 'launch-status.json').read_text())
    assert status['phase'] == 'running' and alive(status['training_pid'])
    runner = status['training_pid']
    workers = [int(x) for x in subprocess.check_output(['pgrep', '-P', str(runner)], text=True).split()]
    assert len(workers) == 8
    pids = [status['supervisor_pid'], runner, *workers]
    for pid in pids:
        assert str(OLD_BASE).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    with Path(status['log']).open('rb') as stream:
        stream.seek(max(0,Path(status['log']).stat().st_size - 20000))
        lines = stream.read().decode(errors='replace').splitlines()
    logged_step = 0
    for line in reversed(lines):
        if line.startswith('{') and '"train/global_step"' in line:
            logged_step = json.loads(line)['train/global_step']
            break
    minimum_step = logged_step // 250 * 250
    while True:
        assert alive(runner), 'Previous baseline stopped while waiting for its checkpoint'
        persistent = (PRODUCTION / 'train/checkpoints/last.pt').resolve()
        local = io._checkpoint_upload_source(persistent)
        if local != persistent:
            candidate = torch.load(local, map_location='cpu', mmap=True, weights_only=False)
            ready = candidate['global_step'] >= minimum_step
            del candidate
            if ready: break
        record(phase='waiting_for_checkpoint', required_step=minimum_step,
               previous_training_pid=runner, baseline_continues=True)
        time.sleep(5)
    assert str(local).startswith('/tmp/')
    # Pin the local inode before stopping the producer or starting W&B uploads.
    os.link(local, BASE / 'anchor.pt')
    payload = torch.load(BASE / 'anchor.pt', map_location='cpu', mmap=True, weights_only=False)
    step = payload['global_step']
    assert len(payload['optimizer']['state']) == 798
    assert {int(x['step']) for x in payload['optimizer']['state'].values()} == {step}
    assert payload['scheduler']['last_epoch'] == step and payload['scheduler']['T_max'] == 62600
    assert payload['checkpoint_world_size'] == len(payload['rng_state_by_rank']) == 8
    assert payload['config']['coeff_target_temperature'] == 0.5
    assert not any(k.startswith(('site_pooling.', 'pair_memory_queries.')) for k in payload['state_dict'])
    (OUT / 'anchor.pt').symlink_to(persistent)
    checkpoint_dir = OUT / 'train/checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'last.pt').symlink_to(persistent)
    plan = json.loads((OUT / 'plan.json').read_text())
    plan.update(status='pinned', anchor_step=step, anchor_epoch=payload['epoch'],
        anchor_batch_idx=payload.get('batch_idx'), source_checkpoint=str(persistent))
    write(OUT / 'plan.json', plan)
    write(OUT / 'anchor-verification.json', dict(passed=True, step=step,
        epoch=payload['epoch'], batch_idx=payload.get('batch_idx'),
        optimizer_states=798, adam_age=step, scheduler=payload['scheduler'],
        rng_ranks=8, local_pinned_checkpoint=str(BASE / 'anchor.pt'),
        persistent_checkpoint=str(persistent), bytes=local.stat().st_size,
        migration_changes_only_best_metric_lists=True))
    del payload
    record(phase='stopping_previous_baseline', anchor_step=step, stopped_pids=pids)
    # Terminate only the verified supervisor/runner; torchrun owns worker shutdown.
    for pid in pids[:2]:
        if alive(pid): os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 35
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.5)
    for pid in pids:
        if alive(pid): os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.5)
    assert not any(alive(pid) for pid in pids), 'Old workers are still alive'
    write(OUT / 'handoff-verification.json', dict(passed=True, stopped_pids=pids,
        anchor_step=step, time=time.time()))
    env['LASER_RESUME_STEP'] = str(step)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
               str(BASE / 'entry.py'), '--config', str(BASE / 'train.yaml')]
    log_path = OUT / 'train.log'
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=BASE / 'source', env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        record(phase='running', branch='baseline-low-noise', training_pid=process.pid,
            resume_step=step, log=str(log_path), wandb_id=RUN_ID)
        write(PRODUCTION / 'launch-status.json', dict(phase='running', branch='baseline-low-noise',
            supervisor_pid=os.getpid(), training_pid=process.pid, log=str(log_path),
            resume_step=step, continuation_status=str(OUT / 'launch-status.json'),
            wandb_id=RUN_ID, time=time.time()))
        result = process.wait()
    final = dict(phase='completed' if result == 0 else 'failed', branch='baseline-low-noise',
        training_pid=process.pid, resume_step=step, log=str(log_path), returncode=result,
        wandb_id=RUN_ID, time=time.time(), supervisor_pid=os.getpid())
    record(**{k: v for k, v in final.items() if k not in ['supervisor_pid', 'time']})
    write(PRODUCTION / 'launch-status.json', final)
    raise SystemExit(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only: prepare()
    else: main()
