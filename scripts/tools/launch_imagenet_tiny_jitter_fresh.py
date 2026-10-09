"""Prepare, check, and launch fresh stage 2 after stopping the owned old run."""
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
OLD_BASE = Path('/tmp/laser-imagenet-low-noise-20261008')
OLD_OUT = ROOT/'outputs/imagenet-rfid421-low-noise-8h100-20261008'
PRODUCTION = ROOT/'outputs/imagenet-rfid421-classcond-8h100-20261007'
RUN_ID = 'imagenet-rfid421-tiny-jitter-fresh-8h100-20261008'
BASE = Path('/tmp/laser-'+RUN_ID)
OUT = ROOT/'outputs'/RUN_ID
ASSETS = Path('/tmp/laser-imagenet-classcond-20261007')
CACHE = Path('/tmp/laser-imagenet-pair-memory-20261007/checkpoint-cache')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def record(**value):
    value.update(time=time.time(), supervisor_pid=os.getpid())
    write(OUT/'launch-status.json', value)
    with (OUT/'launch-history.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')
    print(json.dumps(value), flush=True)


def environment():
    return dict(os.environ,
        WANDB_API_KEY=Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip(),
        WANDB_DIR=str(BASE/'wandb'), WANDB_CACHE_DIR=str(BASE/'wandb-cache'),
        WANDB_DATA_DIR=str(BASE/'wandb-data'), WANDB_ARTIFACT_DIR=str(BASE/'wandb-artifacts'),
        TORCH_HOME=str(ASSETS/'torch-cache'), TORCHINDUCTOR_CACHE_DIR=str(ASSETS/'inductor-cache'),
        TORCHINDUCTOR_COMPILE_THREADS='4', MPLCONFIGDIR=str(BASE/'matplotlib'),
        PYTHONPATH=str(BASE/'source')+':'+str(BASE/'source/runtime'),
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', LASER_ACCUMULATION='4',
        LASER_PAIR_MEMORY_BASE=str(BASE), LASER_PAIR_MEMORY_OUTPUT=str(OUT),
        LASER_CHECKPOINT_STAGING_DIR=str(BASE/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(CACHE), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1', NCCL_NVLS_ENABLE='0')


def prepare():
    import yaml
    calibration = json.loads((OUT/'noise-calibration.json').read_text())
    assert calibration['passed']
    BASE.mkdir(parents=True, exist_ok=True)
    assert not (BASE/'source').exists(), 'Preparation is single-use'
    shutil.copytree(OLD_BASE/'source', BASE/'source', copy_function=shutil.copyfile,
                    ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('wandb', 'checkpoint-staging', 'matplotlib'):
        (BASE/name).mkdir(exist_ok=True)
    for name in ('coefficient_jitter.py', 'generation_logging.py'):
        shutil.copyfile(ROOT/'src/training'/name, BASE/'source/src/training'/name)
        shutil.copyfile(ROOT/'src/training'/name, OUT/name)
    shutil.copyfile(ROOT/'scripts/tools/imagenet_tiny_jitter_entry.py', BASE/'entry.py')
    shutil.copyfile(BASE/'entry.py', OUT/'entry.py')
    config = yaml.safe_load((OLD_BASE/'train.yaml').read_text())
    width = calibration['bin_width_normalized']
    # Kept only as a legacy config field: the new CDF kernel uses sigma/cap,
    # bypasses the native temperature floor, and records its actual parameters.
    legacy_temperature = 2*(calibration['selected']['sigma_bins']*width)**2
    config['options'].update(output=str(OUT/'train'), checkpoint_dir=str(OUT/'train/checkpoints'),
        resume=False, resume_checkpoint=None, init_stage2_checkpoint=None, max_optimizer_steps=0,
        coeff_target_temperature=legacy_temperature, fid_seed=261001,
        wandb_id=RUN_ID, wandb_name='ImageNet rFID4.21 | fresh baseline, 1% bin jitter | 8 H100')
    text = yaml.safe_dump(config, sort_keys=False)
    for path in (BASE/'train.yaml', OUT/'train.yaml', ROOT/'configs/stage2'/f'{RUN_ID}.yaml'):
        path.write_text(text)
    for name in ('launch_imagenet_tiny_jitter_fresh.py', 'calibrate_imagenet_tiny_jitter.py'):
        shutil.copyfile(ROOT/'scripts/tools'/name, OUT/name)
    shutil.copyfile(ROOT/'scripts/tools/launch_imagenet_tiny_jitter_fresh.py', BASE/'launch.py')
    hashes = {}
    for relative in ('src/training/rqtransformer.py', 'src/training/k4_checkpoint_io.py',
                     'src/models/rqtransformer/transformers.py'):
        before = hashlib.sha256((OLD_BASE/'source'/relative).read_bytes()).hexdigest()
        after = hashlib.sha256((BASE/'source'/relative).read_bytes()).hexdigest()
        assert before == after
        hashes[relative] = after
    write(OUT/'source-manifest.json', dict(native_source_unchanged=True,
        source=str(OLD_BASE/'source'), sha256=hashes,
        entry_sha256=hashlib.sha256((BASE/'entry.py').read_bytes()).hexdigest(),
        helper_sha256={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
            for name in ('coefficient_jitter.py','generation_logging.py')}))
    write(OUT/'plan.json', dict(status='prepared', start_step=0, stage2_initialization='scratch',
        stage1_checkpoint=config['options']['checkpoint'], stage1_reconstruction_fid=4.21,
        epochs=100, world_size=8, global_batch=2048, architecture='plain compound pair autoregressive baseline',
        noise_distribution=calibration['distribution'], sigma_bins=0.01, hard_cap_bins=0.025,
        scheduler='fresh 100-epoch cosine', initial_lr=0.0005, scheduler_total_steps=62600,
        fid_every=2, first_fid_epoch=1, fid_samples=50000, fid_seed=261001,
        metric_keys=['eval/fid','eval/inception_score','eval/inception_score_std'],
        wandb_id=RUN_ID, previous_run='helloimlixin-rutgers/laser/'+OLD_OUT.name))
    record(phase='prepared')


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2] != 'Z'


def main():
    lock = (BASE/'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((OUT/'preflight-verification.json').read_text())['passed']
    assert not (OUT/'train/checkpoints/last.pt').exists(), 'Fresh output already has a checkpoint'
    previous = json.loads((PRODUCTION/'launch-status.json').read_text())
    assert previous['phase'] == 'running' and previous['wandb_id'] == OLD_OUT.name
    runner = previous['training_pid']
    assert alive(runner)
    workers = [int(x) for x in subprocess.check_output(['pgrep','-P',str(runner)],text=True).split()]
    assert len(workers) == 8
    pids = [previous['supervisor_pid'], runner, *workers]
    for pid in pids:
        assert str(OLD_BASE).encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    # Keep the last durable previous checkpoint and metrics for comparisons.
    previous_checkpoint = (OLD_OUT/'train/checkpoints/last.pt').resolve()
    assert previous_checkpoint.is_file() and previous_checkpoint.stat().st_size > 16_000_000_000
    write(OUT/'previous-run.json', dict(previous, last_checkpoint=str(previous_checkpoint),
        checkpoint_preserved=True, stage2_weights_or_optimizer_reused=False))
    record(phase='stopping_previous_baseline', stopped_pids=pids, fresh_start_step=0)
    for pid in pids[:2]:
        if alive(pid): os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+35
    while any(alive(pid) for pid in pids) and time.monotonic()<deadline:
        time.sleep(0.5)
    for pid in pids:
        if alive(pid): os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic()+10
    while any(alive(pid) for pid in pids) and time.monotonic()<deadline:
        time.sleep(0.5)
    assert not any(alive(pid) for pid in pids)
    write(OUT/'handoff-verification.json', dict(passed=True, stopped_pids=pids,
        previous_checkpoint=str(previous_checkpoint), start_step=0, fresh_stage2=True, time=time.time()))
    env = environment()
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=8',
               str(BASE/'entry.py'),'--config',str(BASE/'train.yaml')]
    log_path = OUT/'train.log'
    with log_path.open('a') as log:
        process = subprocess.Popen(command,cwd=BASE/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        record(phase='running', branch='baseline-tiny-jitter-fresh',training_pid=process.pid,
            start_step=0,log=str(log_path),wandb_id=RUN_ID)
        write(PRODUCTION/'launch-status.json',dict(phase='running',branch='baseline-tiny-jitter-fresh',
            supervisor_pid=os.getpid(),training_pid=process.pid,log=str(log_path),start_step=0,
            continuation_status=str(OUT/'launch-status.json'),wandb_id=RUN_ID,time=time.time()))
        result = process.wait()
    final = dict(phase='completed' if result == 0 else 'failed',branch='baseline-tiny-jitter-fresh',
        training_pid=process.pid,start_step=0,log=str(log_path),returncode=result,
        wandb_id=RUN_ID,time=time.time(),supervisor_pid=os.getpid())
    record(**{k:v for k,v in final.items() if k not in ('time','supervisor_pid')})
    write(PRODUCTION/'launch-status.json',final)
    raise SystemExit(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare_only else main()
