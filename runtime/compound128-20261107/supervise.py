"""Gate fresh training on full rFID, profile both microbatches, and supervise recovery."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/tmp/laser-compound128-20261107')
BASE = ROOT / 'frozen'
OUT = ROOT / 'production-nvme'
RUNTIME = Path('/workspace/Projects/laser/runtime/compound128-20261107')
RUN = 'imagenet-laser-compound128-native-scratch100-4a100-20261107'
CHILD = None


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(data, indent=2, default=str)+'\n')
    temp.replace(path)


def stop(sig, _):
    if CHILD is not None and CHILD.poll() is None:
        try:
            os.killpg(CHILD.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    write(ROOT/'status.json', dict(phase='interrupted', signal=sig, time=time.time()))
    raise SystemExit(128+sig)


def environment():
    env = dict(os.environ)
    for k in ['WANDB_SERVICE', '_WANDB_SERVICE']:
        env.pop(k, None)
    env.update(WANDB_API_KEY=Path('/root/.config/laser/wandb.key').read_text().strip(),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_MODE='online', WANDB_CONSOLE='off',
        WANDB_DISABLE_GIT='true', WANDB_CACHE_DIR=str(ROOT/'wandb-cache'),
        LASER_STAGE2_OUTPUT=str(OUT), CUDA_VISIBLE_DEVICES='0,1,2,3', PYTHONUNBUFFERED='1',
        TORCH_HOME=str(ROOT/'torch-cache'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
        LASER_FUSED_ADAMW='1', LASER_TRAIN_SDPA='1', LASER_SDPA_MIN_SEQ='16',
        LASER_COMPILE_NATIVE_LOSS='0', LASER_EXACT_BEAM_KERNELS='0',
        TORCHINDUCTOR_CACHE_DIR=str(ROOT/'inductor-cache'), TORCHINDUCTOR_COMPILE_THREADS='2',
        NCCL_NVLS_ENABLE='0', PYTORCH_ALLOC_CONF='expandable_segments:True')
    return env


def command(microbatch, resume=None, smoke=0):
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
        str(BASE/'train.py'), '--microbatch', str(microbatch), '--epochs', '100']
    if resume:
        cmd += ['--resume', str(resume)]
    if smoke:
        cmd += ['--smoke-updates', str(smoke)]
    return cmd


def child(cmd, env, phase, folder):
    global CHILD
    started = time.time()
    with (OUT/(phase+'.log')).open('ab', buffering=0) as log:
        CHILD = subprocess.Popen(cmd, env=env, cwd=BASE, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while CHILD.poll() is None:
            state = dict(phase=phase, supervisor_pid=os.getpid(), child_pid=CHILD.pid,
                progress=read(folder/'progress-rank0.json'), checkpoint=read(OUT/'checkpoint.json'),
                generation=read(OUT/'generation-progress.json'), target_epochs=100, time=time.time())
            write(ROOT/'status.json', state)
            write(OUT/'status.json', state)
            paths = [OUT/(phase+'.log'), folder/'progress-rank0.json',
                     OUT/'generation-progress.json', OUT/'checkpoint-storage-wait.json']
            last = max([started] + [p.stat().st_mtime for p in paths if p.exists()])
            if time.time()-last > 1800:
                os.killpg(CHILD.pid, signal.SIGTERM)
                try:
                    CHILD.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(CHILD.pid, signal.SIGKILL)
            try:
                CHILD.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    return CHILD.returncode


def profile_microbatch(microbatch):
    folder = ROOT/f'smoke-micro{microbatch}'
    code = child(command(microbatch, smoke=3), dict(environment(), LASER_PRIVATE_SMOKE_DIR=str(folder)),
        f'smoke-micro{microbatch}', folder)
    if code:
        log = (OUT/f'smoke-micro{microbatch}.log').read_text()
        if 'out of memory' in log.lower():
            return dict(microbatch=microbatch, passed=False, reason='CUDA out of memory')
        raise RuntimeError(f'Microbatch {microbatch} smoke failed; inspect its log')
    proof = read(folder/'verification.json')
    assert proof['passed'] and proof['frozen_tokenizer_unchanged']
    assert proof['all_adam_steps'] == 3 and proof['optimizer_state_count'] == 837
    for rank in range(4):
        initial = read(folder/f'startup-rank{rank}.json')
        assert initial['fresh_stage2'] and initial['fresh_optimizer']
        assert initial['global_step'] == 0 and initial['global_batch'] == 2048
        assert initial['coefficient_bins'] == 128 and initial['num_atoms'] == 16384
        assert initial['optimizer_state_count_at_start'] == 0 and initial['scheduler_horizon'] == 62600
        targets = read(folder/f'targets-rank{rank}.json')
        assert targets['training_and_evaluation_codec_identical'] and not targets['coefficient_noise']
        generation = read(folder/f'generation-interface-rank{rank}.json')
        assert generation['passed'] and not generation['image_encoder_used']
    rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    speed = sum(r['train/images_per_second'] for r in rows[1:])/len(rows[1:])
    progress = read(folder/'progress-rank0.json')
    for p in (OUT/'smoke/checkpoints').glob('*.pt'):
        p.unlink()
    return dict(microbatch=microbatch, passed=True, images_per_second=speed,
        peak_allocated_gib=progress['peak_allocated_gib'], optimizer_state_count=837,
        verification=proof, discarded_smoke_weights=True)


def latest():
    import torch
    for p in sorted((OUT/'persistent-train/checkpoints').glob('step-*.pt'), reverse=True):
        try:
            state = torch.load(p, map_location='cpu', weights_only=False, mmap=True)
            assert state['run_id'] == RUN and len(state['optimizer']['state']) == 837
            step = int(state['global_step'])
            assert all(int(s['step']) == step for s in state['optimizer']['state'].values())
            del state
            return p, step
        except Exception as error:
            write(OUT/'rejected-checkpoints'/f'{p.stem}.json', dict(error=str(error)))
    return None, 0


def main():
    lock = (ROOT/'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop)
    waiting_since = time.time()
    while not (ROOT/'tokenizer-verification.json').exists():
        write(ROOT/'status.json', dict(phase='reconstruction_validation', supervisor_pid=os.getpid(),
            progress=read(ROOT/'quality/progress.json'), time=time.time()))
        if time.time()-waiting_since > 1200:
            raise RuntimeError('Full reconstruction validation did not finish within twenty minutes')
        time.sleep(5)
    quality = read(ROOT/'tokenizer-verification.json')
    assert quality['passed'] and quality['measured_validation_rfid'] < 4.73
    assert quality['training_and_evaluation_codec_identical']
    assert read(ROOT/'cpu-verification.json')['passed']
    subprocess.run([sys.executable, '-u', str(RUNTIME/'publish_quality.py')], check=True,
        stdout=(ROOT/'quality/publish.log').open('ab'), stderr=subprocess.STDOUT)
    profile_path = ROOT/'throughput-profile.json'
    if not profile_path.exists():
        results = [profile_microbatch(micro) for micro in [128, 64]]
        eligible = [r for r in results if r['passed']]
        assert eligible, 'Neither microbatch passed the real-data training and generation smoke tests'
        selected = max(eligible, key=lambda row:row['images_per_second'])
        write(profile_path, dict(results=results, selected_microbatch=selected['microbatch'],
            global_batch=2048, world_size=4, smoke_checkpoints_discarded=True))
    microbatch = read(profile_path)['selected_microbatch']
    import shutil
    from omegaconf import OmegaConf
    recipe = OmegaConf.load(BASE/'inputs/recipe.yaml')
    recipe.microbatch = microbatch
    OmegaConf.save(recipe, BASE/'inputs/recipe.yaml')
    shutil.copyfile(BASE/'inputs/recipe.yaml',
        Path('/workspace/Projects/laser/configs/stage2/imagenet-laser-compound128-native-scratch100-4a100-20261107.yaml'))
    for name in ['cpu-verification.json', 'tokenizer-verification.json', 'throughput-profile.json']:
        shutil.copyfile(ROOT/name, BASE/'inputs'/name)
    with (OUT/'cloud-worker.log').open('ab', buffering=0) as log:
        uploader = subprocess.Popen([sys.executable, '-u', str(RUNTIME/'cloud_sync.py')],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    failures = 0
    attempt = 0
    while True:
        source, step = latest()
        cmd = command(microbatch, resume=source)
        write(OUT/'launch.json', dict(run_id=RUN, command=cmd, target_epochs=100, target_steps=62600,
            source_step=step, fresh=source is None, global_batch=2048, world_size=4,
            microbatch=microbatch, accumulation_steps=2048//(4*microbatch),
            coefficient_bins=128, atoms=16384, measured_tokenizer_rfid=quality['measured_validation_rfid'],
            url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+RUN,
            uploader_pid=uploader.pid, attempt=attempt, time=time.time()))
        code = child(cmd, environment(), 'training', OUT/'train')
        completed = read(OUT/'completed.json')
        if code == 0 and completed and completed['global_step'] == 62600:
            write(ROOT/'status.json', dict(phase='completed', result=completed, time=time.time()))
            return
        progress = read(OUT/'train/progress-rank0.json') or {}
        failures = 0 if progress.get('train/global_step', step)>step else failures+1
        write(OUT/'last-failure.json', dict(returncode=code, attempt=attempt, progress=progress))
        if failures >= 3:
            raise RuntimeError('Three consecutive training startup failures')
        attempt += 1
        time.sleep(15*max(1,failures))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write(ROOT/'failure.json', dict(error=str(error), time=time.time()))
        raise
