"""Hardware screening, codec verification, and supervised fresh compound128 training."""
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

from omegaconf import OmegaConf

from src.training.continuous_compound_train import prepare
from src.training.continuous_compound import write_json


def run(cfg):
    if cfg.resume is not None or int(cfg.global_batch) != 2048 or int(cfg.training.devices) != 4:
        raise ValueError('This launcher starts a fresh four-GPU experiment with global batch 2048')
    base, output = Path(cfg.runtime_base).resolve(), Path(cfg.output_dir).resolve()
    base.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    lock = (base / 'supervisor.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / 'checkpoint.json').is_file():
        raise FileExistsError('Production checkpoint exists; use the existing supervisor for recovery')
    child = None
    benchmark_children = []

    def stop(sig, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        for process in benchmark_children:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
        write_json(output / 'status.json', dict(phase='interrupted', signal=sig, time=time.time()))
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    write_json(output / 'launch.json', dict(pid=os.getpid(), run_id=str(cfg.run_id),
        fresh_stage2=True, fresh_optimizer=True, devices=4, global_batch=2048,
        runtime_base=str(base), command=[sys.executable, *sys.argv], time=time.time()))
    write_json(output / 'status.json', dict(phase='preparing', supervisor_pid=os.getpid()))
    previous = Path(cfg.data.reuse_verified_data).resolve()
    proof = json.loads((previous / 'verified.json').read_text())
    if proof['md5'] != cfg.training.archive_md5 or proof['images'] != 1281167:
        raise ValueError('Reused ImageNet archive lacks the expected full-corpus verification')
    if not (base / 'data').exists():
        (base / 'data').symlink_to(previous, target_is_directory=True)
    old_torch = previous.parent / 'torch-cache'
    if not (base / 'torch-cache').exists():
        (base / 'torch-cache').symlink_to(old_torch, target_is_directory=True)
    prepared = base / 'prepared.json'
    if not prepared.is_file():
        prepare(cfg, base, output)
        write_json(prepared, dict(stage1_sha256=str(cfg.tokenizer.sha256),
            reference_sha256=str(cfg.training.reference_sha256)))
    else:
        from runtime.rq_imagenet_entry import freeze_shared_runtime
        proof = json.loads(prepared.read_text())
        if proof['stage1_sha256'] != cfg.tokenizer.sha256 or proof['reference_sha256'] != cfg.training.reference_sha256:
            raise ValueError('Prepared inputs belong to another tokenizer or metric reference')
        freeze_shared_runtime(base, Path(__file__).resolve().parents[2], include_stochastic_teacher=False)
    # Explicitly replace stale loss metadata inherited from the old 512-bin config.
    config = OmegaConf.load(base / 'inputs/stage2-training-config.yaml')
    config.loss = dict(type=str(cfg.objective), atom_ce_weight=float(cfg.atom_ce_weight),
        coefficient_ce_weight=1., coefficient_bins=128, stochastic_codes=False,
        soft_targets=False, geometry_weight=0.)
    config.experiment.save_ckpt_freq = 1
    config.experiment.batch_size = int(cfg.microbatch)
    OmegaConf.save(config, base / 'inputs/stage2-training-config.yaml')
    root = Path(__file__).resolve().parents[2]
    probe = base / 'probe.py'
    probe.write_text(
        'import os,sys\nfrom pathlib import Path\nbase=Path(__file__).resolve().parent\n'
        "os.environ['RQ_UPSTREAM_ROOT']=str(base/'upstream')\n"
        "sys.path[:0]=[str(base/'source'),str(base/'upstream'),str(base)]\n"
        'from src.training.compound128_probe import main\nmain(base)\n')
    shutil.copyfile(root / 'src/training/compound128_probe.py', base / 'source/src/training/compound128_probe.py')
    env = dict(os.environ, TORCH_HOME=str(base / 'torch-cache'), PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NCCL_NVLS_ENABLE='0',
        LASER_WANDB_CHECKPOINT_UPLOADS='0', LASER_CHECKPOINT_STAGING_DIR=str(base / 'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base / 'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1',
        PYTORCH_ALLOC_CONF='expandable_segments:True')

    def execute(command, phase, extra=None):
        nonlocal child
        with (output / f'{phase}.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, env=dict(env, **(extra or {})), cwd=root,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            while child.poll() is None:
                state = dict(phase=phase, supervisor_pid=os.getpid(), child_pid=child.pid, time=time.time())
                for name, path in [('progress', output / 'train/progress-rank0.json'),
                                   ('generation', output / 'generation-progress.json'),
                                   ('codec', base / 'quality/progress.json')]:
                    if path.is_file():
                        try: state[name] = json.loads(path.read_text())
                        except json.JSONDecodeError: pass
                write_json(output / 'status.json', state)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: pass
        return child.returncode

    # Disposable candidates run independently on all four GPUs; no candidate weights are reused.
    profile_path = base / 'throughput-profile.json'
    if not profile_path.is_file():
        processes = []
        for gpu, candidate in enumerate(cfg.execution.benchmark_candidates):
            log = (output / f'benchmark-gpu{gpu}.log').open('ab', buffering=0)
            command = [sys.executable, '-u', str(probe), 'benchmark', '--gpu', str(gpu),
                '--microbatch', str(candidate.microbatch), '--attention', str(candidate.attention_backend)]
            process = subprocess.Popen(command, env=env, cwd=root, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            processes.append((process, log))
            benchmark_children.append(process)
        while any(p.poll() is None for p, _ in processes):
            write_json(output / 'status.json', dict(phase='benchmarking', supervisor_pid=os.getpid(),
                candidates=[dict(pid=p.pid, exit_code=p.poll()) for p, _ in processes], time=time.time()))
            time.sleep(5)
        for p, log in processes:
            log.close()
            if p.returncode:
                raise RuntimeError('Hardware benchmark failed unexpectedly; inspect benchmark logs')
        candidates = [json.loads((base / f'benchmark-gpu{gpu}.json').read_text()) for gpu in range(4)]
        write_json(profile_path, dict(candidates=candidates, benchmark_weights_discarded=True))
    candidates = json.loads(profile_path.read_text())['candidates']
    candidates = sorted([c for c in candidates if c['passed']],
        key=lambda c: c['images_per_second_per_gpu'], reverse=True)
    if not candidates:
        raise RuntimeError('No hardware candidate passed')
    distributed = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4']
    selected = None
    for candidate in candidates:
        cfg.microbatch = int(candidate['microbatch'])
        cfg.execution.attention_backend = candidate['attention_backend']
        OmegaConf.save(cfg, base / 'inputs/recipe.yaml', resolve=True)
        smoke = base / f"ddp-micro{cfg.microbatch}-{cfg.execution.attention_backend}"
        command = [*distributed, str(base / 'worker.py'), '--epochs', str(cfg.epochs),
            '--microbatch', str(cfg.microbatch), '--smoke-updates', str(cfg.execution.ddp_preflight_updates),
            '--smoke-split', 'train']
        proof_path = smoke / 'verification.json'
        code = (0 if proof_path.is_file() else
            execute(command, 'distributed-preflight', dict(LASER_PRIVATE_SMOKE_DIR=str(smoke))))
        if code:
            if 'out of memory' in (output / 'distributed-preflight.log').read_text().lower():
                continue
            raise RuntimeError('Distributed preflight failed; see distributed-preflight.log')
        verification = json.loads(proof_path.read_text())
        if not verification['passed'] or not verification['frozen_tokenizer_unchanged']:
            raise RuntimeError('Distributed preflight verification failed')
        for rank in range(4):
            startup = json.loads((smoke / f'startup-rank{rank}.json').read_text())
            assert startup['fresh_stage2'] and startup['fresh_optimizer'] and startup['global_step'] == 0
            assert startup['coefficient_bins'] == 128 and startup['global_batch'] == 2048
            assert startup['microbatch'] == int(cfg.microbatch)
            assert startup['attention_backend'] == str(cfg.execution.attention_backend)
            assert json.loads((smoke / f'targets-rank{rank}.json').read_text())['training_and_evaluation_codec_identical']
        selected = dict(candidate, ddp_verification=verification,
            ddp_progress=json.loads((smoke / 'progress-rank0.json').read_text()), discarded_smoke_weights=True)
        break
    if selected is None:
        raise RuntimeError('No candidate fit the distributed preflight')
    write_json(output / 'throughput-selection.json', selected)
    OmegaConf.save(cfg, output / 'recipe.yaml', resolve=True)
    OmegaConf.save(cfg, base / 'inputs/recipe.yaml', resolve=True)
    # Use the exact selected encoder/quantizer to verify reconstruction before production.
    quality = base / 'tokenizer-verification.json'
    if not quality.is_file():
        code = execute([*distributed, str(probe), 'quality'], 'reconstruction-validation')
        if code:
            raise RuntimeError('128-bin reconstruction verification failed')
    proof = json.loads(quality.read_text())
    if not proof['passed']:
        raise RuntimeError('128-bin codec failed the reconstruction gate')
    shutil.copyfile(quality, output / 'tokenizer-verification.json')
    reproducibility = output / 'reproducibility'; reproducibility.mkdir(exist_ok=True)
    for name in ['recipe.yaml', 'stage2-training-config.yaml', 'stage2-archive-config.yaml']:
        shutil.copyfile(base / 'inputs' / name, reproducibility / name)
    manifest = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ['source', 'runtime', 'upstream'] for p in (base / folder).rglob('*.py')}
    write_json(reproducibility / 'source-manifest.json', dict(files=manifest))
    write_json(output / 'throughput-profile.json', json.loads(profile_path.read_text()))
    monitor_env = dict(env, PYTHONPATH='/tmp/laser-wandb-logging-deps:' + env.get('PYTHONPATH', ''))
    with (output / 'wandb-monitor.log').open('ab', buffering=0) as log:
        monitor = subprocess.Popen([sys.executable, '-u', '-m', 'src.training.wandb_training_monitor',
            '--output', str(output), '--credentials', str(cfg.logging.credentials_file)], env=monitor_env,
            cwd=root, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    (output / 'wandb-monitor.pid').write_text(str(monitor.pid) + '\n')
    resume = None
    for attempt in range(int(cfg.training.max_restarts) + 1):
        command = [*distributed, str(base / 'worker.py'), '--epochs', str(cfg.epochs),
                   '--microbatch', str(cfg.microbatch)]
        if resume is not None:
            command += ['--resume', str(resume)]
        code = execute(command, 'training')
        if code == 0:
            write_json(output / 'status.json', dict(phase='completed', time=time.time()))
            return
        resume = output / 'checkpoints/last.pt'
        if attempt == int(cfg.training.max_restarts) or not resume.is_file():
            write_json(output / 'status.json', dict(phase='failed', exit_code=code, time=time.time()))
            raise RuntimeError('Training failed; see training.log')
