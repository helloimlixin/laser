"""Prepare and supervise full-data continuous-compound ImageNet training."""
import concurrent.futures
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
from omegaconf import OmegaConf

from src.training.continuous_compound import file_digest, write_json


def index_and_copy_training_archive(cfg, base, output):
    """Copy each nested class once to NVMe and index JPEG offsets in memory."""
    source = Path(cfg.data.train_archive).resolve()
    destination = base / 'data/ILSVRC2012_img_train.tar'
    index_path = base / 'data/train-index.npz'
    receipt = base / 'data/verified.json'
    if receipt.is_file() and index_path.is_file() and destination.is_file():
        proof = json.loads(receipt.read_text())
        if proof['source'] == str(source) and proof['bytes'] == destination.stat().st_size:
            return proof
        raise ValueError('Existing training archive has incompatible provenance')
    stat = source.stat()
    class_index = Path(cfg.data.class_index)
    if not class_index.is_file():
        class_index = output / 'archive-class-index.json'
    if not class_index.is_file():
        with tarfile.open(source, 'r:') as archive:
            members = sorted([m for m in archive if m.isfile() and m.name.endswith('.tar')], key=lambda m: m.name)
        write_json(class_index, dict(archive=dict(path=str(source), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
            classes=[dict(name=m.name, size=m.size, offset_data=m.offset_data) for m in members]))
    classes = json.loads(class_index.read_text())
    if classes['archive'] != dict(path=str(source), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns):
        raise ValueError('Training archive differs from its verified class offsets')
    entries = classes['classes']
    if len(entries) != 1000:
        raise ValueError('Full training requires all 1000 ImageNet classes')
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_fd = os.open(source, os.O_RDONLY)
    target_fd = os.open(destination, os.O_RDWR | os.O_CREAT, 0o600)
    os.ftruncate(target_fd, stat.st_size)

    def read_exact(offset, count, fd=source_fd):
        parts = []
        while count:
            # Bound FUSE read buffers; many simultaneous whole-class reads
            # otherwise exhaust the object mount's shared readahead cache.
            part = os.pread(fd, min(count, 8 * 1024**2), offset)
            if not part:
                raise OSError('Incomplete ImageNet archive')
            parts.append(part); offset += len(part); count -= len(part)
        return b''.join(parts)

    def write_exact(offset, data):
        view = memoryview(data)
        while view:
            written = os.pwrite(target_fd, view, offset)
            if not written:
                raise OSError('Incomplete local archive copy')
            view = view[written:]; offset += written

    def one_class(pair):
        label, entry = pair
        first, size = int(entry['offset_data']), int(entry['size'])
        # A class header is committed only after its complete payload. Reuse
        # committed local classes after an interrupted preparation.
        header = os.pread(target_fd, 512, first - 512)
        committed = header[:100].split(b'\0')[0] == entry['name'].encode()
        if committed:
            data = read_exact(first, size, target_fd)
        else:
            # Each reader needs its own FUSE handle so interleaved classes do
            # not defeat the mount's per-handle sequential readahead.
            class_fd = os.open(source, os.O_RDONLY)
            try:
                data = read_exact(first, size, class_fd)
                write_exact(first, data)
                write_exact(first - 512, read_exact(first - 512, 512, class_fd))
            finally:
                os.close(class_fd)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as inner:
            images = sorted([m for m in inner if m.isfile() and m.name.endswith('.JPEG')], key=lambda m: m.name)
        if not images or any(not m.name.startswith(entry['name'].removesuffix('.tar') + '_') for m in images):
            raise ValueError('ImageNet class archive contains unexpected image identities')
        return label, [(first + m.offset_data, m.size, label) for m in images]

    started = time.monotonic()
    results = [None] * 1000
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg.training.index_workers)) as pool:
            futures = [pool.submit(one_class, pair) for pair in enumerate(entries)]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
                label, rows = future.result(); results[label] = rows
                if completed % 10 == 0 or completed == 1:
                    status = dict(phase='preparing_full_training_data', classes=completed, total_classes=1000,
                        images=sum(len(r) for r in results if r is not None), seconds=time.monotonic()-started)
                    write_json(output / 'status.json', status)
                    print(json.dumps(status), flush=True)
        os.fsync(target_fd)
    finally:
        os.close(source_fd); os.close(target_fd)
    rows = np.asarray([row for group in results for row in group], dtype=np.int64)
    if len(rows) != 1281167:
        raise ValueError(f'Expected 1281167 training images; found {len(rows)}')
    np.savez(index_path, offsets=rows[:, 0], sizes=rows[:, 1].astype(np.int32),
        labels=rows[:, 2].astype(np.int16), classes=np.asarray([r['name'].removesuffix('.tar') for r in entries]))
    write_json(output / 'status.json', dict(phase='verifying_full_training_archive', images=len(rows)))
    with destination.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'md5').hexdigest()
    if digest != cfg.training.archive_md5:
        raise ValueError('Local full ImageNet archive MD5 verification failed')
    proof = dict(source=str(source), bytes=stat.st_size, images=len(rows), classes=1000, md5=digest,
        sorted_wnid_labels=True, jpeg_offsets_verified=True, elapsed_seconds=time.monotonic()-started)
    write_json(receipt, proof); write_json(output / 'data-verification.json', proof)
    return proof


def prepare(cfg, base, output):
    from runtime.rq_imagenet_entry import freeze_shared_runtime
    from src.rq_upstream import RQTransformerConfig
    root = Path(__file__).resolve().parents[2]
    inputs = base / 'inputs'; inputs.mkdir(parents=True, exist_ok=True)
    source = Path(cfg.tokenizer.checkpoint)
    target = inputs / 'stage1-model.pt'
    if not target.exists():
        shutil.copyfile(source, target)
    if file_digest(target) != cfg.tokenizer.sha256:
        raise ValueError('Frozen original LASER checkpoint SHA256 mismatch')
    reference = inputs / 'imagenet_256_train.npz'
    if not reference.exists():
        shutil.copyfile(cfg.training.reference_stats, reference)
    if file_digest(reference) != cfg.training.reference_sha256:
        raise ValueError('ImageNet FID reference SHA256 mismatch')
    freeze_shared_runtime(base, root, include_stochastic_teacher=False)
    shutil.copytree(root / 'third_party/rq-vae-transformer', base / 'upstream', dirs_exist_ok=True,
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
    training = OmegaConf.load(cfg.training.rq_config)
    training.arch = RQTransformerConfig.create(cfg.arch)
    training.dataset.vocab_size = int(cfg.tokenizer.num_atoms)
    training.optimizer.init_lr = float(cfg.optimizer.lr)
    training.optimizer.weight_decay = float(cfg.optimizer.weight_decay)
    training.optimizer.betas = cfg.optimizer.betas
    training.optimizer.max_gn = float(cfg.optimizer.max_grad_norm)
    training.experiment.epochs = int(cfg.epochs)
    training.experiment.amp = True
    training.experiment.test_freq = int(cfg.evaluation.every_epochs)
    OmegaConf.save(training, inputs / 'stage2-training-config.yaml', resolve=True)
    OmegaConf.save(OmegaConf.create(dict(arch=OmegaConf.to_container(training.arch, resolve=True))),
                   inputs / 'stage2-archive-config.yaml')
    OmegaConf.save(cfg, inputs / 'recipe.yaml', resolve=True)
    (base / 'wandb').mkdir(exist_ok=True)
    (base / 'torch-cache').mkdir(exist_ok=True)
    (base / 'worker.py').write_text(
        'import os, sys\nfrom pathlib import Path\n'
        'base = Path(__file__).resolve().parent\n'
        "os.environ.setdefault('LASER_RQ_SAFE_CHECKPOINTS', '1')\n"
        "os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')\n"
        "sys.path[:0] = [str(base/'source'), str(base/'upstream'), str(base)]\n"
        'from src.training.rq_imagenet_shared import main\n'
        f"main({str(cfg.get('recipe_kind', 'laser_continuous_compound'))!r}, base)\n")
    index_and_copy_training_archive(cfg, base, output)


def run(cfg):
    if int(cfg.global_batch) != 2048 or int(cfg.training.devices) != 4 or cfg.data.synthetic:
        raise ValueError('This matched full ImageNet recipe requires four GPUs, batch 2048, and real images')
    base, output = Path(cfg.runtime_base).resolve(), Path(cfg.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True); base.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (base / 'supervisor.lock').open('w')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(output / 'launch.json', dict(pid=os.getpid(), run_id=str(cfg.run_id),
        epochs=int(cfg.epochs), devices=4, global_batch=2048, fresh_stage2=cfg.resume is None,
        command=[sys.executable, *sys.argv], runtime_base=str(base), time=time.time()))
    OmegaConf.save(cfg, output / 'recipe.yaml', resolve=True)
    prepare(cfg, base, output)
    env = dict(os.environ, TORCH_HOME=str(base/'torch-cache'), OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NCCL_NVLS_ENABLE='0',
        LASER_WANDB_CHECKPOINT_UPLOADS='0', LASER_CHECKPOINT_STAGING_DIR=str(base/'checkpoint-staging'),
        LASER_CHECKPOINT_UPLOAD_CACHE_DIR=str(base/'checkpoint-cache'), LASER_CHECKPOINT_IMMUTABLE_FILES='1')
    resume = cfg.resume
    child = None

    def stop(sig, frame):
        if child is not None:
            child.send_signal(sig)
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    for attempt in range(int(cfg.training.max_restarts) + 1):
        command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4',
            str(base/'worker.py'), '--epochs', str(cfg.epochs), '--microbatch', str(cfg.microbatch)]
        if resume:
            command += ['--resume', str(resume)]
        write_json(output / 'status.json', dict(phase='starting_full_training', attempt=attempt,
            epochs=int(cfg.epochs), devices=4, resume=str(resume) if resume else None))
        child = subprocess.Popen(command, env=env, cwd=Path(__file__).resolve().parents[2])
        while child.poll() is None:
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                progress = output / 'train/progress-rank0.json'
                if progress.is_file():
                    write_json(output/'status.json', dict(phase='training', supervisor_pid=os.getpid(),
                        worker_launcher_pid=child.pid, progress=json.loads(progress.read_text()), target_epochs=int(cfg.epochs)))
        if child.returncode == 0:
            write_json(output/'status.json', dict(phase='completed', epochs=int(cfg.epochs)))
            return
        checkpoint = output / 'checkpoints/last.pt'
        if not checkpoint.is_file() or attempt == int(cfg.training.max_restarts):
            write_json(output/'status.json', dict(phase='failed', exit_code=child.returncode, attempt=attempt))
            raise RuntimeError(f'Training exited with code {child.returncode}; see training.log')
        resume = checkpoint
