"""Local recovery snapshots with independent, verified cloud publication.

Only local serialization is on the training path. Upload jobs hold hard links
to immutable snapshots, so subsequent saves cannot change an in-flight upload.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import json
import os
import shutil
import time

import torch

# Import this dependency at startup, before any training is launched.
from .verified_wandb_checkpoint_upload import VerifiedCloudUpload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temporary, path)


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _link(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name('.' + target.name + '.link.tmp')
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, target)


class LocalFirstCheckpoints:
    def __init__(self, root, *, cloud_factory=VerifiedCloudUpload,
                 publish_every_epochs=5, final_step=12520):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / '.checkpoint-data'
        self.objects.mkdir(exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cloud-checkpoint')
        self.pending = []
        self.cloud_factory = cloud_factory
        self.cloud = None
        self.closed = False
        self.latest = {}
        self.publish_every_epochs = publish_every_epochs
        self.final_step = final_step
        receipt = self.root / 'latest.json'
        if receipt.exists():
            self.latest = json.loads(receipt.read_text())

    def _target(self, target):
        target = Path(target).absolute()
        if target.parent != self.root:
            raise ValueError(f'Checkpoint target must use local recovery directory: {target}')
        return target

    def save(self, payload, target):
        target = self._target(target)
        started = time.monotonic()
        step = int(payload.get('global_step', 0))
        serialized = self.objects / f'step-{step:09d}-{time.time_ns()}.pt'
        temporary = serialized.with_suffix('.tmp')
        try:
            # Freeze tensors before serialization; no reference to live weights
            # or optimizer buffers is passed to a background worker.
            torch.save(_cpu_snapshot(payload), temporary)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, serialized)
        finally:
            temporary.unlink(missing_ok=True)
        _link(serialized, target)
        self.latest = dict(path=str(serialized), target=str(target),
                           bytes=serialized.stat().st_size, epoch=payload.get('epoch'),
                           batch_idx=payload.get('batch_idx'), global_step=step,
                           fid=payload.get('fid'), saved_unix=time.time(),
                           rng_ranks=len(payload.get('rng_state_by_rank', [])))
        _atomic_json(self.root / 'latest.json', self.latest)
        print(json.dumps(dict(phase='local_checkpoint_saved', **self.latest,
                              seconds=time.monotonic()-started)), flush=True)
        # Best checkpoints and queued uploads retain separate inode links.
        # Keep two recent recovery objects as a fallback, without accumulating
        # a complete 17-GB serialization at every periodic save forever.
        objects = sorted(self.objects.glob('step-*.pt'))
        for old in objects[:-2]:
            old.unlink()

    def snapshot(self, source, target):
        _link(source, self._target(target))

    def upload(self, wb, *, last_checkpoint, best_fid, upload_dir, best_inception=()):
        if wb is None:
            return
        epoch = int(self.latest.get('epoch') or 0)
        step = int(self.latest.get('global_step') or 0)
        # Frequent local recovery saves do not each trigger a multi-GB upload.
        if self.latest.get('fid') is None and step != self.final_step:
            return
        slots = [('last.pt', Path(last_checkpoint))]
        slots += [(f'best-fid-{i:02d}.pt', Path(path)) for i, (_, path) in
                  enumerate(sorted(best_fid, key=lambda item: float(item[0]))[:3], 1)]
        slots += [(f'best-is-{i:02d}.pt', Path(path)) for i, (_, path) in
                  enumerate(sorted(best_inception, key=lambda item: float(item[0]), reverse=True)[:3], 1)]
        job_dir = self.root / 'cloud-jobs' / f'{step:09d}-{time.time_ns()}'
        job_dir.mkdir(parents=True)
        paths = []
        for name, source in slots:
            if not source.is_file():
                raise FileNotFoundError(f'Missing checkpoint selected for upload: {source}')
            target = job_dir / name
            _link(source, target)
            paths.append(target)
        job = dict(epoch=epoch, global_step=step, paths=[str(path) for path in paths],
                   run='/'.join([wb.entity, wb.project, wb.id]), state='pending')
        _atomic_json(job_dir / 'job.json', job)

        def publish():
            try:
                if self.cloud is None:
                    self.cloud = self.cloud_factory(job['run'], self.root / 'cloud-checkpoints.json')
                self.cloud(paths, epoch=epoch)
            except Exception as error:
                job.update(state='failed', error=repr(error), failed_unix=time.time())
                _atomic_json(job_dir / 'job.json', job)
                # The journal and frozen upload inputs remain available for a
                # retry; a failed upload must never prevent the next local save.
                print(json.dumps(dict(phase='checkpoint_cloud_upload_failed', **job)), flush=True)
            else:
                job.update(state='verified', verified_unix=time.time())
                _atomic_json(job_dir / 'job.json', job)
                for path in paths:
                    path.unlink()
                print(json.dumps(dict(phase='checkpoint_cloud_verified', **job)), flush=True)

        self.pending.append(self.pool.submit(publish))

    def drain(self):
        pending, self.pending = self.pending, []
        for job in pending:
            job.result()

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.drain()
            finally:
                self.pool.shutdown(wait=True)
