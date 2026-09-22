"""Resumable detached tokenizer -> calibration -> cache -> prior -> evaluation runner.

Credentials come exclusively from the launch environment. Each phase runs in
the immutable runtime snapshot. Checkpoint mirrors run outside GPU workers.
"""
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
import threading
import time


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointMirror:
    def __init__(self, base, local):
        self.base, self.local = base, local
        self.stop = threading.Event()
        self.published = {}
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def work(self):
        while not self.stop.is_set():
            results = {}
            for phase, names in [('tokenizer', ('tokenizer-last.pt', 'tokenizer-best.pt')),
                                 ('train', ('prior-last.pt', 'prior-best-validation.pt', 'prior-best-fid.pt'))]:
                destination = self.base / phase
                destination.mkdir(exist_ok=True)
                for name in names:
                    source = self.local / phase / name
                    if not source.exists():
                        continue
                    key = f'{phase}/{name}'
                    temporary = destination / (name + '.mirror.tmp')
                    target = destination / name
                    try:
                        stamp = source.stat().st_mtime_ns
                        if self.published.get(key) == stamp and target.exists():
                            results[key] = dict(status='current', source_mtime_ns=stamp)
                            continue
                        with source.open('rb') as stream, temporary.open('wb') as output:
                            info = os.fstat(stream.fileno())
                            shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
                        if temporary.stat().st_size != info.st_size:
                            raise RuntimeError('Incomplete mirror')
                        temporary.replace(target)
                        self.published[key] = info.st_mtime_ns
                        results[key] = dict(status='copied', source_mtime_ns=info.st_mtime_ns, bytes=info.st_size)
                    except Exception as error:
                        temporary.unlink(missing_ok=True)
                        results[key] = dict(status='retry', error=str(error))
            atomic_json(self.base / 'mirror-status.json', dict(time=time.time(), files=results))
            self.stop.wait(15)


class Pipeline:
    def __init__(self, base, reference):
        self.base, self.reference = base, reference
        self.runtime = base / 'runtime'
        self.local = Path('/tmp/laser-var-checkpoints') / base.name
        self.cache = Path('/tmp/laser-var-token-cache') / base.name
        self.python = sys.executable
        self.stopping = False
        self.child = None
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_stop)

    def request_stop(self, *_):
        self.stopping = True
        if self.child and self.child.poll() is None:
            # Signal all ranks directly so they checkpoint before torchrun exits.
            os.killpg(self.child.pid, signal.SIGTERM)

    def status(self, phase, **extra):
        value = dict(phase=phase, time=time.time(), supervisor_pid=os.getpid(),
                     gpus=os.environ['CUDA_VISIBLE_DEVICES'], **extra)
        atomic_json(self.base / 'pipeline-status.json', value)
        print(json.dumps(value), flush=True)

    def torchrun(self, ranks, script, arguments):
        return [self.python, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={ranks}',
                str(self.runtime / script), *map(str, arguments)]

    def phase(self, name, command, receipt):
        if receipt.exists():
            self.status(name + '_already_complete', receipt=str(receipt))
            return
        if self.stopping:
            raise InterruptedError('Pipeline was stopped')
        with (self.base / (name + '.log')).open('a', buffering=1) as log:
            self.child = subprocess.Popen(command, cwd=self.runtime, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.status(name, worker_pid=self.child.pid, command=command, started_unix=time.time())
            while self.child.poll() is None:
                time.sleep(5)
            code = self.child.returncode
            self.child = None
        if code != 0 or self.stopping:
            raise RuntimeError(f'{name} exited with code {code}; see {name}.log')
        if not receipt.exists():
            raise RuntimeError(f'{name} exited without its completion receipt: {receipt}')

    def prepare_selected_tokenizer(self):
        target = self.base / 'tokenizer-selected.pt'
        if target.exists():
            return
        import torch
        torch.set_num_threads(4)
        source = self.local / 'tokenizer/tokenizer-best.pt'
        checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        best = json.loads((self.base / 'tokenizer/tokenizer-best.json').read_text())
        selected = dict(model=checkpoint['model'], progress=checkpoint['progress'],
                        initialization=checkpoint['initialization'], selection=best,
                        source_checkpoint_sha256=sha256(source))
        temporary = target.with_suffix('.tmp')
        torch.save(selected, temporary)
        temporary.replace(target)
        atomic_json(self.base / 'tokenizer-selection.json', dict(checkpoint=str(target), sha256=sha256(target),
                    selected_epoch=checkpoint['progress']['epoch'], validation_images=best['count'],
                    matched_rfid=best['matched_rfid'], optimizer_transferred_to_prior=False))
        # The selected tokenizer is necessary to decode any prior checkpoint.
        import wandb
        run = wandb.init(entity='helloimlixin-rutgers', project='laser',
                         id='celebahq256-var341-tokenizer-20260921', resume='allow',
                         group=self.base.name, dir=str(Path('/tmp/laser-var-wandb') / self.base.name / 'tokenizer'))
        artifact = wandb.Artifact(self.base.name + '-selected-tokenizer', type='model', metadata=best)
        artifact.add_file(str(target), name='tokenizer-selected.pt')
        artifact.add_file(str(self.base / 'tokenizer-selection.json'), name='selection.json')
        artifact.add_file(str(self.base / 'tokenizer/resolved-config.yaml'), name='resolved-config.yaml')
        run.log_artifact(artifact).wait()
        run.finish()

    def run(self):
        self.status('waiting_for_baseline', reference=str(self.reference))
        handed_off = False
        while not (self.reference / 'train/complete.json').exists():
            handoff = self.reference / 'gpu-handoff.json'
            if handoff.exists():
                receipt = json.loads(handoff.read_text())
                if receipt.get('training_budget_complete') and receipt.get('gpu_workers_exited') and receipt.get('durable_checkpoints_verified'):
                    handed_off = True
                    break
            state = json.loads((self.reference / 'supervisor-status.json').read_text())
            if state['phase'] == 'stopped':
                raise RuntimeError('Baseline stopped unexpectedly; inspect its saved state before replacing its GPU workers')
            if self.stopping:
                raise InterruptedError('Pipeline was stopped')
            time.sleep(10)
        # Wait for all old CUDA contexts to exit, including final W&B cleanup.
        while not handed_off and json.loads((self.reference / 'supervisor-status.json').read_text())['phase'] != 'complete':
            if self.stopping:
                raise InterruptedError('Pipeline was stopped')
            time.sleep(5)
        self.mirror = CheckpointMirror(self.base, self.local)
        tokenizer_cfg = self.runtime / 'configs/experiments/celebahq256-var341-tokenizer.yaml'
        prior_cfg = self.runtime / 'configs/experiments/celebahq256-var341-compound.yaml'
        preflight = self.base / 'tokenizer-preflight'
        local_preflight = self.local / 'tokenizer-preflight'
        for phase in ('tokenizer', 'train'):
            (Path('/tmp/laser-var-wandb') / self.base.name / phase).mkdir(parents=True, exist_ok=True)
        smoke_args = ['--config', tokenizer_cfg, 'smoke_steps=6', 'logging.every_steps=1',
                      'tokenizer.adversarial_start=0', 'wandb.mode=disabled',
                      f'output_dir={preflight}', f'execution.checkpoint_dir={local_preflight}']
        self.phase('tokenizer_preflight', self.torchrun(3, 'train.py', smoke_args), preflight / 'tokenizer-smoke-complete.json')
        import torch
        state = torch.load(local_preflight / 'tokenizer-last.pt', map_location='cpu', weights_only=False, mmap=True)
        if state['progress']['step'] < 6 or len(state['rng']) != 3 or not state['optimizer']['state'] or not state['discriminator_optimizer']['state']:
            raise RuntimeError('Tokenizer preflight failed full-state checkpoint validation')
        atomic_json(preflight / 'complete.json', dict(steps=state['progress']['step'], ranks=3,
            finite_gradients=True, generator_and_discriminator_optimizers=True, retained_scale_indices=state['initialization']['retained_source_scale_indices']))
        del state
        self.phase('tokenizer_training', self.torchrun(3, 'train.py', ['--config', tokenizer_cfg]), self.base / 'tokenizer/tokenizer-complete.json')
        self.prepare_selected_tokenizer()
        self.phase('stochastic_calibration', self.torchrun(1, 'train.py', ['--config', prior_cfg, 'compound.mode=audit']), self.base / 'audit/calibration.json')
        self.phase('token_cache', self.torchrun(3, 'scripts/tools/prepare_var_compound_cache.py',
            ['--config', prior_cfg, '--output', self.cache, '--variants', '16', '--batch-size', '128']), self.cache / 'manifest.json')
        self.phase('prior_preflight', self.torchrun(3, 'train.py', ['--config', prior_cfg, 'compound.mode=preflight']), self.base / 'preflight/complete.json')
        self.phase('prior_training', self.torchrun(3, 'train.py', ['--config', prior_cfg]), self.base / 'train/complete.json')
        self.phase('final_evaluation', self.torchrun(3, 'scripts/tools/evaluate_var_compound.py',
            ['--config', prior_cfg, '--checkpoint', self.local / 'train/prior-best-fid.pt', '--samples', '50000']), self.base / 'evaluate-best/complete.json')
        # Do not claim durable pipeline completion before the latest generations have mirrored.
        while True:
            receipt = json.loads((self.base / 'mirror-status.json').read_text())
            pending = []
            for phase, names in [('tokenizer', ('tokenizer-last.pt', 'tokenizer-best.pt')),
                                 ('train', ('prior-last.pt', 'prior-best-validation.pt', 'prior-best-fid.pt'))]:
                for name in names:
                    path = self.local / phase / name
                    entry = receipt['files'].get(f'{phase}/{name}', {})
                    if entry.get('status') not in ('current', 'copied') or entry.get('source_mtime_ns') != path.stat().st_mtime_ns:
                        pending.append(f'{phase}/{name}')
            if not pending:
                break
            self.status('waiting_for_checkpoint_mirror', pending=pending)
            if self.stopping:
                raise InterruptedError('Pipeline was stopped')
            time.sleep(15)
        self.mirror.stop.set()
        self.mirror.thread.join()
        atomic_json(self.base / 'complete.json', dict(completed_unix=time.time(), stages='tokenizer, calibration, cache, prior, held-out evaluation', all_ranks=3))
        self.status('complete')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    args.base.mkdir(parents=True, exist_ok=True)
    with (args.base / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = Pipeline(args.base, args.reference)
        try:
            pipeline.run()
        except BaseException as error:
            pipeline.status('stopped' if pipeline.stopping else 'failed', error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
