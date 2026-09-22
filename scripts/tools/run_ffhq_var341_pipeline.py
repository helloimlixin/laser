"""Supervise scratch FFHQ tokenizer -> stochastic cache -> compound VAR -> FID."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tools.run_var341_pipeline import Pipeline, atomic_json, sha256
from src.training.cli import load_config


class MetadataMirror:
    """Best-effort workspace copies; filesystem failures never block workers."""
    def __init__(self, source, destination):
        self.source, self.destination = source, Path(destination)
        self.stop = threading.Event()
        self.stamps = {}
        self.thread = threading.Thread(target=self.work, daemon=True)
        self.thread.start()

    def work(self):
        while True:
            errors, copied = [], 0
            for root, dirs, files in os.walk(self.source):
                dirs[:] = [d for d in dirs if d not in ('runtime', '__pycache__')]
                for name in files:
                    if name.startswith('.') or name.endswith('.tmp') or name == 'workspace-mirror-status.json':
                        continue
                    source = Path(root)/name
                    relative = source.relative_to(self.source)
                    try:
                        stat = source.stat()
                        stamp = (stat.st_mtime_ns, stat.st_size)
                        if stat.st_size > 16*1024**2 or self.stamps.get(str(relative)) == stamp:
                            continue
                        target = self.destination/relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(target.suffix + '.mirror.tmp')
                        shutil.copyfile(source, temporary)
                        temporary.replace(target)
                        self.stamps[str(relative)] = stamp
                        copied += 1
                    except OSError as error:
                        errors.append(dict(path=str(relative), error=str(error)))
            atomic_json(self.source/'workspace-mirror-status.json',
                        dict(time=time.time(), copied=copied, errors=errors, destination=str(self.destination)))
            if self.stop.wait(60):
                return


def terminate_tree(process):
    """Bound shutdown even when ranks are stuck inside a collective."""
    pending, descendants = [process.pid], []
    while pending:
        pid = pending.pop()
        try:
            children = [int(p) for p in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
        except OSError:
            children = []
        pending.extend(children)
        descendants.extend(children)
    for pid in [process.pid, *descendants]:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    for pid in reversed([process.pid, *descendants]):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


class FFHQPipeline(Pipeline):
    def __init__(self, base, tokenizer_cfg=None, prior_cfg=None):
        super().__init__(base, base)
        self.tokenizer_cfg = Path(tokenizer_cfg) if tokenizer_cfg else self.runtime/'configs/experiments/ffhq256-var341-tokenizer.yaml'
        self.prior_cfg = Path(prior_cfg) if prior_cfg else self.runtime/'configs/experiments/ffhq256-var341-compound.yaml'
        self.tokenizer_config = load_config(self.tokenizer_cfg)
        self.prior_config = load_config(self.prior_cfg)
        self.data = Path(self.tokenizer_config.data.root)
        mirror = os.environ.get('LASER_METADATA_MIRROR')
        self.mirror = MetadataMirror(base, mirror) if mirror else None

    def phase(self, name, command, receipt):
        if receipt.exists():
            self.status(name + '_already_complete', receipt=str(receipt))
            return
        for attempt in range(1, 4):
            if self.stopping:
                raise InterruptedError('Pipeline was stopped')
            started = time.time()
            with (self.base/(name + '.log')).open('a', buffering=1) as log:
                self.child = subprocess.Popen(command, cwd=self.runtime, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.status(name, worker_pid=self.child.pid, attempt=attempt,
                            command=command, started_unix=started)
                while self.child.poll() is None:
                    if name in ('tokenizer_training', 'prior_training') and not self.stopping:
                        phase = 'tokenizer' if name == 'tokenizer_training' else 'train'
                        path = self.base/phase/'status.json'
                        try:
                            progress = json.loads(path.read_text())
                            activity = max(started, float(progress['time']))
                            limit = 3600 if progress['phase'] == 'checkpoint_upload_wait' or receipt.exists() else 600
                        except (OSError, ValueError, KeyError):
                            activity, limit = started, 600
                        if time.time() - activity > limit:
                            self.status('watchdog_restart', phase_name=name, attempt=attempt,
                                        inactive_seconds=time.time()-activity)
                            terminate_tree(self.child)
                            break
                    time.sleep(5)
                code = self.child.returncode
                self.child = None
            if self.stopping:
                raise InterruptedError('Pipeline was stopped')
            if code == 0 and receipt.exists():
                return
            if attempt < 3:
                self.status('phase_retry', phase_name=name, attempt=attempt, exit_code=code)
                time.sleep(5)
        raise RuntimeError(f'{name} failed after 3 attempts; see {name}.log')

    def restore_artifact(self, name, destination):
        import wandb
        cfg = self.tokenizer_config.wandb
        try:
            artifact = wandb.Api(timeout=60).artifact(f'{cfg.entity}/{cfg.project}/{name}:latest')
        except wandb.errors.CommError as error:
            if 'not found' in str(error):
                return False
            raise
        if artifact.state != 'COMMITTED':
            return False
        self.status('restoring_online_artifact', artifact=artifact.qualified_name, destination=str(destination))
        artifact.download(root=str(destination))
        return True

    def prepare_data(self):
        manifest = self.data/'manifest.json'
        if not manifest.exists():
            launch = self.base/'data-preparation-launch.json'
            if launch.exists():
                pid = json.loads(launch.read_text())['pid']
                while Path(f'/proc/{pid}').exists() and not manifest.exists():
                    self.status('waiting_for_data_preparation', worker_pid=pid)
                    if self.stopping:
                        raise InterruptedError('Pipeline stopped')
                    time.sleep(10)
            if not manifest.exists() and not self.restore_artifact(
                    self.tokenizer_config.execution.dataset_artifact, self.data):
                raise RuntimeError('Prepared FFHQ data is missing; run prepare_ffhq_var_data.py first')
        from datasets import load_from_disk
        dataset = load_from_disk(str(self.data/'hf'))
        assert len(dataset['train']) == 60000 and len(dataset['validation']) == 10000
        assert dataset['train'].unique('label') == dataset['validation'].unique('label') == [0]
        assert set(dataset['train']['image_id']) == set(range(60000))
        assert set(dataset['validation']['image_id']) == set(range(60000, 70000))
        record = json.loads(manifest.read_text())
        assert record['original_md5_and_zip_crc_verified']
        shutil.copy2(manifest, self.base/'data-manifest.json')
        atomic_json(self.base/'data-verified.json', dict(train_images=60000, validation_images=10000,
            disjoint_official_split=True, unconditional_labels=True, source_checksums_verified=True))

    def restore_training_state(self, phase, run_id, filename):
        directory = self.local/phase
        directory.mkdir(parents=True, exist_ok=True)
        if (self.base/phase/'resolved-config.yaml').exists() and not (directory/filename).exists():
            if not self.restore_artifact(run_id + '-checkpoints', directory):
                raise RuntimeError(f'{phase} was previously initialized but no resumable checkpoint is available')

    def prepare_selected_tokenizer(self):
        import torch
        import wandb
        torch.set_num_threads(4)
        source = self.local/'tokenizer/tokenizer-best.pt'
        target = Path(self.prior_config.compound.tokenizer_checkpoint)
        selection = self.base/'tokenizer-selection.json'
        if not target.exists():
            checkpoint = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
            assert checkpoint['initialization']['initialization'] == 'scratch'
            best = json.loads((self.base/'tokenizer/tokenizer-best.json').read_text())
            value = dict(model=checkpoint['model'], progress=checkpoint['progress'],
                initialization=checkpoint['initialization'], selection=best,
                source_checkpoint_sha256=sha256(source))
            temporary = target.with_suffix('.tmp')
            torch.save(value, temporary)
            temporary.replace(target)
            atomic_json(selection, dict(checkpoint=str(target), sha256=sha256(target),
                selected_epoch=checkpoint['progress']['epoch'], validation_images=best['count'],
                matched_rfid=best['matched_rfid'], initialization='scratch', optimizer_transferred_to_prior=False))
        receipt = self.base/'tokenizer-selection-upload.json'
        if not receipt.exists():
            cfg = self.tokenizer_config.wandb
            run = wandb.init(entity=cfg.entity, project=cfg.project, id=cfg.id, resume='allow',
                             mode='online', group=cfg.group,
                             dir=str(self.tokenizer_config.execution.wandb_dir))
            artifact = wandb.Artifact(self.base.name + '-selected-tokenizer', type='model',
                                     metadata=json.loads(selection.read_text()))
            artifact.add_file(str(target), name='tokenizer-selected.pt', policy='immutable')
            artifact.add_file(str(selection), name='selection.json')
            artifact.add_file(str(self.base/'tokenizer/resolved-config.yaml'), name='resolved-config.yaml')
            run.log_artifact(artifact, aliases=['latest', 'best-rfid']).wait()
            atomic_json(receipt, dict(artifact=artifact.qualified_name, state=artifact.state))
            run.finish()

    def verify_online_checkpoints(self, phase, required):
        import wandb
        receipt = json.loads((self.base/phase/'checkpoint-upload.json').read_text())
        artifact = wandb.Api(timeout=60).artifact(receipt['artifact'])
        assert artifact.state == 'COMMITTED'
        for name in required:
            entry = artifact.manifest.entries[name]
            assert entry.size == receipt['files'][name]['bytes']
            assert entry.digest == receipt['files'][name]['digest']
        result = dict(artifact=artifact.qualified_name, state=artifact.state,
                      verified_files=required, verified_unix=time.time())
        atomic_json(self.base/phase/'online-checkpoints-verified.json', result)
        return result

    def run(self):
        for phase in ('tokenizer', 'train'):
            (Path('/tmp/laser-var-wandb')/self.base.name/phase).mkdir(parents=True, exist_ok=True)
        self.prepare_data()
        smoke = self.base/'tokenizer-preflight'
        local_smoke = self.local/'tokenizer-preflight'
        if not (smoke/'complete.json').exists():
            self.phase('tokenizer_preflight', self.torchrun(3, 'train.py', [
                '--config', self.tokenizer_cfg, 'smoke_steps=6', 'logging.every_steps=1',
                'tokenizer.adversarial_start=0', 'wandb.mode=disabled',
                f'output_dir={smoke}', f'execution.checkpoint_dir={local_smoke}']),
                smoke/'tokenizer-smoke-complete.json')
            import torch
            checkpoint = torch.load(local_smoke/'tokenizer-last.pt', map_location='cpu', weights_only=False, mmap=True)
            assert checkpoint['progress']['step'] == 6 and len(checkpoint['rng']) == 3
            assert checkpoint['optimizer']['state'] and checkpoint['discriminator_optimizer']['state']
            assert checkpoint['initialization']['initialization'] == 'scratch'
            atomic_json(smoke/'complete.json', dict(steps=6, rng_ranks=3, finite_gradients=True,
                generator_and_discriminator_optimizers=True, initialization='scratch'))
            del checkpoint
        if self.tokenizer_config.model.get('tokenized_sparse_policy'):
            self.phase('tokenized_codec_verification', self.torchrun(1,
                'scripts/tools/verify_tokenized_sparse_pipeline.py', ['--config', self.tokenizer_cfg,
                 '--checkpoint', local_smoke/'tokenizer-last.pt', '--output', smoke/'codec-verified.json']),
                 smoke/'codec-verified.json')
        self.restore_training_state('tokenizer', self.tokenizer_config.wandb.id, 'tokenizer-last.pt')
        self.phase('tokenizer_training', self.torchrun(3, 'train.py', ['--config', self.tokenizer_cfg]),
                   self.base/'tokenizer/tokenizer-complete.json')
        self.verify_online_checkpoints('tokenizer', ['tokenizer-last.pt', 'tokenizer-best.pt'])
        self.prepare_selected_tokenizer()
        self.phase('stochastic_calibration', self.torchrun(1, 'train.py',
            ['--config', self.prior_cfg, 'compound.mode=audit']), self.base/'audit/calibration.json')
        if not (self.cache/'manifest.json').exists() and (self.base/'train/resolved-config.yaml').exists():
            if not self.restore_artifact(self.prior_config.wandb.id + '-token-cache', self.cache):
                raise RuntimeError('Existing prior requires its exact token cache; no committed cache found')
        self.phase('token_cache', self.torchrun(3, 'scripts/tools/prepare_var_compound_cache.py',
            ['--config', self.prior_cfg, '--output', self.cache, '--variants', 16, '--batch-size', 128]),
            self.cache/'manifest.json')
        self.phase('prior_preflight', self.torchrun(3, 'train.py', ['--config', self.prior_cfg,
            'compound.mode=preflight', 'evaluation.fid_reference_images=32']), self.base/'preflight/complete.json')
        self.restore_training_state('train', self.prior_config.wandb.id, 'prior-last.pt')
        self.phase('prior_training', self.torchrun(3, 'train.py', ['--config', self.prior_cfg]),
                   self.base/'train/complete.json')
        online = self.verify_online_checkpoints('train', ['prior-last.pt', 'prior-best-fid.pt'])
        self.phase('final_evaluation', self.torchrun(3, 'scripts/tools/evaluate_var_compound.py',
            ['--config', self.prior_cfg, '--checkpoint', self.local/'train/prior-best-fid.pt', '--samples', 50000]),
            self.base/'evaluate-best/complete.json')
        atomic_json(self.base/'complete.json', dict(completed_unix=time.time(), world_size=3,
            initialization='scratch tokenizer and scratch prior', checkpoint_storage='W&B online',
            artifact=online['artifact'], dataset='FFHQ-256', train_images=60000, validation_images=10000))
        self.status('complete', artifact=online['artifact'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--tokenizer-config', type=Path)
    parser.add_argument('--prior-config', type=Path)
    args = parser.parse_args()
    args.base.mkdir(parents=True, exist_ok=True)
    with (args.base/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pipeline = FFHQPipeline(args.base, args.tokenizer_config, args.prior_config)
        try:
            pipeline.run()
        except BaseException as error:
            pipeline.status('stopped' if pipeline.stopping else 'failed',
                            error_type=type(error).__name__, error=str(error))
            raise


if __name__ == '__main__':
    main()
