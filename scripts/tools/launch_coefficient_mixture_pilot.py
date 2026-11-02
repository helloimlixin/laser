"""Pause at an atomic recovery snapshot, run a bounded head trial, resume training."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from omegaconf import OmegaConf
import psutil

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-imagenet-compound512-audit-20261026')
PILOT = BASE/'mixture-head'
SOURCE = PILOT/'source'
ORIGINAL_SOURCE = BASE/'speedup/source'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


class Controller:
    def __init__(self, config):
        self.config = config
        self.output = ROOT/'outputs'/config.run_id
        self.output.mkdir(parents=True, exist_ok=True)
        self.paused = False
        self.child = None
        self.wb = None
        self.logged_epochs = 0

    def status(self, phase, **value):
        record = dict(phase=phase, supervisor_pid=os.getpid(), updated_unix=time.time(),
                      run_id=self.config.run_id, **value)
        write(PILOT/'status.json', record)
        write(self.output/'status.json', record)
        print(json.dumps(record), flush=True)
        return record

    def environment(self, original=False):
        sys.path.insert(0, str(ORIGINAL_SOURCE/'scripts/tools'))
        from tune_imagenet_compound_execution import environment
        env = environment(4, True)
        if not original:
            env['PYTHONPATH'] = ':'.join(map(str, [SOURCE, SOURCE/'runtime', SOURCE/'scripts/tools']))
        return env

    def pause(self):
        current = json.loads((BASE/'quality/status.json').read_text())
        latest = json.loads((BASE/'recovery/checkpoints/latest.json').read_text())
        target = 100 * math.ceil((max(current['latest_metrics']['train/global_step'], latest['global_step'])+1)/100)
        self.status('waiting_for_atomic_production_checkpoint', target_step=target)
        while True:
            receipt = json.loads((BASE/'recovery/checkpoints/latest.json').read_text())
            if receipt['global_step'] >= target:
                break
            time.sleep(.5)
        # Capture a durable inode before periodic checkpoint cleanup can run.
        temporary = PILOT/'production-checkpoint.link.tmp'
        temporary.unlink(missing_ok=True)
        os.link(receipt['path'], temporary)
        os.replace(temporary, PILOT/'production-checkpoint.pt')
        worker = psutil.Process(current['child_pid'])
        if str(ORIGINAL_SOURCE/'train.py') not in ' '.join(worker.cmdline()):
            raise RuntimeError('Production process identity changed')
        tree = worker.children(recursive=True)
        supervisor = psutil.Process(current['supervisor_pid'])
        supervisor_command = ' '.join(supervisor.cmdline())
        if not any(name in supervisor_command for name in [str(BASE/'quality/controller.py'),
                str(PILOT/'source/scripts/tools/launch_coefficient_mixture_pilot.py')]):
            raise RuntimeError('Production supervisor identity changed')
        # No original controller can restart a worker after the handoff.
        supervisor.terminate()
        self.paused = True
        worker.terminate()
        def alive_processes(processes):
            result = []
            for process in processes:
                try:
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        result.append(process)
                except psutil.NoSuchProcess:
                    pass
            return result
        # These processes belong to the preceding supervisor. Waiting through
        # waitpid is unsupported for them on this runtime; poll their lifetime.
        deadline = time.monotonic() + 45
        alive = [worker] + tree
        while alive and time.monotonic() < deadline:
            alive = alive_processes(alive)
            time.sleep(.25)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
        deadline = time.monotonic() + 120
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            if time.monotonic() > deadline:
                raise RuntimeError('GPU workers did not exit after the checkpoint handoff')
            time.sleep(1)
        write(PILOT/'production-handoff.json', dict(receipt=receipt,
            checkpoint=str(PILOT/'production-checkpoint.pt'), source_checkpoint_step=self.config.source_step,
            original_config=str(BASE/'quality/production.yaml'), optimizer_retained=True, rng_ranks=8))
        self.status('production_checkpoint_preserved', saved_step=receipt['global_step'])

    def execute(self, mode, *, variant='mixture', samples=5000):
        c = self.config
        label = mode if mode != 'evaluate' else f'{variant}-fid{samples}'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
            str(SOURCE/'scripts/tools/experiment_coefficient_mixture.py'), mode,
            '--checkpoint', str(c.source_checkpoint), '--output', str(PILOT),
            '--training-images', str(c.training_images), '--validation-images', str(c.validation_images),
            '--components', str(c.components), '--width', str(c.width), '--seed', str(c.feature_seed),
            '--min-scale', str(c.minimum_scale),
            '--distill-epochs', str(c.distill_epochs), '--fit-epochs', str(c.fit_epochs),
            '--batch-sites', str(c.batch_sites_per_rank), '--variant', variant, '--samples', str(samples)]
        path = PILOT/(label+'.log')
        with path.open('a') as log:
            self.child = subprocess.Popen(command, cwd=SOURCE, env=self.environment(), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while self.child.poll() is None:
                value = dict(child_pid=self.child.pid, log=str(path), variant=variant)
                if mode == 'fit' and (PILOT/'fit-status.json').exists():
                    row = json.loads((PILOT/'fit-status.json').read_text())
                    value['fit'] = row
                    if self.wb is not None and row['epoch'] > self.logged_epochs:
                        self.wb.log(dict(fit_epoch=row['epoch'], phase=row['phase'],
                            mixture_heldout_nll=row['heldout']['mixture']['mean_nll'],
                            control_heldout_nll=row['heldout']['control']['mean_nll'],
                            mixture_train_ce=row['mixture_train_ce'], control_train_ce=row['control_train_ce']))
                        self.logged_epochs = row['epoch']
                self.status(label, **value)
                time.sleep(10)
        code = self.child.returncode
        self.child = None
        if code:
            raise RuntimeError(f'{label} exited {code}; see {path}')
        if mode == 'evaluate':
            return json.loads((PILOT/f'{variant}-fid{samples}.json').read_text())

    def trial(self):
        import wandb
        os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/imagenet-stage2-wandb.key').read_text().strip()
        self.wb = wandb.init(entity='helloimlixin-rutgers', project='laser', id=self.config.run_id,
            name='ImageNet | logistic mixture8 | frozen coefficient-head pilot',
            config=OmegaConf.to_container(self.config, resolve=True), dir=str(BASE/'wandb'))
        write(self.output/'wandb.json', dict(url=self.wb.url, id=self.wb.id))
        self.execute('canary')
        if not json.loads((PILOT/'gpu-canary.json').read_text())['passed']:
            raise RuntimeError('Full-model gradient check failed')
        self.execute('extract')
        self.execute('fit')
        mixture = self.execute('evaluate', variant='mixture', samples=int(self.config.screening_samples))
        control = self.execute('evaluate', variant='control', samples=int(self.config.screening_samples))
        result = dict(frozen_backbone=True, source_step=self.config.source_step,
            screening=dict(original=self.config.stored_original_fid5000, mixture=mixture, control=control),
            fit=json.loads((PILOT/'fit-result.json').read_text()), adopted=False)
        gain = self.config.screening_minimum_fid_gain
        if mixture['fid'] < min(control['fid'], self.config.stored_original_fid5000) - gain:
            result['confirmation'] = dict(
                mixture=self.execute('evaluate', variant='mixture', samples=int(self.config.confirmation_samples)),
                control=self.execute('evaluate', variant='control', samples=int(self.config.confirmation_samples)),
                original=self.config.stored_original_fid50000)
            result['improves_both_comparators'] = result['confirmation']['mixture']['fid'] < min(
                result['confirmation']['control']['fid'], self.config.stored_original_fid50000)
        write(PILOT/'result.json', result)
        self.publish(result)

    def publish(self, result):
        import wandb
        for f in PILOT.iterdir():
            if f.is_file() and (f.suffix in {'.json', '.jsonl', '.yaml', '.xml'} or f.name.endswith('-best.pt') or f.name == 'original-heads.pt'):
                shutil.copyfile(f, self.output/f.name)
                self.wb.save(str(self.output/f.name), base_path=str(self.output), policy='now')
        for variant in ['mixture', 'control']:
            folder = PILOT/f'{variant}-preview'
            if folder.exists():
                for image in folder.rglob('*.png'):
                    destination = self.output/f'{variant}-preview'/image.name
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(image, destination)
                    if image.name.endswith('_'+variant+'.png'):
                        shutil.copyfile(image, self.output/(variant+'-preview.png'))
                        self.wb.log({variant+'_preview': wandb.Image(str(image))})
        self.wb.summary.update(dict(source_step=self.config.source_step,
            baseline_fid5000=self.config.stored_original_fid5000,
            mixture_fid5000=result['screening']['mixture']['fid'],
            control_fid5000=result['screening']['control']['fid'], frozen_backbone=True,
            production_sampler_or_weights_changed=False))
        if 'confirmation' in result:
            self.wb.summary.update(dict(mixture_fid50000=result['confirmation']['mixture']['fid'],
                control_fid50000=result['confirmation']['control']['fid'],
                baseline_fid50000=self.config.stored_original_fid50000))
        self.wb.finish()
        self.wb = None

    def resume(self):
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)
            try:
                self.child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
        if self.wb is not None:
            self.wb.finish(exit_code=1)
            self.wb = None
        path = BASE/'quality/production.yaml'
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                   str(ORIGINAL_SOURCE/'train.py'), '--config', str(path)]
        log_path = PILOT/'resumed-production.log'
        with log_path.open('a') as log:
            child = subprocess.Popen(command, cwd=ORIGINAL_SOURCE, env=self.environment(original=True),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                metrics = json.loads((BASE/'train/metrics.jsonl').read_text().splitlines()[-1])
                record = self.status('production_resumed', child_pid=child.pid, production_run_id=
                    'imagenet-compound512-audited-8a100-20261026', log=str(log_path),
                    latest_metrics={k:metrics[k] for k in ['train/global_step', 'train/loss', 'train/images_per_second']})
                record['status_file'] = str(PILOT/'status.json')
                write(BASE/'active-runtime.json', record)
                write(BASE/'quality/status.json', record)
                write(self.output/'production-status.json', record)
                time.sleep(10)
        if child.returncode:
            raise RuntimeError(f'Original training resume failed; see {log_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/imagenet-coefficient-mixture-headpilot.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if not (PILOT/'tests-passed.json').exists():
        raise RuntimeError('The mixture probability and integration tests must pass first')
    if (PILOT/'production-handoff.json').exists():
        raise RuntimeError('This pilot already has a production handoff; resume its state explicitly')
    controller = Controller(config)
    OmegaConf.save(config, PILOT/'protocol.yaml')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        controller.pause()
        controller.trial()
    except BaseException as error:
        write(PILOT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        controller.status('pilot_failed', error=repr(error))
    finally:
        if controller.paused:
            # Recovery uses the preserved original model/optimizer/data/RNG state.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            controller.resume()


if __name__ == '__main__':
    main()
