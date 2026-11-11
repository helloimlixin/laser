"""Monitoring and checkpoint preservation for the original joint-token stage 1."""
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch


def learning_rate(cfg, step):
    schedule = cfg['continuation']['learning_rate']
    fraction = min(1., max(0., (step - schedule['start_step']) /
                           (schedule['end_step'] - schedule['start_step'])))
    return schedule['minimum'] + .5 * (schedule['maximum'] - schedule['minimum']) * (
        1. + math.cos(math.pi * fraction))


def install(module):
    original_log = module.Experiment.log
    original_save = module.Experiment.save
    original_evaluate = module.Experiment.evaluate
    original_init = module.Experiment.__init__

    def mirror(self, name):
        if self.smoke:
            return
        source = self.out / name
        destination = Path(self.cfg['continuation']['durable']) / name
        if source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + '.pending')
            shutil.copyfile(source, temporary)
            temporary.replace(destination)

    def log(self, phase, **values):
        original_log(self, phase, **values)
        if self.rank == 0:
            with (self.out / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(phase=phase, time=time.time(), **values)) + '\n')
            mirror(self, 'status.json')
            if phase == 'train' and not self.smoke:
                durable = Path(self.cfg['continuation']['durable'])
                with (durable / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(phase=phase, time=time.time(), **values)) + '\n')

    def initialize(self, cfg, smoke=False):
        original_init(self, cfg, smoke)
        if self.run:
            self.run.summary.update({
                'initialization/trainable_weights': 'restored from best full-validation checkpoint',
                'initialization/fresh_optimizer': False,
                'continuation/source_run': cfg['continuation']['source_run'],
                'continuation/source_artifact': cfg['continuation']['source_artifact'],
                'continuation/source_step': 40036,
                'continuation/source_rfid': 2.9342634986184635,
                'reconstruction/best_rfid': 2.9342634986184635,
                'reconstruction/best_epoch': 4,
            })
            self.run.define_metric('reconstruction_full/step')
            self.run.define_metric('reconstruction_full/*', step_metric='reconstruction_full/step')
            self.run.define_metric('reconstruction_full/rfid', summary='min')
            mirror(self, 'wandb.json')
            self.save(publish=True)

    def save(self, publish=False):
        original_save(self, publish)
        if self.rank == 0:
            mirror(self, 'tokenizer-last.pt')
            mirror(self, 'checkpoint.json')
            mirror(self, 'joint-codebook.pt')

    def evaluate(self):
        self.log('evaluation_started', step=self.progress['step'], count=50000)
        started = time.monotonic()
        original_evaluate(self)
        result = json.loads((self.out / f"reconstruction-epoch{self.progress['epoch']:03d}.json").read_text())
        if self.progress['step'] == 40036:
            if abs(result['rfid'] - 2.9342634986184635) > .03:
                raise ValueError(f"Resumed 50k baseline differs from source: {result['rfid']}")
            module.atomic(self.out / 'baseline-verification.json', dict(
                passed=True, source_rfid=2.9342634986184635, **result))
            mirror(self, 'baseline-verification.json')
        module.atomic(self.out / f"reconstruction-step{self.progress['step']:09d}.json", result)
        for name in ['best-reconstruction.json', f"reconstruction-step{self.progress['step']:09d}.json"]:
            mirror(self, name)
        best = json.loads((self.out / 'best-reconstruction.json').read_text())
        if best['step'] == self.progress['step']:
            target = self.out / 'tokenizer-best-rfid.pt'
            temporary = target.with_suffix('.pending')
            shutil.copyfile(self.out / 'tokenizer-last.pt', temporary)
            temporary.replace(target)
            mirror(self, 'tokenizer-best-rfid.pt')
            checkpoint = best.get('checkpoint', {})
            if self.run and checkpoint.get('artifact'):
                import wandb
                artifact = wandb.Api().artifact(checkpoint['artifact'])
                if 'best-rfid' not in artifact.aliases:
                    artifact.aliases.append('best-rfid')
                    artifact.save()
                self.run.summary['reconstruction/best_artifact'] = checkpoint['artifact']
        self.log('evaluation_completed', step=self.progress['step'], rfid=result['rfid'],
                 seconds=time.monotonic() - started, best_rfid=best['rfid'])

    module.Experiment.__init__ = initialize
    module.Experiment.log = log
    module.Experiment.save = save
    module.Experiment.evaluate = evaluate


if __name__ == '__main__':
    import train_matched_stage1
    install(train_matched_stage1)
    train_matched_stage1.main()
