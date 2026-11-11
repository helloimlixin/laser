"""Restore the measured best; adapt learning rates and stop on full-FID regression."""
import json
from pathlib import Path
import shutil
import time

import torch

from dynamics import evaluation_due, initial_state, learning_rate as schedule_lr, observe


def learning_rate(experiment, step):
    return schedule_lr(experiment.cfg['continuation']['learning_rate'],
                       experiment.learning_dynamics, step)


def install(module):
    original_init = module.Experiment.__init__
    original_restore = module.Experiment.restore
    original_log = module.Experiment.log
    original_save = module.Experiment.save
    original_evaluate = module.Experiment.evaluate

    def mirror(self, name):
        if self.smoke:
            return
        source = self.out/name
        destination = Path(self.cfg['continuation']['durable'])/name
        if source.exists():
            temporary = destination.with_suffix(destination.suffix+'.pending')
            shutil.copyfile(source, temporary)
            temporary.replace(destination)

    def log(self, phase, **values):
        original_log(self, phase, **values)
        if self.rank == 0:
            row = json.dumps(dict(phase=phase, time=time.time(), **values))+'\n'
            with (self.out/'metrics.jsonl').open('a') as stream:
                stream.write(row)
            if not self.smoke:
                mirror(self, 'status.json')
                with (Path(self.cfg['continuation']['durable'])/'metrics.jsonl').open('a') as stream:
                    stream.write(row)

    def restore(self):
        original_restore(self)
        payload = torch.load(self.out/'tokenizer-last.pt',map_location='cpu',weights_only=False,mmap=True)
        self.learning_dynamics = dict(payload.get('learning_dynamics',initial_state()))

    def initialize(self, cfg, smoke=False):
        self.learning_dynamics = initial_state()
        self.quality_stop = None
        original_init(self, cfg, smoke)
        policy = cfg['continuation']
        assert self.world == 1 and self.global_batch == 128
        if smoke:
            # Exercise the real resumed sample cursor, not a 128-image toy epoch.
            self.train = module.ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed'])
            self.updates = len(self.train)//self.global_batch
        if self.run:
            self.run.summary.update({
                'initialization/trainable_weights':'full-state resume from current best rFID',
                'initialization/fresh_optimizer':False,
                'continuation/source_run':policy['source_run'],
                'continuation/source_artifact':policy['source_artifact'],
                'continuation/source_step':policy['source_step'],
                'continuation/source_rfid':policy['source_rfid'],
                'reconstruction/best_rfid':policy['source_rfid'],
                'reconstruction/best_epoch':4,
                'baseline/inherited_same_hardware_and_protocol':True,
                'guard/maximum_relative_regression':policy['maximum_relative_regression'],
                'guard/evaluation_every_updates':policy['evaluation_every_updates'],
            })
            self.run.define_metric('reconstruction_full/step')
            self.run.define_metric('reconstruction_full/*',step_metric='reconstruction_full/step')
            self.run.define_metric('reconstruction_full/rfid',summary='min')
            self.save(publish=True)
            best = json.loads((self.out/'best-reconstruction.json').read_text())
            best['checkpoint'] = json.loads((self.out/'checkpoint.json').read_text())
            module.atomic(self.out/'best-reconstruction.json',best)
            preserve_best(self)
            self.log('baseline_inherited',step=policy['source_step'],rfid=policy['source_rfid'],count=50000)
            mirror(self,'wandb.json')

    def save(self, publish=False):
        original_save(self,publish)
        if self.rank == 0 and publish:
            for name in ['tokenizer-last.pt','checkpoint.json','joint-codebook.pt']:
                mirror(self,name)

    def preserve_best(self):
        target = self.out/'tokenizer-best-rfid.pt'
        temporary = target.with_suffix('.pending')
        shutil.copyfile(self.out/'tokenizer-last.pt',temporary)
        temporary.replace(target)
        mirror(self,'tokenizer-best-rfid.pt')
        mirror(self,'best-reconstruction.json')
        best = json.loads((self.out/'best-reconstruction.json').read_text())
        if self.run:
            import wandb
            artifact = wandb.Api().artifact(best['checkpoint']['artifact'])
            if 'best-rfid' not in artifact.aliases:
                artifact.aliases.append('best-rfid')
                artifact.save()
            self.run.summary['reconstruction/best_artifact'] = best['checkpoint']['artifact']

    def evaluate(self):
        policy = self.cfg['continuation']
        previous_best = json.loads((self.out/'best-reconstruction.json').read_text())['rfid']
        self.log('evaluation_started',step=self.progress['step'],count=50000)
        started = time.monotonic()
        original_evaluate(self)
        result = json.loads((self.out/f"reconstruction-epoch{self.progress['epoch']:03d}.json").read_text())
        assert result['count'] == result['reference_count'] == 50000
        self.learning_dynamics, decision = observe(policy,self.learning_dynamics,previous_best,result['rfid'])
        if decision['action'].startswith('stop_'):
            self.stop = True
            self.quality_stop = decision['action']
            if self.run:
                self.run.summary['training/status'] = decision['action']
        # Publish the evaluated state with the updated scheduler/guard counters.
        self.save(publish=True)
        module.atomic(self.out/'learning-dynamics.json',self.learning_dynamics)
        module.atomic(self.out/'guard-decision.json',dict(step=self.progress['step'],**decision))
        module.atomic(self.out/f"reconstruction-step{self.progress['step']:09d}.json",result)
        if result['rfid'] < previous_best:
            best = dict(result,checkpoint=json.loads((self.out/'checkpoint.json').read_text()))
            module.atomic(self.out/'best-reconstruction.json',best)
            preserve_best(self)
        for name in ['learning-dynamics.json','guard-decision.json',
                     f"reconstruction-step{self.progress['step']:09d}.json"]:
            mirror(self,name)
        self.log('evaluation_completed',step=self.progress['step'],rfid=result['rfid'],
                 seconds=time.monotonic()-started,**{k:v for k,v in decision.items() if k!='score'})
        if self.run:
            self.run.summary.update({'guard/action':decision['action'],
                'guard/lr_factor':self.learning_dynamics['lr_factor'],
                'guard/nonimproving_checks':self.learning_dynamics['nonimproving_checks']})

    module.Experiment.__init__ = initialize
    module.Experiment.restore = restore
    module.Experiment.log = log
    module.Experiment.save = save
    module.Experiment.evaluate = evaluate


if __name__ == '__main__':
    import train_matched_stage1
    install(train_matched_stage1)
    train_matched_stage1.main()
