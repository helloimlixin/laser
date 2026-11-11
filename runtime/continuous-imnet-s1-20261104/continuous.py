"""Continuous full-state training: rewind/reduce on quality regression, never idle."""
import argparse
import itertools
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

import continuation
import train_matched_stage1 as trainer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    torch.set_num_threads(4)
    trainer.var_dist.initialize(backend='nccl',timeout=120)
    base_init=trainer.Experiment.__init__
    continuation.install(trainer)
    original_loop=trainer.Experiment.train_loop

    def initialize(self,config,smoke=False):
        self.learning_dynamics=dict(lr_factor=1.,nonimproving_checks=0,evaluations=0,reductions=0)
        self.quality_stop=None
        base_init(self,config,smoke)
        best=json.loads((self.out/'best-reconstruction.json').read_text())
        self.run.summary.update({'initialization/trainable_weights':'full-state resume',
            'initialization/fresh_optimizer':False,'training/status':'running',
            'reconstruction/best_rfid':best['rfid'],
            'reconstruction/best_artifact':best['checkpoint']['artifact'],
            'guard/action_on_regression':'restore_best_reduce_lr_continue'})
        self.run.define_metric('reconstruction_full/step')
        self.run.define_metric('reconstruction_full/*',step_metric='reconstruction_full/step')
        self.run.define_metric('reconstruction_full/rfid',summary='min')
        # Existing best is already independently mirrored and published. Do not
        # block useful GPU work on republishing it every process restart.

    def recover(self,reason):
        best=json.loads((self.out/'best-reconstruction.json').read_text())
        payload=torch.load(self.out/'tokenizer-best-rfid.pt',map_location='cpu',weights_only=False,mmap=True)
        current_factor=self.learning_dynamics['lr_factor']
        previous=dict(self.progress)
        # Keep the data stream moving: weights and Adam ages rewind, while the
        # current epoch/cursor/step continue, explicitly tracked below.
        self.model.load_state_dict(payload['model'])
        self.discriminator.load_state_dict(payload['discriminator'])
        self.optimizer.load_state_dict(payload['optimizer'])
        self.d_optimizer.load_state_dict(payload['discriminator_optimizer'])
        if payload.get('record_hit') is not None:
            self.model.quantize.record_hit=payload['record_hit']
        trainer.restore_rng(payload['rng'][0])
        self.learning_dynamics=dict(payload.get('learning_dynamics',{}))
        self.learning_dynamics.update(lr_factor=max(.01,current_factor*.5),
            nonimproving_checks=0,recoveries=getattr(self,'recoveries',0)+1)
        self.recoveries=self.learning_dynamics['recoveries']
        self.stop=False;self.quality_stop=None
        self.optimizer.zero_grad(set_to_none=True)
        self.d_optimizer.zero_grad(set_to_none=True)
        receipt=dict(reason=reason,stream_progress=previous,
            restored_weight_step=payload['progress']['step'],best_rfid=best['rfid'],
            lr_factor=self.learning_dynamics['lr_factor'],recoveries=self.recoveries,
            optimizer_age=sorted({int(v['step']) for v in self.optimizer.state.values() if 'step' in v}),
            data_cursor_preserved=True,time=time.time())
        trainer.atomic(self.out/'recovery.json',receipt)
        trainer.atomic(Path(cfg['continuation']['durable'])/'recovery.json',receipt)
        self.log('automatic_recovery',**{k:v for k,v in receipt.items() if k!='time'})
        if self.run:
            self.run.summary.update({'training/status':'running',
                'recovery/count':self.recoveries,'recovery/best_rfid':best['rfid'],
                'recovery/lr_factor':self.learning_dynamics['lr_factor']})

    def continuous_loop(self,max_steps=None):
        self.recoveries=self.learning_dynamics.get('recoveries',0)
        pending=self.out/'guard-decision.json'
        if pending.exists():
            decision=json.loads(pending.read_text())
            if decision['action'].startswith('stop_') and decision['step']==self.progress['step']:
                recover(self,'restart_pending_quality_recovery')
        while True:
            original_loop(self,max_steps)
            if self.quality_stop:
                recover(self,self.quality_stop)
                continue
            return

    trainer.Experiment.train_loop=continuous_loop
    trainer.Experiment.__init__=initialize
    experiment=None
    try:
        experiment=trainer.Experiment(cfg,False)
        experiment.run.summary.update({'training/target_epochs':'until explicitly stopped',
            'training/continuous':True,'guard/action_on_regression':'restore_best_reduce_lr_continue',
            'training/optimizer_age_equals_stream_step_after_rewind':False})
        experiment.train_loop()
        if experiment.run:
            experiment.run.summary['training/status']='user_stop_checkpointed'
            experiment.run.finish()
    except Exception as error:
        if experiment is not None:
            trainer.atomic(experiment.out/'failure-rank0.json',dict(
                type=type(error).__name__,error=str(error),progress=experiment.progress))
            if experiment.run:experiment.run.finish(exit_code=1)
        raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
