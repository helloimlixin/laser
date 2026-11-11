"""Bounded, memory-capped decoder-only experiment beside the live trainer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

HERE=Path(__file__).resolve().parent
SOURCE=Path('/tmp/laser-s1-lr1e9-b200-20261105')
BASE=Path('/tmp/laser-decoder-only-pilot-20261105')
RUNTIME=BASE/'runtime'
FROZEN=('encoder.','quant_conv.','quantize.')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    BASE.mkdir(exist_ok=True)
    shutil.copytree(SOURCE/'runtime',RUNTIME,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    path=RUNTIME/'train_matched_stage1.py'
    text=path.read_text()
    old="for group in self.d_optimizer.param_groups:group['lr']=learning_rate(self,self.progress['step'])"
    assert text.count(old)==1
    path.write_text(text.replace(old,"for group in self.d_optimizer.param_groups:group['lr']=cfg['discriminator_lr']"))
    shutil.copyfile(Path(__file__),RUNTIME/'decoder_only_pilot.py')
    cfg=json.loads((SOURCE/'run-output/config.json').read_text())
    best=json.loads((SOURCE/'run-output/best-reconstruction.json').read_text())
    checkpoint=SOURCE/'run-output/tokenizer-best-rfid.pt'
    assert sha(checkpoint)==best['checkpoint']['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    progress=dict(state['progress'])
    assert progress['batch']%2==0
    progress['batch']*=4 # microbatch64/accum2 -> microbatch16/accum8
    out=BASE/'run-output';out.mkdir()
    cfg.update(output=str(out),microbatch=16,accumulation=8,workers=0,
               lr=1e-7,discriminator_lr=1e-9)
    cfg['decoder_only_pilot']=dict(updates=30,frozen_prefixes=list(FROZEN),
        generator_lr=1e-7,discriminator_lr=1e-9,
        source_step=state['progress']['step'],source_rfid=best['rfid'],
        source_checkpoint_sha256=best['checkpoint']['checkpoint_sha256'],
        maximum_gpu_memory_fraction=.20,private_no_wandb=True)
    cfg['continuation']['learning_rate'].update(maximum=1e-7,minimum=1e-7,
        absolute_floor=1e-7,start_step=state['progress']['step'])
    cfg['runtime_sha256']={name:sha(RUNTIME/name) for name in cfg['runtime_sha256']}
    cfg['runtime_sha256']['decoder_only_pilot.py']=sha(RUNTIME/'decoder_only_pilot.py')
    contract={k:v for k,v in cfg.items() if k not in {'output','wandb','workers','checkpoint_every','preview_every','source_bundle'}}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    payload=dict(state,progress=progress,contract_sha256=digest,
        learning_dynamics=dict(lr_factor=1.,nonimproving_checks=0,evaluations=0,reductions=0,recoveries=0))
    torch.save(payload,out/'tokenizer-last.pt')
    write(out/'config.json',cfg);write(out/'contract.json',contract)
    write(out/'best-reconstruction.json',best)
    write(out/'initialization.json',dict(initialization='private_decoder_only_resume',
        source=cfg['decoder_only_pilot'],progress=progress))
    return cfg,best,state


def main():
    torch.set_num_threads(2)
    cfg,best,state=prepare()
    sys.path.insert(0,str(RUNTIME))
    import train_matched_stage1 as trainer
    trainer.var_dist.initialize(backend='nccl',timeout=120)
    torch.cuda.set_per_process_memory_fraction(.20,0)
    original=trainer.FourScaleJointVAE
    class DecoderOnly(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.encoder.requires_grad_(False)
            self.quant_conv.requires_grad_(False)
            self.quantize.requires_grad_(False)
    trainer.FourScaleJointVAE=DecoderOnly
    started=time.monotonic()
    experiment=None
    try:
        experiment=object.__new__(trainer.Experiment)
        experiment.learning_dynamics=dict(lr_factor=1.,nonimproving_checks=0,evaluations=0,reductions=0,recoveries=0)
        trainer.Experiment.__init__(experiment,cfg,True)
        experiment.train=trainer.ArchiveImages(cfg['train_archive'],cfg['train_index'],True,cfg['seed'])
        experiment.updates=len(experiment.train)//128
        experiment.val=trainer.ArchiveImages(cfg['val_archive'],cfg['val_index'],False,cfg['seed'],labels=cfg['val_labels'])
        frozen={name:parameter.detach().cpu().clone() for name,parameter in experiment.model.named_parameters() if name.startswith(FROZEN)}
        assert all(not parameter.requires_grad for name,parameter in experiment.model.named_parameters() if name.startswith(FROZEN))
        for name,parameter in experiment.model.named_parameters():
            assert torch.equal(parameter.detach().cpu(),state['model'][name]),name
        probe=torch.stack([experiment.val[i][0] for i in range(16)]).cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            experiment.model.eval();experiment.model(probe)
            before_codes=experiment.model.quantize.last_atom_ids.detach().cpu().clone()
        source_step=state['progress']['step']
        experiment.train_loop(max_steps=source_step+30)
        for name,parameter in experiment.model.named_parameters():
            if name in frozen:assert torch.equal(frozen[name],parameter.detach().cpu()),name
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            experiment.model.eval();experiment.model(probe)
            after_codes=experiment.model.quantize.last_atom_ids.detach().cpu()
        assert torch.equal(before_codes,after_codes),'Frozen assignment drifted'
        decoder_changes=sum(int((parameter.detach().cpu()!=state['model'][name]).sum())
            for name,parameter in experiment.model.named_parameters() if name.startswith(('decoder.','post_quant_conv.')))
        assert decoder_changes>0
        training_proof=dict(passed=True,updates=30,encoder_quantizer_parameters_unchanged=True,
            probe_token_changes=0,decoder_parameters_changed=decoder_changes,
            training_seconds=time.monotonic()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        write(BASE/'training-verification.json',training_proof)
        print(json.dumps(training_proof),flush=True)
        experiment.cfg['workers']=4
        experiment.evaluate()
        result=json.loads((Path(cfg['output'])/f"reconstruction-epoch{experiment.progress['epoch']:03d}.json").read_text())
        report=dict(source_rfid=best['rfid'],candidate_rfid=result['rfid'],
            full_validation_images=result['count'],source_step=source_step,
            candidate_step=experiment.progress['step'],improved=result['rfid']<best['rfid'],
            fid_protocol_unchanged=True,training_proof=training_proof,
            checkpoint=str(Path(cfg['output'])/'tokenizer-last.pt'),
            seconds=time.monotonic()-started)
        write(BASE/'pilot-result.json',report)
        print(json.dumps(report),flush=True)
    except Exception as error:
        write(BASE/'pilot-failure.json',dict(error=str(error),type=type(error).__name__,
            live_training_unchanged=True,time=time.time()))
        raise
    finally:
        trainer.td.destroy_process_group()


if __name__=='__main__':main()
