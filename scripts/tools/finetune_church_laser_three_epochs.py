#!/usr/bin/env python3
"""Observe the recovered successful LASER stage-1 driver with Church's recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/church-laser-three-epoch-20260914'
SOURCE=BASE/'stage1-source'
UPSTREAM=SOURCE/'third_party/rq-vae-transformer'
CHECKPOINT=ROOT/'outputs/imagenet-scaled-rq-stage2-20260913/assets/best_rfid_slot1_model.pt'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    out=args.output.resolve();rank=int(os.environ['RANK'])
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(UPSTREAM),str(SOURCE)]
    if args.smoke:os.environ['SMOKE_TEST']='1'
    os.environ.setdefault('LASER_VGG_LPIPS_DIR',str(ROOT/'vgg_lpips'))
    os.environ.setdefault('LASER_VGG16_WEIGHTS','/workspace/tmp/laser-vgg/vgg16-397923af.pth')
    os.environ.setdefault('TORCH_HOME','/workspace/tmp/official-rqvae-eval-cache')
    import torch
    torch.set_num_threads(8)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from rqvae.utils.writer import Writer
    import rqvae.utils.setup as setup
    import rqvae.optimizer as optimizers
    from rqvae.trainers.trainer_rqvae import Trainer
    from omegaconf import OmegaConf
    started=time.time();progress=dict(step=0,epoch=0)
    expected=2 if args.smoke else 2961
    def status(phase,**values):
        if rank==0:
            record=dict(phase=phase,optimizer_step=progress['step'],
                elapsed_seconds=time.time()-started,updated_unix=time.time(),pid=os.getpid(),**values)
            write(out/'status.json',record)
            print(json.dumps(record),flush=True)

    class ObservedWriter(Writer):
        def __init__(self,result_path):
            super().__init__(result_path)
            write(out/'run-directory.json',dict(path=str(result_path)))
            if self.wandb_run:
                write(out/'wandb.json',dict(run_id=self.wandb_run.id,url=self.wandb_run.url))
        def add_scalar(self,tag,scalar,mode,epoch=0):
            value=float(scalar)
            assert torch.isfinite(torch.tensor(value)),(tag,value)
            super().add_scalar(tag,scalar,mode,epoch)
            with (out/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(tag=tag,value=value,mode=mode,step_or_epoch=epoch,time=time.time()))+'\n')
    setup.Writer=ObservedWriter
    original_optimizer=optimizers.create_optimizer
    def create_optimizer(model,config):
        optimizer=original_optimizer(model,config)
        assert not optimizer.state
        for group in optimizer.param_groups:
            assert group['lr']==4e-6 and group['betas']==(.5,.9) and group['weight_decay']==0
        if rank==0:
            write(out/'initialization.json',dict(source_checkpoint=str(CHECKPOINT),
                source_checkpoint_sha256=digest(CHECKPOINT),initial_optimizer_entries=0,
                fine_tune_from_imagenet=True,imagenet_source_rfid=4.210914134979248,
                requested_epochs=3,actual_epochs=1 if args.smoke else 3,smoke_only=args.smoke,
                global_batch=128,microbatch_per_gpu=64,precision='float32',
                source=str(SOURCE),source_recovery=str(BASE/'source-recovery/reconstruction.json')))
        def observe(optimizer,positional,keyword):
            progress['step']+=1
            step=progress['step']
            assert all(abs(g['lr']-4e-6)<1e-15 for g in optimizer.param_groups)
            if step<=3 or step%10==0:
                status('training',epoch=step/(2 if args.smoke else 987),lr=optimizer.param_groups[0]['lr'],
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        optimizer.register_step_post_hook(observe)
        return optimizer
    optimizers.create_optimizer=create_optimizer
    original_train=Trainer.train
    def train(self,*positional,**keyword):
        progress['epoch']=keyword.get('epoch',0)
        result=original_train(self,*positional,**keyword)
        status('evaluating_reconstruction',epoch=progress['epoch']+1,
            evaluation_population='126227 training images; released LSUN factory',smoke_only=args.smoke)
        return result
    Trainer.train=train
    original_save=Trainer.save_ckpt
    def save(self,optimizer,scheduler,epoch,*positional,**keyword):
        result=original_save(self,optimizer,scheduler,epoch,*positional,**keyword)
        if rank==0 and epoch>0 and keyword.get('batch_idx',0)==0:
            folder=Path(self.config.result_path)
            archive=folder/f'epoch{epoch}_model.pt'
            if not archive.exists():os.link(folder/'last_model.pt',archive)
            write(out/f'epoch{epoch}-complete.json',dict(epoch=epoch,checkpoint=str(archive),
                global_step=progress['step'],smoke_only=args.smoke))
        return result
    Trainer.save_ckpt=save
    config_path=BASE/'stage1-config.yaml'
    config=OmegaConf.load(config_path)
    assert config.experiment.epochs==3 and config.experiment.total_batch_size==128
    assert config.optimizer.init_lr==config.optimizer.warmup.min_lr==4e-6
    assert config.gan.disc.optimizer.init_lr==config.gan.disc.optimizer.warmup.min_lr==4e-6
    sys.argv=[str(UPSTREAM/'main_stage1.py'),'-m',str(config_path),'-r',str(out/'upstream-results'),
        '-l',str(CHECKPOINT),'--precision','float32']
    if args.smoke:sys.argv+=['experiment.epochs=1']
    try:
        runpy.run_path(str(UPSTREAM/'main_stage1.py'),run_name='__main__')
        assert progress['step']==expected,(progress['step'],expected)
        if rank==0:
            folder=Path(json.loads((out/'run-directory.json').read_text())['path'])
            epoch=1 if args.smoke else 3
            full=folder/f'epoch{epoch}_model.pt'
            state=torch.load(full,map_location='cpu',weights_only=False,mmap=True)
            assert state['epoch']==epoch and state['global_step']==expected
            assert state['optimizer']['state'] and state['discriminator_optimizer']['state']
            assert len(state['rng_state_by_rank'])==2
            # Tensor-only export lets the frozen stage-2 adapter avoid importing
            # historical stage-1 runtime classes when reading the checkpoint.
            exported=out/f'epoch{epoch}-tokenizer.pt'
            torch.save(dict(state_dict=state['state_dict'],epoch=epoch,global_step=expected,
                source_checkpoint_sha256=digest(CHECKPOINT)),exported)
            write(out/'complete.json',dict(phase='complete',epoch=epoch,optimizer_steps=expected,
                checkpoint=str(exported),checkpoint_sha256=digest(exported),full_checkpoint=str(full),
                config=str(folder/'config.yaml'),smoke_only=args.smoke,finished_unix=time.time()))
            status('complete',epoch=epoch,smoke_only=args.smoke)
    except BaseException as error:
        write(out/f'failure-rank{rank}.json',dict(type=type(error).__name__,error=str(error),time=time.time()))
        raise
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
