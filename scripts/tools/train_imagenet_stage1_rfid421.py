#!/usr/bin/env python3
"""Train the small dictionary with the preserved rFID4.21 Stage-1 runtime.

The runtime is extracted from the audited ffhq-k4-continue150-20260923 archive.
Its dictionary updater is AST-identical to the ImageNet rFID4.210914 updater.
This entry point changes data access and telemetry, not the training objective.
"""
import argparse
import faulthandler
from datetime import timedelta
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-data', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    faulthandler.enable()
    if args.smoke:faulthandler.dump_traceback_later(180, repeat=True)
    runtime = args.runtime_root.resolve()
    project = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(runtime), str(runtime/'third_party/rq-vae-transformer')]

    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Subset
    from torchvision.utils import make_grid, save_image
    from omegaconf import OmegaConf
    from rqvae.models import create_model
    from rqvae.optimizer import create_optimizer, create_scheduler
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.trainers.trainer_rqvae import Trainer
    from rqvae.utils.dist import DistEnv
    from rqvae.utils.writer import Writer
    # Import indexed JPEG access without importing the workspace's src package.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'indexed_imagenet', project/'src/training/rq_imagenet_runtime.py')
    indexed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(indexed)
    write_json = indexed.write_json

    rank, local, world = (int(os.environ[x]) for x in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    assert world in (4, 8), 'Use four or eight GPUs with global batch128.'
    torch.set_num_threads(4)
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    env = DistEnv(world, rank, local, 1, rank == 0, torch.cuda.get_device_name(local))
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(args.config)
    assert world * cfg.experiment.batch_size == cfg.experiment.total_batch_size == 128
    if args.smoke_data:
        assert args.smoke
        cfg.dataset.root = str(args.smoke_data.resolve())
    cfg.result_path = str(args.output.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    source_hashes = {str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(runtime.rglob('*.py'))}
    signature = dict(config=OmegaConf.to_container(cfg, resolve=True), source_sha256=source_hashes,
                     driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    signature['sha256'] = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    train = indexed.TarImageNet(cfg.dataset.root, 'train', create_transforms(cfg.dataset,'train'), seed=cfg.seed)
    val = indexed.TarImageNet(cfg.dataset.root, 'val', create_transforms(cfg.dataset,'val'), seed=cfg.seed)
    if args.smoke:
        assert len(train) >= 384 and len(val) >= 128
        # Three complete real-image updates, with the production batch and BN scope.
        train = Subset(train, range(384)); val = Subset(val, range(128))
        cfg.experiment.num_workers = 2
        cfg.experiment.recovery_ckpt_freq_steps = 0
    else:
        assert len(train) == 1281167 and len(val) == 50000
    model, _ = create_model(cfg.arch)
    assert isinstance(model.quant_conv, torch.nn.Conv2d)
    assert isinstance(model.post_quant_conv, torch.nn.Conv2d)
    assert model.quantizer.dictionary_update_mode == 'alternating_residual'
    assert not model.quantizer.dictionary.requires_grad
    model.to(device)
    optimizer = create_optimizer(model,cfg)
    # Keep smoke LR above zero so every trainable component must actually update.
    steps = math.ceil(1281167/128)
    scheduler = create_scheduler(optimizer,cfg.optimizer.warmup,steps,cfg.experiment.epochs,env)
    ck = torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if ck:
        assert ck['resume_signature']['sha256'] == signature['sha256']
        model.load_state_dict(ck['state_dict'],strict=True)
        optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler'])
        for state in optimizer.state.values():
            for name,value in state.items():
                if torch.is_tensor(value) and name != 'step': state[name] = value.to(device)
    model = DDP(model,device_ids=[local],broadcast_buffers=True)

    class NullWriter:
        wandb_run = None
        def __getattr__(self,name): return lambda *a,**kw:None

    class LiveWriter(Writer):
        quantizer = None
        def add_scalar(self,tag,scalar,mode,epoch=0):
            super().add_scalar(tag,scalar,mode,epoch)
            if mode == 'train' and tag == 'lr_step' and self.quantizer is not None:
                q = self.quantizer
                record = dict(phase='training',completed_updates=epoch+1,time=time.time(),
                    dictionary_updated_atoms=int(q._last_dictionary_updated_atom_count),
                    dictionary_relative_improvement=float(q._last_dictionary_update_relative_improvement))
                write_json(args.output/'progress.json',record)
                self.wandb_run.log({'train/dictionary_updated_atoms':record['dictionary_updated_atoms'],
                    'train/dictionary_relative_improvement':record['dictionary_relative_improvement'],
                    'train/completed_updates':epoch+1})

    writer = LiveWriter(cfg.result_path) if rank == 0 and not args.smoke else NullWriter()
    if rank == 0 and not args.smoke:
        writer.wandb_run.define_metric('valid/loss/rfid', summary='min')
        writer.wandb_run.summary.update(dict(target_rfid=4.210914134979248,
            rfid_protocol='original-rqvae; reconstructed validation50k versus original validation50k'))

    class AuditedTrainer(Trainer):
        @torch.no_grad()
        def reconstruct(self, xs, epoch, mode='valid'):
            # Epoch logging is rank zero only; avoid DDP forward collectives here.
            raw = self.model.module
            was = raw.training
            rng = self._capture_rng_state()
            raw.eval()
            real = xs[:16]
            fake = raw(real)[0]
            real, fake = raw.get_recon_imgs(real, fake)
            grid = make_grid(torch.cat([real[:8],fake[:8],real[8:],fake[8:]]),nrow=8)
            self.writer.add_image('reconstruction',grid,mode,epoch)
            raw.train(was)
            self._restore_rng_state(rng)

        @torch.no_grad()
        def _log_laser_visuals(self, step, *, include_dictionary=False):
            # Fixed panel, native 256px tiles, explicitly converted by Writer.
            fixed = getattr(self,'_fixed_train_visual_inputs',None)
            if fixed is None: return
            raw = self.model.module
            was = raw.training
            rng = self._capture_rng_state()
            raw.eval()
            xs = fixed[:32].to(self.device)
            ys = raw(xs)[0]
            grid = make_grid(torch.cat([(xs*.5+.5).clamp(0,1),
                                        (ys*.5+.5).clamp(0,1)]),nrow=8)
            save_image(grid,Path(self.config.result_path)/'reconstruction-latest.png')
            self.writer.add_image('originals_then_reconstructions',grid,'train',step)
            raw.train(was)
            self._restore_rng_state(rng)

        def train(self,*a,epoch=0,**kw):
            dataset = self.dataset_trn
            if isinstance(dataset,Subset):dataset = dataset.dataset
            dataset.set_epoch(epoch)
            result = super().train(*a,epoch=epoch,**kw)
            if self.distenv.master:
                write_json(Path(self.config.result_path)/'progress.json',
                           dict(phase='validating',completed_train_epochs=epoch+1,time=time.time()))
            return result

    trainer = AuditedTrainer(model,None,train,val,cfg,writer,device,env,
        resume_signature=signature,
        disc_state_dict=ck['discriminator'] if ck else None,
        disc_optimizer_state_dict=ck['discriminator_optimizer'] if ck else None,
        disc_scheduler_state_dict=ck['discriminator_scheduler'] if ck else None,
        resume_parent_checkpoint_id=ck['checkpoint_id'] if ck else None)
    # Explicit worker seeds plus indexed per-image/epoch augmentation permit replay.
    for name,dataset,sampler in [('trn',train,trainer.sampler_trn),('val',val,trainer.sampler_val)]:
        setattr(trainer,'loader_'+name,DataLoader(dataset,batch_size=cfg.experiment.batch_size,sampler=sampler,
            num_workers=cfg.experiment.num_workers,pin_memory=True,prefetch_factor=2))
    assert trainer.discriminator_batchnorm_scope['reference_per_replica_batch_size'] == 64
    assert trainer.discriminator_batchnorm_scope['sync_group_size'] == 64 // cfg.experiment.batch_size
    trainer.perceptual_loss.eval()
    if rank == 0 and not args.smoke:writer.quantizer = model.module.quantizer
    if args.smoke:
        for opt in [optimizer,trainer.disc_optimizer]:
            for group in opt.param_groups:group['lr'] = 4e-5
        # Discriminator schedule is based on the tiny loader; disable its warmup for smoke.
        class FixedLR:
            def step(self): pass
            def get_last_lr(self): return [4e-5]
            def state_dict(self): return {'fixed_lr':4e-5}
        scheduler = FixedLR(); trainer.disc_scheduler = FixedLR()
        before = {k:v.detach().clone() for k,v in model.module.state_dict().items() if
                  k in ['encoder.conv_in.weight','decoder.conv_out.weight','quant_conv.weight','post_quant_conv.weight','quantizer.dictionary']}
        disc_before = next(trainer.discriminator.parameters()).detach().clone()
        trainer.train(optimizer,scheduler,epoch=0)
        after = model.module.state_dict()
        deltas = {k:float((after[k]-v).abs().max()) for k,v in before.items()}
        deltas['discriminator'] = float((next(trainer.discriminator.parameters()).detach()-disc_before).abs().max())
        assert all(v > 0 for v in deltas.values()),deltas
        assert all(torch.isfinite(p).all() for p in model.parameters())
        dictionary = model.module.quantizer.dictionary.detach()
        gathered = [torch.empty_like(dictionary) for _ in range(world)]
        dist.all_gather(gathered,dictionary)
        assert all(torch.equal(dictionary,d) for d in gathered)
        metric_summary = trainer.eval(epoch=0)
        rfid = float(metric_summary['rfid'])
        assert math.isfinite(rfid)
        trainer.save_ckpt(optimizer,scheduler,0,batch_idx=3,global_step=3,upload=False)
        reloaded = torch.load(args.output/'last_model.pt',map_location='cpu',weights_only=False)
        for key in before:
            torch.testing.assert_close(reloaded['state_dict'][key],model.module.state_dict()[key].cpu(),rtol=0,atol=0)
        assert reloaded['discriminator_optimizer'] and reloaded['optimizer']
        if rank == 0:
            write_json(args.output/'verification.json',dict(passed=True,updates=3,world_size=world,
                global_batch=128,component_max_parameter_changes=deltas,dictionary_identical_across_ranks=True,
                original_rqvae_rfid128_smoke_only=rfid,checkpoint_reloaded_exactly=True,
                discriminator_batchnorm=trainer.discriminator_batchnorm_scope,signature=signature))
    else:
        if rank == 0:
            writer.update_config(cfg)
            writer.wandb_run.config.update(dict(reference_run='imga16384k4altbn64-b128-b300-20260830000755',
                target_rfid=4.210914134979248, atoms=int(cfg.arch.hparams.n_embed),
                fixes=['learned_quant_conv_and_post_quant_conv','alternating_residual_dictionary',
                       'reference_encoder_decoder','fp32','bn64','reference_cosine_warmup','original_rqvae_rfid'],
                coefficient_quantization_bits=0,initialization='scratch',world_size=world,
                effective_batch_size=128,training_images=1281167,validation_images=50000),allow_val_change=True)
            writer.wandb_run.save(str(args.config.resolve()),base_path=str(args.config.resolve().parent))
            write_json(args.output/'progress.json',dict(phase='training',pid=os.getpid(),world_size=world,
                initialized_from_checkpoint=bool(ck),target_epochs=cfg.experiment.epochs,time=time.time()))
        trainer.run_epoch(optimizer,scheduler,epoch_st=int(ck['epoch']) if ck else 0,
            resume_batch_idx=int(ck['batch_idx']) if ck else 0,
            resume_rng_state=ck['rng_state_by_rank'][rank] if ck else None,
            resume_epoch_start_rng_state=ck['epoch_start_rng_state_by_rank'][rank] if ck else None,
            resume_accumulator_state=ck['train_accumulator_state_by_rank'][rank] if ck else None)
        if rank == 0:write_json(args.output/'progress.json',dict(phase='completed',epochs=cfg.experiment.epochs,time=time.time()))
    dist.barrier()
    if rank == 0 and not args.smoke:writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
