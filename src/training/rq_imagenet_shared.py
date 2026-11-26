#!/usr/bin/env python3
"""One upstream ImageNet training loop for native RQ and the LASER adapters."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import concurrent.futures
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Subset
from omegaconf import OmegaConf
from src import rq_upstream
from rqvae.utils.config import load_config,augment_arch_defaults,augment_optimizer_defaults
from rqvae.optimizer.optimizer import create_resnet_optimizer
from rqvae.optimizer.scheduler import create_scheduler
from rqvae.img_datasets.transforms import create_transforms
from src.training.exact_global_batch import ExactGlobalBatchSampler
from src.training.rq_imagenet_runtime import (
    TarImageNet,cpu_snapshot,sha256,write_json,local_training_logs,
)
from src.training import k4_checkpoint_io as checkpoint_io
from runtime.amp_replay import ReplayWindow,recoverable_amp_step
from src.training.rq_imagenet_recipes import RQRecipe,LaserRecipe
from src import rqvae_metrics as original_metrics


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def fresh_model(device):
    config=load_config(BASE/'inputs/stage2-training-config.yaml')
    archived=load_config(BASE/'inputs/stage2-archive-config.yaml')
    config.arch=augment_arch_defaults(config.arch)
    config.arch.vocab_size=config.dataset.vocab_size
    assert OmegaConf.to_container(config.arch)==OmegaConf.to_container(augment_arch_defaults(archived.arch))
    config.optimizer=augment_optimizer_defaults(config.optimizer)
    model=PROFILE.model(config.arch).to(device)
    config.arch=model.config
    config.dataset.vocab_size=max(model.vocab_size)
    optimizer=create_resnet_optimizer(model,config.optimizer)
    if getattr(PROFILE,'recipe',{}).get('execution',{}).get('fused_adamw',False):
        if not isinstance(optimizer,torch.optim.AdamW):
            raise ValueError('Fused AdamW requires the AdamW optimizer')
        for group in optimizer.param_groups:
            group.update(fused=True,foreach=False)
    assert not optimizer.state
    return model,optimizer,config


def schedule_metadata(config, scheduler):
    """Describe the configured schedule, including its actual warmup start."""
    return dict(initial_lr=float(config.optimizer.init_lr),
        initial_optimizer_lr=float(scheduler.get_last_lr()[0]),
        minimum_lr=float(config.optimizer.warmup.min_lr),
        warmup_epochs=float(config.optimizer.warmup.epoch))


def training_backward(profile,model,aux,images,labels,scaler,scale,*,amp=True,on_targets=None):
    """Identical forward/backward path; only the representation is injected."""
    targets,codes=profile.targets(aux,images)
    if on_targets is not None:on_targets(targets,codes)
    logits=model(codes,model_aux=aux,cond=labels,amp=amp)
    unwrapped=model.module if isinstance(model,DDP) else model
    loss=profile.loss(unwrapped,logits,targets)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite token cross entropy')
    scaler.scale(loss*scale).backward()
    return loss.detach()


def ensure_optimizer_covers_model(optimizer, model):
    """Enroll trainable parameters attached after optimizer construction."""
    named = dict(model.named_parameters())
    all_ids = {id(parameter) for parameter in named.values()}
    enrolled = [parameter for group in optimizer.param_groups for parameter in group['params']]
    enrolled_ids = [id(parameter) for parameter in enrolled]
    if len(enrolled_ids) != len(set(enrolled_ids)):
        raise ValueError('Optimizer contains duplicate parameter references')
    if not set(enrolled_ids).issubset(all_ids):
        raise ValueError('Optimizer refers to parameters absent from the adapted model')
    existing = set(enrolled_ids)
    missing = [(name, parameter) for name, parameter in named.items()
               if parameter.requires_grad and id(parameter) not in existing]
    if missing:
        if len(optimizer.param_groups) != 1:
            raise ValueError('New adapter parameters need explicit placement with multiple optimizer groups')
        optimizer.param_groups[0]['params'].extend(parameter for _, parameter in missing)
    covered = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
    assert all(id(parameter) in covered for parameter in named.values() if parameter.requires_grad)
    return dict(passed=True, added_parameter_names=[name for name, _ in missing],
                trainable_parameter_tensors=sum(parameter.requires_grad for parameter in named.values()),
                optimizer_parameter_tensors=sum(len(group['params']) for group in optimizer.param_groups))


def fid_workspace_title(name):
    # W&B run names allow separators that its workspace validator rejects.
    return ' '.join(''.join(c if c.isascii() and (c.isalnum() or c in '-_. ') else ' '
                           for c in name).split()) + ' FID'


def fid_workspace(run):
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr
    receipt=OUT/'workspace.json'
    if receipt.exists():return
    workspace=ws.Workspace(entity='helloimlixin-rutgers',project='laser',
        name=fid_workspace_title(PROFILE.name),
        runset_settings=ws.RunsetSettings(filters=[ws.Config('wandb_id')==RUN]),
        sections=[ws.Section(name='FID',is_open=True,pinned=True,panels=[
            wr.MarkdownPanel(markdown=(
                '**Shared upstream RQ training; one token cross entropy.** '
                'FID uses 50,000 generated images against all 1,281,167 training images, '
                'the original RQ Inception backend, and no rejection. '
                'Fresh runs evaluate after epoch 1, then every two epochs. '
                'Validated continuations also show their starting checkpoint measurement. '
                'The original 100-epoch cosine LR horizon is retained.')),
            wr.LinePlot(title='FID50k vs full ImageNet train',x='train/epoch',y=['eval/fid']),
            wr.LinePlot(title='Inception Score',x='train/epoch',y=['eval/inception_score']),
            wr.LinePlot(title='Only training loss: token cross entropy',
                x='train/global_step',y=['train/loss']),
            wr.LinePlot(title='Learning rate',x='train/global_step',y=['train/lr']),
        ])])
    if getattr(PROFILE, 'recipe', {}).get('preview'):
        workspace.sections.append(ws.Section(name='Samples',is_open=True,panels=[
            wr.MediaBrowser(title='Fixed classes and seed',media_keys=['samples/fixed'],num_columns=1)]))
        if PROFILE.recipe.preview.get('designated_classes'):
            from src.training.rq_sample_panels import CLASSES
            workspace.sections.append(ws.Section(name='Per-class samples',is_open=True,panels=[
                wr.MediaBrowser(title=name.title()+' - 64 fixed-seed samples',
                    media_keys=['samples/'+name.replace(' ','_')],num_columns=1)
                for name,_ in CLASSES]))
    if PROFILE.metadata().get('loss_component_diagnostics'):
        workspace.sections[0].panels.extend([
            wr.LinePlot(title='Joint CE components: atom, sign, magnitude',x='train/global_step',
                y=['train/atom_ce','train/sign_ce','train/magnitude_ce']),
            wr.LinePlot(title='Correct coefficient sign probability given target atom',x='train/global_step',
                y=[f'train/correct_sign_probability_depth{d}' for d in range(1,5)]),
            wr.LinePlot(title='Joint CE divided by two: per scalar event',x='train/global_step',
                y=['train/joint_ce_per_scalar_event']),
        ])
    workspace.save()
    write_json(receipt,dict(url=workspace.url,run_url=run.url,auxiliary_loss_terms=0))
    run.save(str(receipt),base_path=str(OUT),policy='now')


@torch.no_grad()
def preview(model, aux, device, rank, world, step, run):
    """Fixed class previews, with all training RNG streams restored afterward."""
    if getattr(PROFILE,'recipe',{}).get('preview',{}).get('designated_classes'):
        from src.training.rq_sample_panels import designated_preview
        return designated_preview(PROFILE,model,aux,device,rank,world,step,run,OUT)
    from torchvision.utils import save_image
    classes = torch.tensor([269,612,265,628,689,301,932,798],device=device)
    labels = classes.repeat_interleave(4)[rank::world]
    was_training = model.training
    model.eval()
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(261001+rank)
        torch.cuda.manual_seed(261001+rank)
        codes = PROFILE.sample(model,aux,len(labels),labels)
        images = torch.cat([PROFILE.decode(aux,c).float().mul(.5).add(.5).clamp(0,1)
            for c in codes.split(4)])
        gathered = [None]*world
        dist.all_gather_object(gathered,(torch.arange(rank,32,world),images.cpu()))
        if rank == 0:
            ordered = torch.empty(32,*images.shape[1:])
            for indices,values in gathered:
                ordered[indices] = values
            destination = (Path(os.environ.get('LASER_PRIVATE_SMOKE_DIR',str(OUT/'smoke')))
                if getattr(PROFILE, 'execution_phase', 'train') == 'smoke' else OUT)
            path = destination/'samples'/f'step{step:06d}-fixed.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            save_image(ordered,path,nrow=4)
            if run is not None:
                import wandb
                run.log({'samples/fixed':wandb.Image(str(path),caption=(
                    f'Step {step}; four samples per row: timber wolf, jinrikisha, toy poodle, '
                    'liner, overskirt, ladybug, pretzel, slide rule.')),
                    'samples/global_step':step})
    model.train(was_training)
    torch.cuda.empty_cache()


@torch.no_grad()
def generation(model,aux,device,rank,world,epoch,run,batch=None,seed=261001):
    batch=batch or PROFILE.generation_batch
    was_training=model.training;model.eval()
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(seed+rank);torch.cuda.manual_seed(seed+rank)
        metric=original_metrics.DistributedOriginalRQVAEMetrics(device,
            compute_inception_score=True,reference_stats_path=BASE/'inputs/imagenet_256_train.npz')
        local_count=50000//world+(rank<50000%world)
        for first in range(0,local_count,batch):
            count=min(batch,local_count-first)
            labels=(torch.arange(first,first+count,device=device)*world+rank)%1000
            codes=PROFILE.sample(model,aux,count,labels)
            for chunk in codes.split(8):
                images=PROFILE.decode(aux,chunk).float().mul(.5).add(.5).clamp(0,1)
                metric.update(images,real=False)
            if rank==0:
                write_json(OUT/'generation-progress.json',dict(epoch=epoch,
                    generated_on_rank=first+count,samples_on_rank=local_count,time=time.time()))
                if first==0 and not getattr(PROFILE,'recipe',{}).get('preview'):
                    from torchvision.utils import save_image
                    path=OUT/f'samples/epoch{epoch:03d}.png';path.parent.mkdir(exist_ok=True)
                    save_image(images[:8],path,nrow=4)
                    if run is not None:
                        import wandb
                        run.log({'samples/generated':wandb.Image(str(path)),'train/epoch':epoch})
        assert int(metric.fake_count)==local_count
        fid,is_mean,is_std=metric.compute(shuffle_for_inception=True)
        del metric
    model.train(was_training);torch.cuda.empty_cache()
    return dict(fid=fid,inception_score=is_mean,inception_score_std=is_std)


@torch.no_grad()
def reconstruction_check(aux,config,device,rank,world):
    dataset=TarImageNet(BASE/'data','val',create_transforms(config.dataset,split='val'))
    loader=DataLoader(Subset(dataset,range(rank,len(dataset),world)),batch_size=16,
        num_workers=4,pin_memory=True)
    metric=original_metrics.DistributedOriginalRQVAEMetrics(device)
    before=PROFILE.digest(aux)
    for index,(images,_) in enumerate(loader):
        images=images.to(device,non_blocking=True)
        recon,_,_=aux(images)
        metric.update(images.mul(.5).add(.5).clamp(0,1),real=True)
        metric.update(recon.mul(.5).add(.5).clamp(0,1),real=False)
        if rank==0 and index%50==0:
            print(json.dumps({'phase':'stage1_rfid','batches_on_rank':index+1}),flush=True)
    assert PROFILE.digest(aux)==before
    rfid,_,_=metric.compute()
    if rank==0:
        write_json(OUT/'stage1-rfid-verification.json',dict(passed=True,
            measured_validation_rfid=rfid,published_validation_rfid=4.73,
            real_images=50000,reconstructed_images=50000,codebooks_unchanged=True,
            checkpoint_sha256=sha256(BASE/'inputs/stage1-model.pt'),
            original_rqvae_backend=True))
        print(json.dumps({'stage1_validation_rfid':rfid}),flush=True)


def main(kind,base):
    global BASE,RUN,OUT,PROFILE
    BASE=Path(base)
    if kind == 'laser_compound128':
        from src.training.compound128_recipe import Compound128Recipe
        PROFILE = Compound128Recipe(BASE)
    elif kind == 'laser_continuous_compound':
        from src.training.continuous_compound_recipe import ContinuousCompoundRecipe
        PROFILE = ContinuousCompoundRecipe(BASE)
    elif kind == 'laser_native_support_order':
        from src.training.laser_native_support_order_recipe import NativeSupportOrderRecipe
        PROFILE = NativeSupportOrderRecipe(BASE)
    elif kind == 'laser_compound_replay':
        from src.training.laser_compound_replay_recipe import CompoundReplayRecipe
        PROFILE = CompoundReplayRecipe(BASE)
    elif kind == 'laser_signed_joint':
        from src.training.laser_signed_joint_recipe import SignedJointLaserRecipe
        PROFILE = SignedJointLaserRecipe(BASE)
    elif kind == 'laser_fine_joint':
        from src.training.laser_fine_joint_recipe import FineJointLaserRecipe
        PROFILE = FineJointLaserRecipe(BASE)
    elif kind == 'laser_joint':
        from src.training.laser_joint_recipe import JointLaserRecipe
        PROFILE = JointLaserRecipe(BASE)
    elif kind == 'laser_packed_integer':
        from src.training.laser_packed_integer_recipe import PackedIntegerLaserRecipe
        PROFILE = PackedIntegerLaserRecipe(BASE)
    elif kind == 'laser_catalog_integer':
        from src.training.laser_catalog_integer_recipe import CatalogIntegerLaserRecipe
        PROFILE = CatalogIntegerLaserRecipe(BASE)
    else:
        PROFILE=(RQRecipe if kind=='rq' else LaserRecipe)(BASE)
    RUN=PROFILE.run_id
    OUT=Path('/workspace/Projects/laser/outputs')/RUN
    if os.environ.get('LASER_RQ_SAFE_CHECKPOINTS') == '1':
        from runtime.rq_checkpoint_store import install
        install(checkpoint_io, BASE, OUT)
    original_metrics.INCEPTION_PATH=BASE/'upstream/rqvae/metrics/inception.py'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-updates',type=int,default=0)
    p.add_argument('--smoke-split',choices=('train','val'),default='val',
        help='Verified real-image split used for a disposable preflight')
    p.add_argument('--rfid-only',action='store_true')
    p.add_argument('--microbatch',type=int,default=PROFILE.microbatch)
    p.add_argument('--epochs',type=int,default=PROFILE.epochs)
    p.add_argument('--resume',type=Path)
    p.add_argument('--sampling-override',type=Path,
        help='Resume with a sampler-only change validated on matched FID50k')
    p.add_argument('--new-wandb-run',action='store_true',
        help='Create a separate W&B record while continuing a verified checkpoint')
    args=p.parse_args()
    if args.new_wandb_run and not args.resume:p.error('--new-wandb-run requires --resume')
    if args.sampling_override and not args.resume:p.error('--sampling-override requires --resume')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world>=1 and args.microbatch>0
    accumulation=math.ceil(2048/(world*args.microbatch))
    device=torch.device('cuda',local);torch.cuda.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group('nccl',timeout=timedelta(minutes=60))
    seed_all(261001)
    aux,stage1_config=PROFILE.tokenizer(device)
    if args.rfid_only:
        if kind!='rq':p.error('--rfid-only requires the native RQ-VAE recipe')
        reconstruction_check(aux,stage1_config,device,rank,world)
        dist.destroy_process_group();return
    phase='smoke' if args.smoke_updates else 'train'
    PROFILE.execution_phase=phase
    folder=Path(os.environ.get('LASER_PRIVATE_SMOKE_DIR',str(OUT/phase))) if phase=='smoke' else OUT/phase
    if phase=='train':
        if rank==0:local_training_logs(BASE,folder)
        dist.barrier()
    else:folder.mkdir(parents=True,exist_ok=True)
    initialization=getattr(PROFILE,'recipe',{}).get('initialization',{})
    if not args.resume and initialization.get('reset_stage2_seed_after_tokenizer'):
        seed_all(int(PROFILE.recipe.seed))
    model,optimizer,config=fresh_model(device)
    PROFILE.adapt(model,aux)
    optimizer_coverage=ensure_optimizer_covers_model(optimizer,model)
    parameters=sum(v.numel() for v in model.parameters())
    expected_hash=getattr(PROFILE,'recipe',{}).get('optimization',{}).get('probe_initial_weight_sha256')
    if not args.resume and expected_hash and rank==0:
        import hashlib
        initial_digest=hashlib.sha256()
        for name,value in model.state_dict().items():
            initial_digest.update(name.encode())
            initial_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        assert initial_digest.hexdigest()==expected_hash
        write_json(folder/'initial-weight-verification.json',dict(passed=True,
            initial_weight_sha256=expected_hash,same_initial_weights_as_lr_screen=True,
            stage2_checkpoint_loaded=False,global_step=0))
    assert len(model.body_transformer.blocks)==42 and len(model.head_transformer.blocks)==6
    assert model.config.embed_dim==1536 and model.config.vocab_size_cond==1000
    assert model.block_size==torch.Size(PROFILE.shape)
    dataset=TarImageNet(BASE/'data',args.smoke_split if args.smoke_updates else 'train',
        create_transforms(config.dataset,split='train'),seed=261001)
    # Smoke uses verified real images with identical global batch/accumulation.
    sampler=ExactGlobalBatchSampler(dataset,2048,world,rank,accumulation,261001)
    steps_per_epoch=626
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,steps_per_epoch,PROFILE.schedule_epochs)
    recipe = getattr(PROFILE, 'recipe', {})
    logging_settings = recipe.get('logging', {})
    training_dtype = recipe.get('execution', {}).get('training_autocast_dtype', 'float16')
    if training_dtype not in ('float16','bfloat16'):
        raise ValueError('Training autocast dtype must be float16 or bfloat16')
    scaler = torch.amp.GradScaler(
        'cuda', enabled=bool(config.experiment.amp) and training_dtype == 'float16',
    )
    completed_epoch=start_micro=global_step=0
    restored_best_fid=math.inf
    restored_evaluation=None
    sampling_change=None
    if args.resume:
        payload=torch.load(checkpoint_io._checkpoint_upload_source(args.resume.resolve()),
            map_location='cpu',weights_only=False,mmap=True)
        assert payload['run_id']==RUN and payload[PROFILE.checkpoint_key]==sha256(BASE/'inputs/stage1-model.pt')
        if args.sampling_override:
            from src.training.rq_sampling_override import validated_sampling_recipe
            sampling_change=json.loads(args.sampling_override.read_text())
            PROFILE.name=str(sampling_change.get('display_name',PROFILE.name))
            assert int(payload['global_step'])>=int(sampling_change['source_step'])
            adapted=validated_sampling_recipe(OmegaConf.to_container(PROFILE.recipe,resolve=True),
                payload['recipe'],sampling_change)
            PROFILE.recipe.sampling=OmegaConf.create(adapted['sampling'])
            # This adjusts only the in-memory metadata used by strict recipe
            # validation; serialized weights and optimizer state stay intact.
            payload['recipe']=adapted
        PROFILE.verify_resume(payload)
        model.load_state_dict(payload['state_dict'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler']);scaler.load_state_dict(payload['scaler'])
        completed_epoch=payload['next_epoch'];start_micro=payload['next_microbatch'];global_step=payload['global_step']
        restored_best_fid=payload.get('best_fid',math.inf)
        restored_evaluation=payload.get('evaluation')
        saved_rng=payload['rng_state_by_rank'][rank]
        random.setstate(saved_rng['python']);np.random.set_state(saved_rng['numpy'])
        torch.set_rng_state(saved_rng['cpu']);torch.cuda.set_rng_state(saved_rng['cuda'],device)
        if hasattr(PROFILE,'coefficient_generator'):
            PROFILE.coefficient_generator.set_state(saved_rng['coefficient'])
        del payload
    else:
        assert not list((folder/'checkpoints').glob('*.pt')), 'Fresh run refuses existing checkpoints'
    ddp=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True,bucket_cap_mb=100)
    rng_metadata={}
    rng_options=getattr(PROFILE,'recipe',{}).get('training_rng')
    if rng_options is not None:
        from src.training.rq_rng import POLICY,configure_training_rng,audit_training_rng
        if rng_options.policy != POLICY:
            raise ValueError(f'Unsupported training RNG policy: {rng_options.policy}')
        rng_metadata=configure_training_rng(int(rng_options.base_seed),rank,resuming=bool(args.resume))
        write_json(folder/f'rng-startup-rank{rank}.json',
            dict(rank=rank,**rng_metadata,**audit_training_rng(device)))
    if not args.resume:
        assert not optimizer.state
        assert optimizer.param_groups[0]['lr']==scheduler.get_last_lr()[0]
    initial=dict(fresh_stage2=not bool(args.resume),fresh_optimizer=not bool(optimizer.state),
        optimizer_coverage=optimizer_coverage,
        global_step=global_step,parameter_count=parameters,parameter_tensors=len(list(model.parameters())),
        world_size=world,microbatch=args.microbatch,accumulation_steps=accumulation,
        global_batch=2048,optimizer_updates_per_epoch=626,training_epochs=args.epochs,
        lr_schedule_epochs=PROFILE.schedule_epochs,**schedule_metadata(config,scheduler),
        training_autocast_dtype=training_dtype,grad_scaler_enabled=scaler.is_enabled(),
        pid=os.getpid(),**PROFILE.metadata(),**rng_metadata)
    write_json(folder/f'startup-rank{rank}.json',initial)
    run=None
    resume_logged_step=0
    if rank==0 and not args.smoke_updates and logging_settings.get('wandb', True):
        import wandb
        key_file = Path.home() / '.config/laser/imagenet-stage2-wandb.key'
        if not os.environ.get('WANDB_API_KEY') and key_file.is_file():
            os.environ['WANDB_API_KEY'] = key_file.read_text().strip()
        run=wandb.init(entity='helloimlixin-rutgers',project='laser',id=RUN,
            name=PROFILE.name,
            resume='must' if args.resume and not args.new_wandb_run else 'allow',dir=str(BASE/'wandb'),
            config=None if args.resume and not args.new_wandb_run else dict(initial,wandb_id=RUN,published_config=OmegaConf.to_container(config,resolve=True),
                **{PROFILE.checkpoint_key:sha256(BASE/'inputs/stage1-model.pt')},
                **PROFILE.checkpoint_extra(),
                fid_num_samples=50000,fid_real_split='train',fid_reference_images=1281167,
                fid_reference_sha256=sha256(BASE/'inputs/imagenet_256_train.npz'),
                **PROFILE.logging_sampling_settings()))
        if args.new_wandb_run:
            assert int(run.summary.get('train/global_step',0))<=global_step
            assert run.config[PROFILE.checkpoint_key]==sha256(BASE/'inputs/stage1-model.pt')
            run.summary['training/continued_from_checkpoint']=str(args.resume)
            run.summary['training/continued_from_step']=global_step
        if not args.resume:
            # A logging-only startup failure may leave an empty W&B run.
            # Reuse its record only when no production update was logged.
            assert int(run.summary.get('train/global_step',0))==0, 'Existing W&B run has training updates'
            assert run.config[PROFILE.checkpoint_key]==sha256(BASE/'inputs/stage1-model.pt')
            assert int(run.config['parameter_count'])==parameters
            run.summary['training/reused_empty_wandb_run']=bool(run.resumed)
        resume_logged_step=int(run.summary.get('train/global_step',0)) if args.resume and not args.new_wandb_run else 0
        run.define_metric('train/global_step');run.define_metric('train/epoch')
        run.define_metric('train/*',step_metric='train/global_step')
        if getattr(PROFILE,'recipe',{}).get('preview'):
            run.define_metric('samples/global_step')
            run.define_metric('samples/*',step_metric='samples/global_step')
        run.define_metric('eval/fid',step_metric='train/epoch',overwrite=True)
        run.define_metric('eval/inception_score',step_metric='train/epoch',summary='max,last')
        run.define_metric('eval/inception_score_std',step_metric='train/epoch')
        run.summary.update({'evaluation/generated_images':50000,'evaluation/reference_images':1281167,
            'evaluation/reference_split':'train','training/status':'running','training/stop_after_epochs':args.epochs,
            'training/auxiliary_loss_terms':0,'training/objective':PROFILE.metadata()['objective']})
        if sampling_change is not None:
            run.config.update(dict(native_sampling=PROFILE.metadata()['native_sampling'],
                recipe=OmegaConf.to_container(PROFILE.recipe,resolve=True),
                published_config=OmegaConf.to_container(config,resolve=True),
                **PROFILE.logging_sampling_settings()),allow_val_change=True)
            run.summary.update({'sampling/changed_at_update':global_step,
                'sampling/change':sampling_change.get('change','validated sampler-only update'),
                'sampling/validation_fid50k':sampling_change['fid'],
                'sampling/validation_baseline_fid50k':sampling_change['baseline_fid']})
            run.save(str(args.sampling_override),base_path=str(args.sampling_override.parent),policy='now')
        for path in (OUT/'reproducibility').glob('*'):
            if path.is_file():run.save(str(path),base_path=str(OUT),policy='now')
        write_json(OUT/'wandb.json',dict(url=run.url,id=RUN))
        if logging_settings.get('create_workspace', True):
            fid_workspace(run)
        if args.new_wandb_run and restored_evaluation is not None:
            run.log({'train/global_step':global_step,'train/epoch':global_step/steps_per_epoch,
                'eval/fid':restored_evaluation['fid'],
                'eval/inception_score':restored_evaluation['inception_score'],
                'eval/inception_score_std':restored_evaluation['inception_score_std']})
            run.summary['evaluation/initial_point_origin']='verified matched 50000-sample checkpoint evaluation'
    before=PROFILE.digest(aux)
    if sampling_change is not None:
        preview(model,aux,device,rank,world,global_step,run)
    preview_every = int(getattr(PROFILE,'recipe',{}).get('preview',{}).get('every_updates',0))
    writer=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    save_future=None
    best_fid=restored_best_fid

    def checkpoint(epoch,next_epoch,next_microbatch,evaluation=None):
        nonlocal save_future,best_fid
        rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                 cuda=torch.cuda.get_rng_state(device))
        if hasattr(PROFILE,'coefficient_generator'):
            rng['coefficient']=PROFILE.coefficient_generator.get_state()
        all_rng=[None]*world
        dist.all_gather_object(all_rng,rng)
        if rank==0:
            if save_future:save_future.result()
            better=evaluation is not None and evaluation['fid']<best_fid
            if better:best_fid=evaluation['fid']
            state=cpu_snapshot(dict(run_id=RUN,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),epoch=epoch,global_step=global_step,
                next_epoch=next_epoch,next_microbatch=next_microbatch,rng_state_by_rank=all_rng,
                config=OmegaConf.to_container(config,resolve=True),runtime=initial,
                **{PROFILE.checkpoint_key:sha256(BASE/'inputs/stage1-model.pt')},
                **PROFILE.checkpoint_extra(),
                evaluation=evaluation,best_fid=best_fid))
            def save():
                # Persist checkpoints independently of the public live-log
                # symlink, which an object-backed mount may materialize.
                directory=OUT/'persistent-train/checkpoints' if phase=='train' else folder/'checkpoints'
                target=directory/'last.pt'
                checkpoint_io.atomic_torch_save(state,target)
                if better:
                    # Both aliases point to the same committed immutable payload.
                    best=directory/'best.pt'
                    if hasattr(checkpoint_io, 'link_committed_checkpoint'):
                        checkpoint_io.link_committed_checkpoint(target,best)
                    else:
                        temporary=best.with_suffix('.linktmp')
                        temporary.unlink(missing_ok=True)
                        os.symlink(os.readlink(target),temporary);temporary.replace(best)
                write_json(folder/'checkpoint.json',dict(epoch=epoch,global_step=state['global_step'],
                    path=str(target),bytes=target.stat().st_size,best_fid=best_fid))
                if phase=='train':
                    write_json(OUT/'checkpoint.json',dict(epoch=epoch,global_step=state['global_step'],
                        path=str(OUT/'checkpoints/last.pt'),bytes=target.stat().st_size,best_fid=best_fid))
                if run is not None and os.environ.get('LASER_WANDB_CHECKPOINT_UPLOADS', '1') != '0':
                    uploads=BASE/'uploads/checkpoints';uploads.mkdir(parents=True,exist_ok=True)
                    for name in (['last.pt','best.pt'] if better else ['last.pt']):
                        source=checkpoint_io._checkpoint_upload_source(directory/name)
                        alias=uploads/name;temporary=alias.with_suffix('.tmp');temporary.unlink(missing_ok=True)
                        os.link(source,temporary);temporary.replace(alias)
                        run.save(str(alias),base_path=str(BASE/'uploads'),policy='now')
                if hasattr(checkpoint_io, 'prune_committed_checkpoint_cache'):
                    checkpoint_io.prune_committed_checkpoint_cache()
            save_future=writer.submit(save)
        dist.barrier()

    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=8,pin_memory=True,
        persistent_workers=True,prefetch_factor=2)
    optimizer.zero_grad(set_to_none=True)
    start=time.monotonic();update_loss=torch.zeros((),device=device);update_images=0
    verified_targets=False
    micro_diagnostics={}
    replay=ReplayWindow(device,generators=[PROFILE.coefficient_generator]
                        if hasattr(PROFILE,'coefficient_generator') else ())

    def backward_microbatch(images,labels,absolute):
        nonlocal verified_targets
        images=images.to(device,non_blocking=True);labels=labels.to(device,non_blocking=True)
        sync=(absolute+1)%accumulation==0
        with nullcontext() if sync else ddp.no_sync():
            def audit(targets,codes):
                nonlocal verified_targets
                if not verified_targets:
                    write_json(folder/f'targets-rank{rank}.json',PROFILE.verify_targets(aux,targets,codes))
                    verified_targets=True
            loss=training_backward(PROFILE,ddp,aux,images,labels,scaler,
                world*len(images)/sampler.global_count(absolute),on_targets=audit)
            if hasattr(PROFILE,'last_loss_metrics'):
                # A recovered AMP replay overwrites the same microbatch key.
                micro_diagnostics[absolute]=(len(images),PROFILE.last_loss_metrics)
        return loss,len(images)

    def overflow_event(event):
        row=dict(event,global_step=global_step+1,epoch=epoch+1,time=time.time())
        write_json(folder/f'amp-overflow-rank{rank}.json',row)
        if rank==0:
            print(json.dumps(dict(amp_overflow_replaying_same_batch=row)),flush=True)
            with (folder/'amp-overflows.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')

    for epoch in range(completed_epoch,args.epochs):
        sampler.set_epoch(epoch);sampler.set_start_batch(start_micro)
        model.train()
        for relative,(images,labels) in enumerate(loader):
            absolute=start_micro+relative
            if absolute%accumulation==0:
                replay.begin();micro_diagnostics.clear()
            replay.append(images,labels,absolute)
            loss_value,image_count=backward_microbatch(images,labels,absolute)
            update_loss.add_(loss_value*image_count);update_images+=image_count
            del images,labels,loss_value
            sync=(absolute+1)%accumulation==0
            if sync:
                norm,overflow_retries=recoverable_amp_step(model.parameters(),optimizer,scaler,
                    max_norm=float(config.optimizer.max_gn),
                    replay_backward=lambda: replay.replay(backward_microbatch),on_overflow=overflow_event)
                scheduler.step();optimizer.zero_grad(set_to_none=True);global_step+=1
                stats=torch.stack([update_loss,torch.tensor(update_images,device=device)])
                dist.all_reduce(stats)
                metrics={'train/global_step':global_step,'train/epoch':epoch,
                    'train/loss':float(stats[0]/stats[1]),'train/lr':optimizer.param_groups[0]['lr'],
                    'train/grad_norm':float(norm),'train/amp_scale':scaler.get_scale(),
                    'train/amp_overflow_retries':overflow_retries,'train/images_per_second':float(stats[1])/(time.monotonic()-start)}
                if micro_diagnostics:
                    names=tuple(next(iter(micro_diagnostics.values()))[1])
                    diagnostic_sums=torch.stack([sum(n*values[name] for n,values in micro_diagnostics.values())
                                                  for name in names])
                    dist.all_reduce(diagnostic_sums)
                    metrics.update({f'train/{name}':float(value/stats[1])
                                    for name,value in zip(names,diagnostic_sums)})
                age=int(next(iter(optimizer.state.values()))['step'])
                assert age==global_step
                write_json(folder/f'progress-rank{rank}.json',dict(metrics,adam_age=age,pid=os.getpid(),
                    peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,time=time.time()))
                if rank==0 and (global_step<=3 or global_step%10==0):
                    print(json.dumps(metrics),flush=True)
                    if run and global_step>resume_logged_step:run.log(metrics)
                    with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
                update_loss.zero_();update_images=0;start=time.monotonic()
                if args.smoke_updates and global_step>=args.smoke_updates:
                    assert PROFILE.digest(aux)==before
                    if preview_every:
                        from src.training.rq_sample_panels import PREVIEW_IMAGES
                        cpu_rng=torch.get_rng_state().clone()
                        cuda_rng=torch.cuda.get_rng_state(device).clone()
                        preview(model,aux,device,rank,world,global_step,None)
                        assert torch.equal(cpu_rng,torch.get_rng_state())
                        assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
                        write_json(folder/f'preview-verification-rank{rank}.json',dict(
                            passed=True,global_step=global_step,
                            images=PREVIEW_IMAGES if PROFILE.recipe.preview.get('designated_classes') else 32,
                            training_rng_unchanged=True,optimizer_unchanged=True))
                    checkpoint(epoch,epoch,absolute+1)
                    if rank==0:
                        save_future.result()
                        write_json(folder/'verification.json',dict(passed=True,global_step=global_step,
                            optimizer_state_count=len(optimizer.state),all_adam_steps=age,
                            frozen_tokenizer_unchanged=True,initialization=initial))
                    writer.shutdown();dist.destroy_process_group();return
                first_preview=int(getattr(PROFILE,'recipe',{}).get('preview',{}).get('first_update',0))
                if preview_every and (global_step%preview_every==0 or global_step==first_preview):
                    preview(model,aux,device,rank,world,global_step,run)
                checkpoint_every=int(getattr(PROFILE,'execution_options',{}).get('checkpoint_every_updates',
                    getattr(PROFILE,'recipe',{}).get('checkpoint',{}).get('every_updates',0)))
                first_checkpoint=(global_step==1 and getattr(PROFILE,'recipe',{}).get('checkpoint',{}).get('at_first_update',False))
                if first_checkpoint or (checkpoint_every and global_step%checkpoint_every==0):
                    checkpoint(epoch,epoch,absolute+1)
        start_micro=0
        evaluation=None
        extra_evaluation_epochs=getattr(PROFILE,'recipe',{}).get('evaluation',{}).get('extra_epochs',[])
        if epoch==0 or (epoch+1)%int(config.experiment.test_freq)==0 or epoch+1 in extra_evaluation_epochs:
            if preview_every and global_step%preview_every:
                preview(model,aux,device,rank,world,global_step,run)
            evaluation=generation(model,aux,device,rank,world,epoch+1,run)
            assert PROFILE.digest(aux)==before
            if rank==0:
                row={'train/epoch':epoch+1,'train/global_step':global_step,
                    'eval/fid':evaluation['fid'],'eval/inception_score':evaluation['inception_score'],
                    'eval/inception_score_std':evaluation['inception_score_std']}
                write_json(OUT/f'evaluations/epoch{epoch+1:03d}.json',dict(row,generated_images=50000,
                    real_images=1281167,real_split='train',rejection_sampling=False))
                print(json.dumps(row),flush=True)
                if run:run.log(row);run.save(str(OUT/f'evaluations/epoch{epoch+1:03d}.json'),base_path=str(OUT),policy='now')
        if evaluation is not None or (epoch+1)%int(config.experiment.save_ckpt_freq)==0:
            checkpoint(epoch+1,epoch+1,0,evaluation)
        start=time.monotonic()
    if rank==0:
        if save_future:save_future.result()
        write_json(OUT/'completed.json',dict(epochs=args.epochs,global_step=global_step,best_fid=best_fid,time=time.time()))
        if run:run.summary['training/status']='completed';run.finish()
    writer.shutdown();dist.barrier();dist.destroy_process_group()

