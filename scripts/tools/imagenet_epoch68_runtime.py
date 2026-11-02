"""Resume the measured ImageNet epoch 68 branch with a zero-floor FID schedule."""
import copy
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist

from src.training.fid_adaptive_schedule import FidAdaptiveSchedule

ANCHOR_STEP = 42568
TOTAL_STEPS = 62600
SOURCE_FID = 16.926755510053226
SOURCE_LR = 0.000116043301255249
ACTIVE_SCHEDULER = None


def continuation_state(saved, completed_steps):
    """Change the LR policy once, at the scored source; never reset Adam."""
    policy = dict(initial_lr=SOURCE_LR, min_lr=0., total_steps=TOTAL_STEPS,
        baseline_fid=SOURCE_FID, patience=3, min_delta=.05, factor=.5, cooldown=1,
        decay_start_step=ANCHOR_STEP, decay_steps=TOTAL_STEPS-ANCHOR_STEP)
    if saved.get('kind') == 'fid-adaptive-cosine-v1':
        if saved.get('policy') != policy or saved.get('last_epoch') != completed_steps:
            raise ValueError('Continuation policy or checkpoint cursor changed')
        return copy.deepcopy(saved)
    if (completed_steps != ANCHOR_STEP or saved.get('last_epoch') != ANCHOR_STEP
            or saved.get('T_max') != TOTAL_STEPS or saved.get('eta_min') != 0.
            or saved.get('base_lrs') != [.0005]
            or len(saved.get('_last_lr', [])) != 1
            or not all(math.isclose(v, SOURCE_LR, rel_tol=1e-10)
                       for v in saved.get('_last_lr', []))):
        raise ValueError('Expected the exact scored epoch68 cosine state')
    return dict(kind='fid-adaptive-cosine-v1', policy=policy,
        last_epoch=ANCHOR_STEP, multiplier=1., best=SOURCE_FID, bad_epochs=0,
        cooldown_remaining=0, reductions=0, last_observation_step=ANCHOR_STEP,
        last_fid=SOURCE_FID)


def create_scheduler(optimizer, *, initial_lr, min_lr, total_steps,
                     completed_steps=0, state_dict=None):
    global ACTIVE_SCHEDULER
    if min_lr != 0. or total_steps != TOTAL_STEPS or state_dict is None:
        raise ValueError('Continuation must end at epoch100 with LR exactly zero')
    state = continuation_state(state_dict, completed_steps)
    expected = FidAdaptiveSchedule.lr_at_step(state['policy'], completed_steps, state['multiplier'])
    if any(not math.isclose(g['lr'], expected, rel_tol=1e-10, abs_tol=1e-15)
           for g in optimizer.param_groups):
        raise ValueError('Saved optimizer and continuation schedule disagree')
    ACTIVE_SCHEDULER = FidAdaptiveSchedule(optimizer, **state['policy'],
        completed_steps=completed_steps, state_dict=state)
    return ACTIVE_SCHEDULER


def install(namespace):
    """Extend the frozen source runtime without changing its model or objective."""
    training = namespace['training']
    base, out, phase = (namespace[k] for k in ('BASE', 'OUT', 'PHASE'))
    record = namespace['record']
    from src.training import portable_imagenet_recovery as portable
    from src.training.rq_imagenet_runtime import TarImageNet
    training.validate_resume_layout = portable.validate_layout
    training.restore_rank_rng_state = portable.restore_rng
    training.create_cosine_lr_scheduler = create_scheduler

    def images(dataset, root, transform, *, split='train'):
        ready = json.loads((root/'ready.json').read_text())
        assert ready['ready'] and ready['training_images'] == 1281167
        result = TarImageNet(root, split, transform, seed=261001)
        assert len(result) == (1281167 if split == 'train' else 50000)
        assert len(result.classes) == 1000
        return result
    training.source_image_dataset = images

    native_load = torch.load
    def load(source, *args, **kwargs):
        value = native_load(source, *args, **kwargs)
        args_obj = namespace.get('ARGS')
        if (args_obj is not None and isinstance(source, (str, os.PathLike))
                and Path(source) == args_obj.resume_checkpoint):
            value = portable.remap_payload(value, world=int(os.environ['WORLD_SIZE']),
                accumulation=int(os.environ['LASER_ACCUMULATION']),
                checkpoint_dir=out/'train/checkpoints', reference=args_obj.fid_reference_stats,
                reference_sha256=os.environ['LASER_REFERENCE_SHA256'])
            ages = sorted({int(v['step']) for v in value['optimizer']['state'].values()})
            assert ages == [value['global_step']]
            assert len(value['optimizer']['state']) == 837
            namespace['CURRENT_STEP'] = value['global_step']
            record('recovery-rank'+os.environ['RANK']+'.json', dict(passed=True,
                epoch=value['epoch'], step=value['global_step'], adam_steps=ages,
                adam_parameters=837, cursor=value['config']['portable_resume_cursor'],
                saved_lr=value['optimizer']['param_groups'][0]['lr']))
        return value
    torch.load = load

    # All ranks observe the same all-reduced metric before the scheduler is saved.
    native_fid = training.evaluate_generation_metrics
    def compute_fid(*args, **kwargs):
        result = native_fid(*args, **kwargs)
        fid = result[0] if isinstance(result, tuple) else result
        if isinstance(fid, dict):
            fid = fid.get('fid', fid.get('val/fid'))
        if ACTIVE_SCHEDULER is None:
            raise RuntimeError('FID evaluation arrived before scheduler restoration')
        decision = ACTIVE_SCHEDULER.observe(float(fid))
        if dist.get_rank() == 0 and decision is not None:
            record('lr-decision-step'+str(ACTIVE_SCHEDULER.last_epoch)+'.json', decision)
            run = namespace.get('REAL_RUN')
            if run is not None:
                run.log({'train/global_step':ACTIVE_SCHEDULER.last_epoch,
                    'schedule/lr':decision['lr_after'], 'schedule/reductions':decision['reductions'],
                    'schedule/multiplier':decision['multiplier']})
        return result
    training.evaluate_generation_metrics = compute_fid

    from src.training import k4_checkpoint_io as checkpoint_io
    import chunked_checkpoint_io
    chunked_checkpoint_io.LOCAL = base/'chunk-cache'
    chunked_checkpoint_io.install(checkpoint_io)

    # Verify cloud digests in a background queue. Every best winner is retained;
    # pending last-only uploads coalesce while GPU training continues.
    queue = None
    if int(os.environ['RANK']) == 0:
        import hashlib
        import tempfile
        from src.training.checkpoint_upload_queue import CheckpointFile, CheckpointUploadQueue
        from verified_wandb_checkpoint_upload import VerifiedCloudUpload
        cloud = VerifiedCloudUpload('helloimlixin-rutgers/laser/'+os.environ['LASER_ONLINE_RUN'],
            base/'cloud-checkpoint-receipt.json')
        def publish(item, slots):
            with tempfile.TemporaryDirectory(dir=base/'checkpoint-upload-cache') as temporary:
                paths=[]
                for slot in slots:
                    p=Path(temporary)/slot
                    os.link(item.path,p)
                    paths.append(p)
                cloud(paths,item.epoch)
        queue=CheckpointUploadQueue(publish,retry_delay=5,max_retry_delay=120)
        queued_sources={}

        def upload(run, *, last_checkpoint, best_fid, upload_dir, best_inception=None, **kwargs):
            if run is None:return []
            namespace['WRITER'].wait()
            queue.check()
            selected=[('last.pt',Path(last_checkpoint))]
            selected += [(f'best-fid-{i:02d}.pt',Path(p)) for i,(_,p) in enumerate(best_fid,1)]
            selected += [(f'best-is-{i:02d}.pt',Path(p)) for i,(_,p) in enumerate(best_inception or [],1)]
            groups={}
            for slot,path in selected:
                local=checkpoint_io._checkpoint_upload_source(path.resolve())
                if queued_sources.get(slot)==str(local):continue
                queued_sources[slot]=str(local)
                groups.setdefault(str(local),[]).append(slot)
            for source,slots in groups.items():
                # Pin the already serialized local inode; training may advance.
                fd,name=tempfile.mkstemp(suffix='.pt',dir=base/'checkpoint-upload-cache')
                os.close(fd);Path(name).unlink();os.link(source,name)
                queue.submit(CheckpointFile(path=Path(name),step=int(namespace['CURRENT_STEP']),
                    epoch=int(namespace['CURRENT_STEP'])//626,size=Path(name).stat().st_size,
                    md5='',slots=tuple(slots)))
            return selected
        training.upload_selected_checkpoint_files=upload
        original_finish=namespace['LoggedRun'].finish
        def finish(self,*args,**kwargs):
            namespace['WRITER'].wait()
            queue.close()
            self.run.summary['continuation/target_reached']=namespace['CURRENT_STEP']==TOTAL_STEPS
            self.run.summary['continuation/final_step']=namespace['CURRENT_STEP']
            return original_finish(self,*args,**kwargs)
        namespace['LoggedRun'].finish=finish

    import wandb
    original_init = namespace['native_wandb_init']
    def wandb_init(*args, **kwargs):
        kwargs['resume'] = 'allow'
        run = original_init(*args, **kwargs)
        namespace['REAL_RUN'] = run
        namespace['original_logging'].define_metrics(run)
        run.config.update(dict(resume_source_epoch=68, resume_source_global_step=ANCHOR_STEP,
            resume_source_fid=SOURCE_FID, parent_run=os.environ['LASER_PARENT_RUN'],
            saved_adam_retained=True, target_epoch=100, target_global_step=TOTAL_STEPS,
            continuation_lr='saved LR, remaining-interval cosine to zero; FID plateau factor0.5/patience3/cooldown1',
            preview_classes=[name for name, _ in CLASSES], sampler=SAMPLER,
            hardware='8 x RTX PRO 6000 Blackwell 96GB', global_batch=2048,
            microbatch=128, accumulation=2), allow_val_change=True)
        run.summary['continuation/target_reached'] = False
        for name in ('plan.json','checkpoint-verification.json','train.yaml','runtime-source.tar.gz'):
            p = base/name if name in ('runtime-source.tar.gz','train.yaml') else base/'metadata'/name
            if p.is_file():
                import shutil
                staging=base/'online-metadata'/str(os.getpid())
                staging.mkdir(parents=True,exist_ok=True)
                local=staging/name
                shutil.copyfile(p,local)
                run.save(str(local),base_path=str(staging),policy='now')
        record('wandb.json', dict(id=run.id,url=run.url,online=run.settings.mode=='online'))
        return namespace['LoggedRun'](run)
    wandb.init = wandb_init
    training.sample_class_grid = sample_class_grid


CLASSES = (('cheeseburger',933),('teapot',849),('samoyed',258),('bald eagle',22),
           ('red fox',277),('hotpot',926),('wombat',106),('timber wolf',269))
SAMPLER = dict(name='scored-epoch68-native-ancestral-nucleus',atom_temperature=.9,
    atom_top_k=0,atom_top_p=.9,coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.85,
    support_mask_before_feedback=True,rejection_sampling=False)


@torch.no_grad()
def sample_class_grid(model, aux, class_names, output_dir, step, wb=None, **kwargs):
    from sample_panels import save_panels
    device = next(model.parameters()).device
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    positions = torch.arange(rank,512,world)
    labels = torch.tensor([label for _, label in CLASSES],device=device).repeat_interleave(64)[rank::world]
    was_training = model.training
    model.eval()
    images = []
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(261001+rank)
            for batch in labels.split(32):
                atoms, coefficients = model.sample_compound(len(batch),aux,cond=batch,
                    atom_temperature=.9,atom_top_k=aux.num_atoms,atom_top_p=.9,
                    coeff_temperature=1.,coeff_top_k=0,coeff_top_p=.85,amp=True)
                for a,c in zip(atoms.split(8),coefficients.split(8)):
                    decoded = aux.decode_compound(a,c).float().mul(.5).add(.5).clamp(0,1)
                    assert torch.isfinite(decoded).all()
                    images.append(decoded.mul(255).round().to(torch.uint8).cpu())
        local = torch.cat(images)
        if world > 1:
            gathered = [None]*world
            dist.all_gather_object(gathered,(positions,local))
        else:
            gathered = [(positions,local)]
        if rank != 0:
            return None
        ordered = torch.empty(512,3,256,256,dtype=torch.uint8)
        for ids,values in gathered:
            ordered[ids]=values
        paths = save_panels(ordered,Path(output_dir)/'samples',step)
        metadata = dict(step=step,classes=[dict(name=n,id=i) for n,i in CLASSES],
            samples_per_class=64,overview_per_class=10,seed=261001,rank_seed='seed+rank',
            world_size=world,sampler=SAMPLER)
        (Path(output_dir)/'samples'/f'step{step:06d}-panels.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if wb is not None:
            import wandb
            wb.log(dict({'train/global_step':step},**{key:wandb.Image(str(path),
                caption=f'{key}; step {step}; fixed seed; measured epoch68 ancestral sampler')
                for key,path in paths.items()}))
        return paths['samples/fixed']
    finally:
        model.train(was_training)
        torch.cuda.empty_cache()
