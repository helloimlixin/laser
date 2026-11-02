"""Hardware and I/O adaptations for the evaluated epoch98 ImageNet resume."""
import hashlib
import json
import os
from pathlib import Path
import time
import types
import uuid

import torch


def install(g):
    base, out, training, io = g['BASE'], g['OUT'], g['training'], g['checkpoint_io']
    benchmark = os.environ.get('LASER_THROUGHPUT_BENCHMARK') == '1'
    from imagenet_tar import TarImageNet
    from compound_sampling import Policy, sample
    policy = Policy(**json.loads((base/'selected-sampling-policy.json').read_text())['policy'])
    policy.validate()

    def source_images(dataset, root, transform, *, split='train'):
        assert dataset == 'imagenet' and split == 'train'
        if benchmark:
            class Synthetic(torch.utils.data.Dataset):
                classes = [str(i) for i in range(1000)]
                def __init__(self):
                    self.images = torch.randn(16,3,256,256,
                        generator=torch.Generator().manual_seed(261001)).share_memory_()
                def __len__(self): return 1281167
                def __getitem__(self, index): return self.images[index%16], index%1000
            return Synthetic()
        ready = json.loads((base/'data/ready.json').read_text())
        assert ready['ready'] and ready['training_images'] == 1281167
        return TarImageNet(base/'data', split, transform, seed=261001)
    training.source_image_dataset = source_images
    native_sampler_epoch = training.ExactGlobalBatchSampler.set_epoch
    def sampler_epoch(self, epoch):
        if hasattr(self, 'dataset') and hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
        return native_sampler_epoch(self, epoch)
    native_sampler_init = training.ExactGlobalBatchSampler.__init__
    def sampler_init(self, dataset, *args, **kwargs):
        self.dataset = dataset
        return native_sampler_init(self, dataset, *args, **kwargs)
    training.ExactGlobalBatchSampler.__init__ = sampler_init
    training.ExactGlobalBatchSampler.set_epoch = sampler_epoch
    native_loader = training.DataLoader
    def loader(*args, **kwargs):
        kwargs.update(num_workers=int(os.environ.get('LASER_DATA_WORKERS','12')),prefetch_factor=3)
        return native_loader(*args, **kwargs)
    training.DataLoader = loader

    native_factory = training.build_model
    def factory(*args, **kwargs):
        model = native_factory(*args, **kwargs)
        def sampler(self, batch_size, aux, cond=None, **unused):
            return sample(self, aux, batch_size, cond, policy)
        model.sample_compound = types.MethodType(sampler, model)
        return model
    training.build_model = factory

    from cpu_checkpoint_snapshot import CPUCheckpointSnapshotter
    from src.training.checkpoint_upload import CheckpointUploader
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    snapshotter = CPUCheckpointSnapshotter()
    uploader = None
    committed = {}
    cache = base/'native-checkpoint-cache'
    cache.mkdir(exist_ok=True)
    fallback = io._checkpoint_upload_source

    def write(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp = path.with_name(path.name+'.tmp')
        tmp.write_text(json.dumps(value,indent=2,default=str)+'\n')
        tmp.replace(path)

    def alias(payload, target):
        tmp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.link')
        tmp.symlink_to(os.path.relpath(payload,target.parent));tmp.replace(target)

    def persist(serialized, target):
        import shutil
        identity=uuid.uuid4().hex
        folder=target.parent/'.checkpoint-data';folder.mkdir(parents=True,exist_ok=True)
        payload=folder/(target.stem+'-'+identity+'.pt')
        local=cache/(identity+'.pt')
        size=serialized.stat().st_size
        with serialized.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        for attempt in range(3):
            try:
                shutil.copyfile(serialized,payload)
                assert payload.stat().st_size==size
                with serialized.open('rb') as a,payload.open('rb') as b:
                    assert a.read(64)==b.read(64)
                    a.seek(-64,2);b.seek(-64,2)
                    assert a.read(64)==b.read(64)
                break
            except (OSError,AssertionError):
                payload.unlink(missing_ok=True)
                if attempt==2:raise
                time.sleep(2*(attempt+1))
        os.link(serialized,local)
        metadata=torch.load(serialized,map_location='cpu',mmap=True,weights_only=False)
        row=dict(local=str(local),persistent=str(payload),bytes=size,sha256=sha,
            global_step=int(metadata['global_step']),epoch=int(metadata['epoch']),
            public_alias=str(target),time=time.time())
        del metadata
        write(folder/(identity+'.json'),row)
        committed[str(target)]=row
        alias(payload,target)
        serialized.unlink()
        write(out/'last-local-save.json',row)
    io._persist_serialized_checkpoint = persist
    def upload_source(path):
        row=committed.get(str(path))
        return Path(row['local']) if row else fallback(Path(path))
    io._checkpoint_upload_source = upload_source
    folder=out/'train/checkpoints/.checkpoint-data'
    for p in folder.glob('*.json'):
        row=json.loads(p.read_text())
        if Path(row['local']).is_file():
            prev=committed.get(row['public_alias'])
            if prev is None or row['time']>prev['time']:
                committed[row['public_alias']]=row

    def upload(*args, **kwargs):
        if uploader is None:return
        slots=base/'upload-slots';slots.mkdir(exist_ok=True)
        sources=[('last.pt',kwargs['last_checkpoint'])]
        for kind in ('fid','is'):
            prefix='best_fid' if kind=='fid' else 'best_is'
            candidates=[(name,row) for name,row in committed.items() if Path(name).name.startswith(prefix)]
            if candidates:
                # The training loop retains exactly one winner of each type.
                candidates=[item for item in candidates if Path(item[0]).exists()]
                if candidates:
                    name,row=max(candidates,key=lambda item:item[1]['time'])
                    sources.append((f'best-{kind}-01.pt',Path(name)))
        paths=[]
        for name,source in sources:
            row=committed.get(str(source))
            if row is None:continue
            io._replace_hard_link(Path(row['local']),slots/name)
            paths.append(slots/name)
        if paths:uploader.submit(paths, max(committed[str(p)]['epoch'] for _,p in sources if str(p) in committed))
    training.upload_selected_checkpoint_files = upload

    def save(payload, target):
        if benchmark:
            g['record']('benchmark-endpoint.json',dict(global_step=payload['global_step'],updates_discarded=True))
            return
        payload=dict(payload,selected_sampling_policy=policy.__dict__,continuation_target_epoch=100)
        g['WRITER'].wait()
        frozen=snapshotter.freeze(payload)
        def commit():
            io.atomic_torch_save(frozen,target,collect_garbage=False)
            upload(last_checkpoint=target)
        g['WRITER'].submit(commit)
    training.atomic_torch_save = save
    def snapshot(source, target):
        if benchmark:return
        def commit():
            row=committed[str(source)]
            committed[str(target)]=dict(row)
            alias(Path(row['persistent']),Path(target))
            upload(last_checkpoint=out/'train/checkpoints/last.pt')
        g['WRITER'].append(commit)
    training.snapshot_checkpoint = snapshot

    def wandb_init(*args, **kwargs):
        nonlocal uploader
        kwargs['resume']='allow' if not benchmark else None
        run=g['native_wandb_init'](*args,**kwargs)
        g['REAL_RUN']=run
        run.config.update(dict(stage2_initialization='full_checkpoint_resume',
            source_run='imagenet-rfid421-ffhq-q995-clipped-sigma42p65-scratch-4h200-20261009',
            source_global_step=61348,source_completed_epochs=98,target_epoch=100,
            source_train_fid_two_seed_mean=15.753511832304127,
            source_checkpoint_sha256='a3392c62eefe01ea378d09a1f4a671574d73365e1e589a1d59344f8e92e4ca61',
            selected_sampling_policy=policy.__dict__,scheduler_state_preserved=True,
            optimizer_state_preserved=True,objective_preserved=True,
            training_precision='bfloat16',frozen_encoder_precision='float32',
            microbatch=g['ARGS'].batch_size,accumulation=int(os.environ['LASER_ACCUMULATION']),
            global_batch=2048,execution_hardware='4 NVIDIA H200 NVLink',
            trajectory_bitwise_identical=False),allow_val_change=True)
        g['original_logging'].define_metrics(run)
        run.summary['execution/target_epoch']=100
        run.summary['execution/state']='benchmark' if benchmark else 'training'
        run.summary['source/fid_full_training_two_seed_mean']=15.753511832304127
        if not benchmark and g['ARGS'].wandb_mode=='online':
            uploader=CheckpointUploader(base/'upload-snapshots',VerifiedCloudUpload(
                'helloimlixin-rutgers/laser/'+run.id,out/'cloud-checkpoint-receipt.json'),immutable_sources=True)
        write(out/'wandb.json',dict(id=run.id,url=run.url))
        g['record']('wandb.json',dict(id=run.id,url=run.url))
        wrapped=g['LoggedRun'](run)
        native_finish=wrapped.finish
        def finish(*args,**kwargs):
            g['WRITER'].wait()
            if uploader is not None:uploader.close()
            final=g['CURRENT_STEP']
            if not benchmark:
                run.summary['execution/final_step']=final
                run.summary['execution/completed_epoch']=final/626
                run.summary['continuation/target_reached']=final==62600
                run.summary['execution/state']='completed' if final==62600 else 'stopped_resumable'
                write(out/'training-exit.json',dict(global_step=final,target_step=62600,
                    completed=final==62600,time=time.time()))
            return native_finish(*args,**kwargs)
        wrapped.finish=finish
        return wrapped
    g['wandb'].init = wandb_init

    native_step=torch.optim.AdamW.step
    times=[]
    last=None
    def step(optimizer,*args,**kwargs):
        nonlocal last
        result=native_step(optimizer,*args,**kwargs)
        now=time.monotonic()
        if last is not None:times.append(now-last)
        last=now
        if g['updates'] in (1,2,20):
            tensors=[p for group in optimizer.param_groups for p in group['params']]
            tensors += [s[k] for s in optimizer.state.values() for k in ('exp_avg','exp_avg_sq')]
            assert bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())
            g['record']('finite-step'+str(g['updates'])+'-rank'+os.environ['RANK']+'.json',dict(
                passed=True,global_step=g['CURRENT_STEP'],all_weights_and_moments_finite=True))
        if os.environ['RANK']=='0' and times:
            recent=times[-5:]
            row=dict(global_step=g['CURRENT_STEP'],updates=g['updates'],
                images_per_second=2048*len(recent)/sum(recent),step_seconds=recent,
                microbatch=g['ARGS'].batch_size,accumulation=int(os.environ['LASER_ACCUMULATION']),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,time=time.time())
            g['record']('throughput.json',row)
        return result
    torch.optim.AdamW.step=step
