#!/usr/bin/env python3
"""Supervise an isolated two-arm beam16 consistency pilot and matched FID50k.

The active production run is never stopped or edited. Both pilot arms restore
one pinned checkpoint, use the same images, Adam history, cosine clock, batch,
precision, RNG state, and sampler. Only consistency regularization differs.
"""
import argparse
from contextlib import nullcontext
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
import zipfile


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def configure(config):
    local = Path(config['local'])
    frozen = local/'frozen'
    os.environ.update(RQ_UPSTREAM_ROOT=str(frozen/'upstream'), TORCH_HOME=str(frozen/'torch-cache'),
        WANDB_BASE_URL='https://api.forge.coreweave.com', WANDB_DISABLE_GIT='true',
        LASER_FUSED_ADAMW='1', PYTHONUNBUFFERED='1')
    os.environ['WANDB_API_KEY'] = Path('/root/.config/laser/wandb.key').read_text().strip()
    sys.path[:0] = [str(local/'support'), str(frozen/'source'), str(frozen), str(frozen/'upstream')]
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = True
    from runtime.precision import install_bf16_training
    install_bf16_training()
    from src import rqvae_metrics as metrics
    metrics.INCEPTION_PATH = frozen/'upstream/rqvae/metrics/inception.py'
    return local, frozen


def restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'], device)


def rng_state(device):
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))


def build(config, arm, resume=None):
    local, frozen = configure(config)
    import torch
    from omegaconf import OmegaConf
    from beam_consistency import BeamConsistencyHeads, extend_adam_payload, remap_data_cursor
    from src.training.laser_joint_recipe import JointLaserRecipe
    from rqvae.optimizer.scheduler import create_scheduler
    from runtime.training_acceleration import configure_optimizer
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(config['memory_gib']*2**30 /
        torch.cuda.get_device_properties(device).total_memory, device)
    profile = JointLaserRecipe(frozen)
    path = Path(resume) if resume else local/'source-checkpoint.pt'
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    plan = read(local/'plan.json')
    if resume:
        assert payload['consistency_trial']['source_checkpoint_sha256'] == plan['source_checkpoint_sha256']
        assert payload['consistency_trial']['arm'] == arm
        assert payload['consistency_trial']['configuration'] == config
        assert payload['recipe'] == OmegaConf.to_container(profile.recipe, resolve=True)
    else:
        profile.verify_resume(payload)
        assert payload['global_step'] == config['source_step']
    assert payload['stage1_checkpoint_sha256'] == profile.recipe.tokenizer.sha256
    assert payload['joint_codebook_sha256'] == profile.codebook_sha256
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(payload['config']['arch']))
    original_names = [name for name, _ in model.named_parameters()]
    assert len(original_names) == 780
    # Head initialization is identical in both arms and does not perturb source RNG.
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(config['head_seed'])
        heads = BeamConsistencyHeads(hidden_dim=1536, latent_dim=256,
            projection_dim=config['projection_dim'], shallow_layer=config['shallow_layer'],
            deep_layer=config['deep_layer'])
    model.add_module('beam_consistency', heads)
    result = model.load_state_dict(payload['state_dict'], strict=False, assign=True)
    assert not result.unexpected_keys
    expected_missing = set() if resume else {'beam_consistency.'+name for name in heads.state_dict()}
    assert set(result.missing_keys) == expected_missing
    model.to(device)
    parameters = list(model.parameters())
    primary = [parameter for name, parameter in model.named_parameters() if name in set(original_names)]
    added = list(heads.parameters())
    assert len(primary) == 780 and len(added) == 8
    saved_optimizer = payload['optimizer'] if resume else extend_adam_payload(payload['optimizer'], len(added))
    optimizer = torch.optim.AdamW(parameters, lr=payload['optimizer']['param_groups'][0]['lr'],
        fused=True, foreach=False)
    optimizer.load_state_dict(saved_optimizer)
    configure_optimizer(optimizer)
    source_config = OmegaConf.create(payload['config'])
    scheduler = create_scheduler(optimizer, source_config.optimizer.warmup,
        config['steps_per_epoch'], 100)
    scheduler.load_state_dict(payload['scheduler'])
    current_step = int(payload['global_step'])
    assert scheduler.after_scheduler.last_epoch == current_step
    assert scheduler.after_scheduler.T_max == 62600
    assert optimizer.param_groups[0]['lr'] == scheduler.after_scheduler._last_lr[0]
    assert {int(optimizer.state[p]['step']) for p in primary} == {current_step}
    if resume and arm == 'consistency':
        assert {int(optimizer.state[p]['step']) for p in added} == {current_step-config['source_step']}
    else:
        assert not any(p in optimizer.state for p in added)
    accumulation = math.ceil(config['global_batch']/config['microbatch'])
    saved_accumulation = int(payload['runtime']['accumulation_steps'])
    epoch, cursor = remap_data_cursor(int(payload['next_epoch']), int(payload['next_microbatch']),
        current_step, saved_accumulation, accumulation, config['steps_per_epoch'])
    aux, _ = profile.tokenizer(device)
    assert not any(p.requires_grad for p in aux.parameters()) and aux.reconstruction_beam_width == 16
    noise = torch.Generator(device=device).manual_seed(config['noise_seed'])
    if resume:
        noise.set_state(payload['noise_rng'])
    restore_rng(payload['rng_state_by_rank'][0], device)
    runtime = dict(payload['runtime'])
    runtime.update(world_size=1, microbatch=config['microbatch'], accumulation_steps=accumulation,
        global_batch=config['global_batch'], auxiliary_loss_terms=2 if arm=='consistency' else 0,
        training_objective='hard_beam16_CE_plus_spatial_consistency' if arm=='consistency' else 'hard_beam16_CE',
        source_step=config['source_step'], source_epoch=config['source_step']/626)
    initial = dict(source_step=config['source_step'], restored_step=current_step,
        source_checkpoint_sha256=plan['source_checkpoint_sha256'], primary_optimizer_states=780,
        restored_primary_adam_age=current_step, head_optimizer_age=current_step-config['source_step'] if resume and arm=='consistency' else 0,
        initial_lr=optimizer.param_groups[0]['lr'], scheduler_step=scheduler.after_scheduler.last_epoch,
        scheduler_horizon=scheduler.after_scheduler.T_max, epoch=epoch, next_microbatch=cursor,
        accumulation_steps=accumulation, global_batch=config['global_batch'], microbatch=config['microbatch'],
        shared_parameters_restored=True, new_head_parameter_tensors=8,
        new_head_parameter_count=sum(p.numel() for p in added),
        frozen_tokenizer_digest=profile.digest(aux), targets='exact unchanged hard beam16 IDs',
        arm=arm, world_size=1)
    del payload, saved_optimizer
    gc.collect()
    return dict(local=local, frozen=frozen, model=model, heads=heads, aux=aux, profile=profile,
        optimizer=optimizer, scheduler=scheduler, config=source_config, device=device,
        primary=primary, added=added, parameters=parameters, noise=noise, epoch=epoch,
        cursor=cursor, step=current_step, accumulation=accumulation, runtime=runtime, initial=initial)


def loader_for(state, config):
    import torch
    from torch.utils.data import DataLoader
    from rqvae.img_datasets.transforms import create_transforms
    from src.training.rq_imagenet_runtime import TarImageNet
    from src.training.exact_global_batch import ExactGlobalBatchSampler
    dataset = TarImageNet(state['frozen']/'data', 'train',
        create_transforms(state['config'].dataset, split='train'), seed=261001)
    assert len(dataset) == 1281167
    sampler = ExactGlobalBatchSampler(dataset, config['global_batch'], 1, 0,
        state['accumulation'], 261001)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=config['loader_workers'],
        pin_memory=True, persistent_workers=True, prefetch_factor=1, multiprocessing_context='spawn')
    return loader, sampler


def forward_loss(state, config, images, labels, *, noisy_cap, consistency):
    import torch
    from beam_consistency import corrupt_code_context
    profile, model, aux, heads = [state[k] for k in ['profile','model','aux','heads']]
    targets, codes = profile.targets(aux, images)
    assert torch.equal(targets, codes) and targets.dtype == torch.long
    if consistency:
        with torch.no_grad():
            embeddings, _ = aux.get_code_emb_with_depth(codes)
            clean = embeddings.sum(-2).reshape(len(images), 64, 256)
        context, fraction = corrupt_code_context(codes, 16384, noisy_cap, state['noise'])
        with heads.capture(model, clean):
            logits = model(context, model_aux=aux, cond=labels, amp=True)
        current, following = heads.losses['current'], heads.losses['next']
    else:
        logits = model(codes, model_aux=aux, cond=labels, amp=True)
        current = following = logits.new_zeros((), dtype=torch.float32)
        fraction = logits.new_zeros((), dtype=torch.float32)
    assert logits.dtype == torch.bfloat16
    ce = profile.loss(model, logits, targets)
    assert bool(torch.isfinite(ce)) and float(ce.detach()) < 20
    assert bool(torch.isfinite(current)) and bool(torch.isfinite(following))
    return ce, current, following, fraction, targets


def calibrate(config):
    state = build(config, 'consistency')
    local = state['local']
    import torch
    loader, sampler = loader_for(state, config)
    sampler.set_epoch(state['epoch'])
    sampler.set_start_batch(state['cursor'])
    images, labels = next(iter(loader))
    images = images[:config['calibration_images']].to(state['device'])
    labels = labels[:config['calibration_images']].to(state['device'])
    state['model'].train()
    ce, current, following, _, targets = forward_loss(state, config, images, labels,
        noisy_cap=0, consistency=True)
    ce_value, aux_value = float(ce.detach()), float((current+following).detach())
    def norm(loss, retain):
        grads = torch.autograd.grad(loss, state['parameters'], retain_graph=retain, allow_unused=True)
        value = float(torch.stack([g.detach().double().square().sum() for g in grads if g is not None]).sum().sqrt())
        del grads
        return value
    ce_norm = norm(ce, True)
    aux_norm = norm(current+following, False)
    assert math.isfinite(ce_norm) and math.isfinite(aux_norm) and min(ce_norm,aux_norm)>0
    weight = min(config['maximum_auxiliary_weight'],
        config['maximum_initial_auxiliary_gradient_ratio']*ce_norm/aux_norm)
    assert 0 < weight <= config['maximum_auxiliary_weight']
    # Frozen targets and native decoding must still agree exactly with the measured beam encoder.
    proof = state['profile'].verify_targets(state['aux'], targets, targets)
    result = dict(passed=True, auxiliary_weight=weight, ce=ce_value, raw_auxiliary=aux_value,
        ce_gradient_norm=ce_norm, auxiliary_gradient_norm=aux_norm,
        weighted_gradient_ratio=weight*aux_norm/ce_norm,
        calibration_images=len(images), gradient_ratio_is_initial_batch_only=True,
        target_verification=proof, initial=state['initial'],
        tokenizer_unchanged=state['profile'].digest(state['aux'])==state['initial']['frozen_tokenizer_digest'])
    assert result['tokenizer_unchanged']
    write(local/'calibration.json', result)
    print(json.dumps(result), flush=True)


def evaluate(state, config, folder, run):
    import torch
    from src import rqvae_metrics as metrics
    model, aux, profile, device = [state[k] for k in ['model','aux','profile','device']]
    settings = config['evaluation']
    assert settings['generated_images']==50000 and settings['real_images']==1281167
    assert settings['reference_split']=='train' and settings['inception_splits']==10
    assert dict(profile.recipe.sampling)=={k:settings[k] for k in ['temperature','top_k','top_p']}
    model.eval()
    started = time.monotonic()
    with torch.random.fork_rng(devices=[device.index]), torch.inference_mode():
        torch.manual_seed(settings['seed'])
        metric = metrics.DistributedOriginalRQVAEMetrics(device, compute_inception_score=True,
            reference_stats_path=state['frozen']/'inputs/imagenet_256_train.npz')
        for first in range(0,50000,settings['batch']):
            count = min(settings['batch'],50000-first)
            labels = torch.arange(first,first+count,device=device)%1000
            codes = profile.sample(model,aux,count,labels)
            for part in codes.split(settings['decode_batch']):
                pixels = profile.decode(aux,part).float().mul(.5).add(.5).clamp(0,1)
                metric.update(pixels,real=False)
            write(folder/'generation-progress.json',dict(generated_images=first+count,target=50000,time=time.time()))
        assert int(metric.fake_count)==50000
        torch.manual_seed(settings['seed']+1)
        fid, inception, std = metric.compute(shuffle_for_inception=True)
    assert state['profile'].digest(state['aux'])==state['initial']['frozen_tokenizer_digest']
    row = dict(fid=fid,inception_score=inception,inception_score_std=std,
        generated_images=50000,real_images=1281167,reference_split='train',
        seed=settings['seed'],inception_splits=10,rejection_sampling=False,
        sampling={k:settings[k] for k in ['temperature','top_k','top_p']},
        source_step=config['source_step'],global_step=state['step'],
        trial_updates=state['step']-config['source_step'],elapsed_seconds=time.monotonic()-started,
        reference_sha256=sha(state['frozen']/'inputs/imagenet_256_train.npz'))
    assert math.isfinite(fid) and math.isfinite(inception)
    write(folder/'evaluation.json',row)
    if run:
        run.log({'eval/fid':fid,'eval/inception_score':inception,'eval/inception_score_std':std,
            'train/global_step':state['step'],'trial/update':row['trial_updates']})
        run.summary.update({'evaluation/fid50k':fid,'evaluation/inception_score':inception,
            'evaluation/inception_score_std':std})
    print(json.dumps(dict(event='evaluation_complete',**row)),flush=True)
    model.train()
    return row


def train(config, arm, smoke_updates=0, resume=None, tag=None):
    import concurrent.futures
    import torch
    from beam_consistency import trial_schedule
    from runtime.amp_replay import ReplayWindow, recoverable_amp_step
    from src.training.rq_imagenet_runtime import cpu_snapshot
    from omegaconf import OmegaConf
    folder = Path(config['local'])/(tag or arm)
    folder.mkdir(parents=True,exist_ok=True)
    arm_lock = (folder/'training.lock').open('a')
    fcntl.flock(arm_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = build(config, arm, resume)
    local = state['local']
    checkpoints = folder/'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    os.environ['LASER_STAGE2_OUTPUT'] = str(folder)
    write(folder/'initialization.json',state['initial'])
    run = None
    run_id = f'imnet-beam16-{arm}-s{config["source_step"]}-2ep-1b200-20261105'
    if not smoke_updates:
        import wandb
        (folder/'wandb').mkdir(exist_ok=True)
        os.environ['WANDB_DIR'] = str(folder/'wandb')
        run = wandb.init(entity='helloimlixin-rutgers',project='laser',id=run_id,
            name=f'ImageNet beam16 | {arm} | matched +2ep | frozen tokenizer',
            job_type='beam-consistency-trial',resume='allow',dir=str(folder/'wandb'),
            config=dict(experiment=config,initialization=state['initial'],arm=arm,
                auxiliary_weight=read(local/'calibration.json')['auxiliary_weight'] if arm=='consistency' else 0,
                source_artifact=read(local/'source-checkpoint-receipt.json')['artifact']))
        run.use_artifact(read(local/'source-checkpoint-receipt.json')['artifact'])
        run.use_artifact(str(state['profile'].recipe.tokenizer.paired_tokenizer_artifact))
        run.define_metric('trial/update')
        run.define_metric('train/*',step_metric='trial/update')
        run.define_metric('eval/*',step_metric='trial/update')
        write(folder/'wandb.json',dict(url=run.url,id=run_id))
    calibration = read(local/'calibration.json')
    base_weight = calibration['auxiliary_weight'] if arm=='consistency' else 0
    loader, sampler = loader_for(state,config)
    optimizer, scheduler = state['optimizer'],state['scheduler']
    scaler = torch.amp.GradScaler('cuda',enabled=False)
    replay = ReplayWindow(state['device'],generators=[state['noise']])
    optimizer.zero_grad(set_to_none=True)
    model = state['model']
    model.train()
    source_epoch, source_cursor = state['epoch'],state['cursor']
    stop_step = config['source_step'] + (smoke_updates or config['updates'])
    assert state['step'] < stop_step
    writer = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    save_future = None
    next_epoch, next_cursor = source_epoch, source_cursor
    latest_metrics = {}
    first_targets = False

    def save_checkpoint(final_evaluation=None):
        nonlocal save_future
        if save_future is not None:
            save_future.result()
        ages = {int(optimizer.state[p]['step']) for p in state['primary']}
        assert ages=={state['step']}
        updates = state['step']-config['source_step']
        if arm=='consistency':
            assert {int(optimizer.state[p]['step']) for p in state['added']}=={updates}
        else:
            assert not any(p in optimizer.state for p in state['added'])
        payload = cpu_snapshot(dict(run_id=run_id,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
            global_step=state['step'],epoch=next_epoch,next_epoch=next_epoch,next_microbatch=next_cursor,
            rng_state_by_rank=[rng_state(state['device'])],noise_rng=state['noise'].get_state(),
            config=OmegaConf.to_container(state['config'],resolve=True),runtime=state['runtime'],
            native_sampling=state['profile'].sampling_settings(),
            stage1_checkpoint_sha256=str(state['profile'].recipe.tokenizer.sha256),
            **state['profile'].checkpoint_extra(),evaluation=final_evaluation,
            consistency_trial=dict(arm=arm,configuration=config,source_checkpoint_sha256=read(local/'plan.json')['source_checkpoint_sha256'],
                auxiliary_weight=base_weight,primary_adam_age=state['step'],head_adam_age=updates if arm=='consistency' else None)))
        path = checkpoints/f'step-{state["step"]:07d}.pt'
        step = state['step']
        def write_checkpoint():
            temporary=path.with_suffix('.partial')
            try:
                torch.save(payload,temporary)
                temporary.replace(path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            receipt=dict(path=str(path),global_step=step,bytes=path.stat().st_size,
                trial_updates=step-config['source_step'],arm=arm)
            if run is not None:
                import wandb
                artifact=wandb.Artifact(run_id+'-checkpoint',type='model',metadata=dict(receipt,evaluation=final_evaluation))
                artifact.add_file(str(path),name='last.pt',policy='immutable',skip_cache=True)
                artifact=run.log_artifact(artifact,aliases=['latest',f'step-{step}']+(['final'] if final_evaluation else []))
                artifact.wait()
                assert str(artifact.state)=='COMMITTED'
                receipt.update(artifact=artifact.qualified_name,state='COMMITTED')
            write(folder/'checkpoint.json',receipt)
            write(checkpoints/'latest.json',receipt)
            if run is not None:
                for previous in checkpoints.glob('step-*.pt'):
                    if previous != path:previous.unlink()
            print(json.dumps(dict(event='checkpoint_saved',**receipt)),flush=True)
        save_future=writer.submit(write_checkpoint)

    accumulator = {}
    started = time.monotonic()
    last_update = state['step']-config['source_step']

    def backward(images,labels,absolute):
        nonlocal first_targets
        images=images.to(state['device'],non_blocking=True)
        labels=labels.to(state['device'],non_blocking=True)
        schedule=trial_schedule(state['step']-config['source_step'],config['updates'],
            config['ramp_updates'],config['maximum_noise'])
        ce,current,following,fraction,targets=forward_loss(state,config,images,labels,
            noisy_cap=schedule['noise_cap'] if arm=='consistency' else 0,
            consistency=arm=='consistency')
        if not first_targets:
            proof=state['profile'].verify_targets(state['aux'],targets,targets)
            assert proof['training_targets_equal_reconstruction_beam_codes']
            write(folder/'target-verification.json',proof)
            first_targets=True
        weight=base_weight*schedule['auxiliary_ramp']
        loss=ce+weight*(current+following)
        assert bool(torch.isfinite(loss))
        (loss*len(images)/sampler.global_count(absolute)).backward()
        accumulator[absolute]=dict(images=len(images),ce=float(ce.detach()),
            current=float(current.detach()),next=float(following.detach()),
            loss=float(loss.detach()),auxiliary_weight=weight,noise_cap=schedule['noise_cap'] if arm=='consistency' else 0,
            corrupted_fraction=float(fraction))
        state['heads'].losses={}

    try:
        epoch=source_epoch
        while state['step']<stop_step:
            sampler.set_epoch(epoch)
            sampler.set_start_batch(source_cursor)
            for relative,(images,labels) in enumerate(loader):
                if save_future is not None and save_future.done():save_future.result()
                absolute=source_cursor+relative
                if absolute%state['accumulation']==0:
                    replay.begin();accumulator.clear();started=time.monotonic()
                replay.append(images,labels,absolute)
                backward(images,labels,absolute)
                if (absolute+1)%state['accumulation']:
                    continue
                norm,retries=recoverable_amp_step(state['parameters'],optimizer,scaler,
                    max_norm=.1,replay_backward=lambda:replay.replay(backward))
                scheduler.step();optimizer.zero_grad(set_to_none=True);state['step']+=1
                assert int(optimizer.state[state['primary'][0]]['step'])==state['step']
                count=sum(row['images'] for row in accumulator.values())
                means={key:sum(row['images']*row[key] for row in accumulator.values())/count
                    for key in ['ce','current','next','loss','corrupted_fraction','auxiliary_weight','noise_cap']}
                next_epoch,next_cursor=epoch,absolute+1
                if next_cursor==sampler.steps_per_epoch*state['accumulation']:
                    next_epoch,next_cursor=epoch+1,0
                updates=state['step']-config['source_step']
                latest_metrics=dict(arm=arm,global_step=state['step'],trial_update=updates,
                    epoch=state['step']/626,learning_rate=optimizer.param_groups[0]['lr'],
                    gradient_norm=float(norm),amp_overflow_retries=retries,
                    images_per_second=count/(time.monotonic()-started),
                    peak_allocated_gib=torch.cuda.max_memory_allocated(state['device'])/2**30,
                    adam_age=state['step'],time=time.time(),**means)
                write(folder/'progress.json',latest_metrics)
                if updates<=3 or updates%10==0:
                    print(json.dumps(latest_metrics),flush=True)
                    if run:
                        run.log({'trial/update':updates,**{'train/'+k:v for k,v in latest_metrics.items() if isinstance(v,(float,int))}})
                        run.summary['training/current_step']=state['step']
                        run.summary['training/completed_updates']=updates
                    with (folder/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(latest_metrics)+'\n')
                if updates%config['save_every_updates']==0 and not smoke_updates:
                    save_checkpoint()
                if state['step']>=stop_step:break
            epoch+=1;source_cursor=0
        assert state['profile'].digest(state['aux'])==state['initial']['frozen_tokenizer_digest']
        if save_future is not None:save_future.result()
        if smoke_updates:
            save_checkpoint();save_future.result()
            write(folder/'verification.json',dict(passed=True,arm=arm,initialization=state['initial'],
                final_global_step=state['step'],trial_updates=state['step']-config['source_step'],
                primary_adam_ages=sorted({int(optimizer.state[p]['step']) for p in state['primary']}),
                head_adam_ages=sorted({int(optimizer.state[p]['step']) for p in state['added'] if p in optimizer.state}),
                tokenizer_unchanged=True,latest_metrics=latest_metrics))
        else:
            evaluation=evaluate(state,config,folder,run)
            save_checkpoint(evaluation);save_future.result()
            completed=dict(arm=arm,global_step=state['step'],updates=config['updates'],
                evaluation=evaluation,tokenizer_unchanged=True,initialization=state['initial'],
                checkpoint=read(folder/'checkpoint.json'))
            write(folder/'completed.json',completed)
            run.summary['training/status']='completed'
            run.save(str(folder/'completed.json'),base_path=str(folder),policy='now')
            run.save(str(folder/'evaluation.json'),base_path=str(folder),policy='now')
            run.finish();run=None
    finally:
        writer.shutdown(wait=True)
        if run is not None:run.finish(exit_code=1)


def controller(config, config_path):
    local=Path(config['local']);output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    lock=(local/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    child=None
    parent=None
    def status(phase,**values):
        value=dict(phase=phase,pid=os.getpid(),time=time.time(),**values)
        write(local/'status.json',value);write(output/'status.json',value)
        for arm in config['arms']:
            for name in ['progress.json','evaluation.json','completed.json','wandb.json','checkpoint.json']:
                p=local/arm/name
                if p.exists():write(output/arm/name,read(p))
    def interrupt(sig,_):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGINT)
        status('interrupted',signal=sig)
        raise SystemExit(128+sig)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,interrupt)
    command_base=[sys.executable,'-u',str(local/'runner.py'),'--config',str(config_path)]
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUNBUFFERED='1',PYTORCH_ALLOC_CONF='expandable_segments:True')
    for key in ['WANDB_SERVICE','_WANDB_SERVICE']:environment.pop(key,None)
    def launch(arguments,phase):
        nonlocal child
        with (local/(phase+'.log')).open('ab',buffering=0) as log:
            child=subprocess.Popen(command_base+arguments,cwd=local,env=environment,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                status(phase,child_pid=child.pid,log=str(local/(phase+'.log')))
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:pass
        if child.returncode:raise RuntimeError(f'{phase} failed with {child.returncode}; inspect {local/(phase+".log")}')
    try:
        status('verifying_source')
        configure(config)
        import torch
        from beam_consistency import remap_data_cursor
        source=local/'source-checkpoint.pt';receipt=read(local/'source-checkpoint-receipt.json')
        payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
        assert payload['global_step']==config['source_step']==receipt['global_step']
        assert len(payload['optimizer']['state'])==780
        assert {int(row['step']) for row in payload['optimizer']['state'].values()}=={config['source_step']}
        cursor=remap_data_cursor(payload['next_epoch'],payload['next_microbatch'],payload['global_step'],
            payload['runtime']['accumulation_steps'],math.ceil(config['global_batch']/config['microbatch']))
        plan=dict(configuration=config,source_checkpoint_sha256=sha(source),source_artifact=receipt['artifact'],
            source_step=payload['global_step'],target_step=config['source_step']+config['updates'],
            source_epoch=config['source_step']/626,target_epoch=(config['source_step']+config['updates'])/626,
            original_learning_rate=payload['optimizer']['param_groups'][0]['lr'],
            original_scheduler=payload['scheduler'],mapped_cursor=cursor,source_optimizer_states=780,
            method='reAR-inspired spatial RQ-stack adaptation, not an exact paper reproduction',
            target='sum of four unchanged beam16 physical code vectors at each spatial site',
            training_recipe_unchanged_sha256=sha(Path(config['source_base'])/'inputs/recipe.yaml'),
            primary_production_run_untouched=True,evaluation_generated_images=50000,
            evaluation_reference_images=1281167,evaluation_reference_split='train')
        del payload;gc.collect()
        if (local/'plan.json').exists():assert read(local/'plan.json')==plan
        write(local/'plan.json',plan);write(output/'plan.json',plan)
        if not (local/'calibration.json').exists():launch(['--mode','calibrate'],'calibration')
        write(output/'calibration.json',read(local/'calibration.json'))
        for arm in ['control','consistency']:
            tag='preflight-'+arm
            if not (local/tag/'verification.json').exists():
                launch(['--mode','train','--arm',arm,'--smoke-updates','3','--tag',tag],tag)
            proof=read(local/tag/'verification.json')
            assert proof['passed'] and proof['primary_adam_ages']==[config['source_step']+3]
            assert proof['head_adam_ages']==([3] if arm=='consistency' else [])
            assert proof['initialization']['restored_primary_adam_age']==config['source_step']
            assert proof['tokenizer_unchanged']
            write(output/(tag+'.json'),proof)
        resume=read(local/'preflight-consistency/checkpoint.json')['path']
        if not (local/'preflight-reload/verification.json').exists():
            launch(['--mode','train','--arm','consistency','--smoke-updates','4',
                '--resume',resume,'--tag','preflight-reload'],'preflight-reload')
        proof=read(local/'preflight-reload/verification.json')
        assert proof['passed'] and proof['initialization']['head_optimizer_age']==3
        assert proof['primary_adam_ages']==[config['source_step']+4] and proof['head_adam_ages']==[4]
        write(output/'preflight-reload.json',proof)
        # Smoke updates never enter production. Both arms load the original pinned source.
        for tag in ['preflight-control','preflight-consistency','preflight-reload']:
            for p in (local/tag/'checkpoints').glob('*.pt'):p.unlink()
        configure(config)
        import wandb
        (local/'wandb').mkdir(exist_ok=True)
        parent=wandb.init(entity='helloimlixin-rutgers',project='laser',
            id=f'imnet-beam16-consistency-compare-s{config["source_step"]}-20261105',
            name='ImageNet beam16 | consistency vs CE | matched 2ep comparison',
            job_type='matched-consistency-comparison',resume='allow',dir=str(local/'wandb'),config=plan)
        parent.use_artifact(receipt['artifact'])
        parent.use_artifact('helloimlixin-rutgers/laser/'+config['source_run']+'-source:v1')
        parent.use_artifact('helloimlixin-rutgers/laser/imagenet-atoms4096-joint16384-native-teacher-beam-reconstruction-tokenizer:v0')
        write(output/'wandb.json',dict(url=parent.url,id=parent.id))
        archive=local/'experiment-source.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for folder in ['support','frozen/source','frozen/upstream','frozen/runtime','frozen/inputs']:
                for p in (local/folder).rglob('*'):
                    if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.stat().st_size<50*2**20:
                        z.write(p,p.relative_to(local))
            for name in ['runner.py','config.yaml','plan.json','calibration.json']:z.write(local/name,name)
        artifact=wandb.Artifact(parent.id+'-source',type='code',metadata=dict(source_step=config['source_step'],
            frozen_beam_tokenizer=True,training_recipe_unchanged=True))
        artifact.add_file(str(archive),name='source.zip',policy='immutable',skip_cache=True)
        artifact=parent.log_artifact(artifact);artifact.wait();assert str(artifact.state)=='COMMITTED'
        write(output/'source-artifact.json',dict(artifact=artifact.qualified_name,state='COMMITTED'))
        # Serialized arms fit alongside the original training run under one memory cap.
        for arm in config['arms']:
            if not (local/arm/'completed.json').exists():
                arguments=['--mode','train','--arm',arm]
                checkpoint=local/arm/'checkpoint.json'
                if checkpoint.exists():arguments+=['--resume',read(checkpoint)['path']]
                launch(arguments,'training-'+arm)
            completed=read(local/arm/'completed.json')
            assert completed['updates']==config['updates'] and completed['tokenizer_unchanged']
            assert completed['evaluation']['generated_images']==50000
            assert completed['evaluation']['real_images']==1281167
        consistency=read(local/'consistency/completed.json')['evaluation']
        control=read(local/'control/completed.json')['evaluation']
        report=dict(plan=plan,consistency=consistency,control=control,
            fid_improvement=control['fid']-consistency['fid'],
            inception_score_change=consistency['inception_score']-control['inception_score'],
            beam_tokenizer_unchanged=True,primary_production_recipe_unchanged=
                sha(Path(config['source_base'])/'inputs/recipe.yaml')==plan['training_recipe_unchanged_sha256'],
            limitations='One checkpoint, one two-epoch pilot, one matched evaluation seed; not proof of long-run convergence.')
        assert report['primary_production_recipe_unchanged']
        write(local/'report.json',report);write(output/'report.json',report)
        parent.log({'comparison/results':wandb.Table(columns=['arm','fid50k','inception_score','inception_std'],
            data=[[arm,r['fid'],r['inception_score'],r['inception_score_std']] for arm,r in [('control',control),('consistency',consistency)]])})
        parent.summary.update(dict(fid_improvement=report['fid_improvement'],
            inception_score_change=report['inception_score_change'],
            consistency_fid=consistency['fid'],control_fid=control['fid']))
        parent.save(str(output/'report.json'),base_path=str(output),policy='now')
        parent.finish()
        parent=None
        status('completed',fid_improvement=report['fid_improvement'],inception_score_change=report['inception_score_change'])
    except BaseException as error:
        status('failed',error=str(error),error_type=type(error).__name__)
        if parent is not None:parent.finish(exit_code=1)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=['controller','calibrate','train'],default='controller')
    parser.add_argument('--arm',choices=['consistency','control'])
    parser.add_argument('--smoke-updates',type=int,default=0)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--tag')
    args=parser.parse_args()
    import yaml
    config=yaml.safe_load(args.config.read_text())
    assert config['global_batch']==2048 and config['updates']==1252
    configure(config)
    if args.mode=='controller':controller(config,args.config.resolve())
    elif args.mode=='calibrate':calibrate(config)
    else:
        assert args.arm
        train(config,args.arm,args.smoke_updates,args.resume,args.tag)


if __name__=='__main__':main()
