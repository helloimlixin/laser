#!/usr/bin/env python3
"""Train the released Church RQTransformer from scratch after one-epoch RQ-VAE FT.

KakaoBrain did not publish its stage-2 training loop. This driver uses its released
model, initialization helper, optimizer helper, scheduler, soft targets and sampler.
"""
import argparse
from datetime import timedelta
from itertools import islice
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--stage1-directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--reference', type=Path, default=ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
    return p


def main():
    args = parser().parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from omegaconf import OmegaConf
    from rqvae.optimizer import create_scheduler
    from rqvae.metrics.fid import get_inception_model
    from src.original_rq_training import (atomic_json, file_sha256, seed_all, state_sha256,
        load_stage2_config, load_tokenizer, fresh_transformer, build_latent_cache,
        accumulated_update, evaluate_samples, validation_latents, evaluate_validation)

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    assert world == 2
    output = args.output.resolve()
    run = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        # A detached launcher may have already shut down its W&B service.
        # This run must create its own service rather than inherit that socket.
        for name in ('WANDB_SERVICE', '_WANDB_SERVICE'):
            os.environ.pop(name, None)
        import wandb
        run = wandb.init(project='laser', entity='helloimlixin-rutgers', id=args.run_id,
            name=args.run_id, resume='never', dir=str(output), config={
                'pipeline': 'original-rqvae-original-rqtransformer',
                'stage1_source_published_imagenet_rfid': 4.73,
                'stage1_finetune_epochs': 1, 'stage2_from_scratch': True,
                'upstream_commit': '341395e562ac347f5eb62db9f5f08b9f2cc42a60',
                'stage2_trainer': 'new driver; upstream did not publish training loop',
                'configuration_authority': 'released repository YAML; paper differs',
            })
        run.summary['training_status'] = 'waiting_for_one_epoch_tokenizer_finetuning'
        atomic_json(output / 'wandb.json', {'run_id': run.id, 'url': run.url})

    # Wait without initializing CUDA; stage 1 uses both devices until it exits.
    marker = args.stage1_directory / 'complete.json'
    last_stage1_update = None
    deadline = time.monotonic() + 10800
    while not marker.exists():
        failures = list(args.stage1_directory.glob('failure-rank*.json'))
        if failures:
            raise RuntimeError(f'Stage1 failed; stage2 blocked: {failures}')
        if time.monotonic() > deadline:
            raise TimeoutError('One-epoch fine-tuning did not complete within 3 hours')
        if rank == 0:
            status_path = args.stage1_directory / 'status.json'
            if status_path.exists():
                status = json.loads(status_path.read_text())
                atomic_json(output / 'status.json', {'phase': 'waiting_for_stage1', 'stage1': status,
                                                     'updated_unix': time.time()})
                if status.get('updated_unix') != last_stage1_update:
                    values = {f'stage1/{k}': status[k] for k in
                              ('optimizer_step', 'epoch_fraction', 'lr', 'elapsed_seconds') if k in status}
                    if values:
                        run.log(values)
                    last_stage1_update = status.get('updated_unix')
        time.sleep(10)
    completed = json.loads(marker.read_text())
    assert completed['epoch'] == 1 and completed['optimizer_steps'] == 987
    assert completed['smoke_only'] is False
    checkpoint = Path(completed['checkpoint'])
    assert file_sha256(checkpoint) == completed['checkpoint_sha256']
    # The completion marker is written after upstream's final barrier and save.
    # Wait for torchrun to release the stage-1 CUDA contexts too.
    stage1_pid = json.loads((args.stage1_directory / 'launch.json').read_text())['pid']
    while Path(f'/proc/{stage1_pid}/cmdline').exists() and Path(f'/proc/{stage1_pid}/cmdline').read_bytes():
        time.sleep(2)

    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    # Retain FP32 tokenizer distances; no TF32 approximation to quantizer matmuls.
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=3))
    dist.barrier()
    seed_all(0)
    config = load_stage2_config(args.upstream, checkpoint)
    if rank == 0:
        OmegaConf.save(config, output / 'config.yaml')
        atomic_json(output / 'stage1-complete.json', completed)
        run.config.update(OmegaConf.to_container(config, resolve=True))
        run.summary['training_status'] = 'caching_frozen_tokenizer_latents'
        # Publish stage-1 curves retained by the unchanged upstream loop.
        for row in (args.stage1_directory / 'metrics.jsonl').read_text().splitlines():
            record = json.loads(row)
            run.log({f'stage1/{record["mode"]}/{record["tag"]}': record['value'],
                     'stage1/log_index': record['step_or_epoch']})

    tokenizer, _ = load_tokenizer(checkpoint, completed['config'], device)
    seed_all(0)  # Initialization must not depend on tokenizer constructor RNG use.
    model, optimizer = fresh_transformer(config)  # stays on CPU during caching
    initial_hash = state_sha256(model)
    hashes = [None] * world
    dist.all_gather_object(hashes, initial_hash)
    assert len(set(hashes)) == 1
    assert len(optimizer.state) == 0
    initialization = {'stage2_from_scratch': True, 'seed': 0,
        'pretrained_stage2_checkpoint': None, 'initial_optimizer_entries': len(optimizer.state),
        'initial_weights_sha256': initial_hash, 'initial_weights_identical_across_ranks': True,
        'parameters': sum(p.numel() for p in model.parameters()),
        'initialization': 'released Stage2Model._init_weights (normal std 0.02)',
        'optimizer_grouping': 'released create_resnet_optimizer: all parameters, AdamW',
        'tokenizer_checkpoint': str(checkpoint), 'tokenizer_sha256': completed['checkpoint_sha256'],
        'upstream_commit': '341395e562ac347f5eb62db9f5f08b9f2cc42a60'}
    if rank == 0:
        atomic_json(output / 'initialization.json', initialization)
        print(json.dumps({'phase': 'fresh_stage2_initialized', **initialization}), flush=True)

    dataset = build_latent_cache(tokenizer, config, output, device, args.reference, rank, world)
    heldout = validation_latents(tokenizer, config, device, rank, world)
    seed_all(0)  # cache and reconstruction evaluation must not consume training RNG
    model.to(device)
    # Moving a model may replace Parameter objects on future PyTorch versions.
    from rqvae.optimizer.optimizer import create_resnet_optimizer
    optimizer = create_resnet_optimizer(model, config.optimizer)
    assert not optimizer.state
    ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=0)
    loader = DataLoader(dataset, sampler=sampler, batch_size=config.experiment.batch_size,
                        num_workers=8, pin_memory=True, persistent_workers=True, drop_last=False)
    accumulation = config.experiment.total_batch_size // (world * config.experiment.batch_size)
    assert accumulation * world * config.experiment.batch_size == 256
    steps_per_epoch = math.ceil(len(loader) / accumulation)
    assert steps_per_epoch == 494
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, steps_per_epoch,
                                 config.experiment.epochs)
    scaler = torch.amp.GradScaler('cuda')
    seed_all(rank)
    inception = get_inception_model().eval().requires_grad_(False).to(device)
    if rank == 0:
        run.log({f'stage1/{k}': v for k, v in json.loads((output / 'cache.json').read_text()).items()
                 if k.startswith('rfid_')})
        run.summary['training_status'] = 'training_stage2_from_scratch'

    stopping = {'requested': False}
    def on_signal(signum, frame):
        stopping['requested'] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    step, attempts, skipped = 0, 0, 0
    started = time.time()
    last_saved = time.monotonic()

    def save(epoch, batch, name='last.pt'):
        states = [None] * world
        dist.all_gather_object(states, {'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device), 'numpy': __import__('numpy').random.get_state(),
            'python': __import__('random').getstate()})
        if rank == 0:
            dest = output / name
            torch.save({'epoch': epoch, 'batch_in_epoch': batch, 'step': step, 'attempts': attempts,
                        'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                        'rng_states': states, 'tokenizer': completed,
                        'config': OmegaConf.to_container(config, resolve=True)}, dest.with_suffix('.tmp'))
            dest.with_suffix('.tmp').replace(dest)
        dist.barrier()

    for epoch_index in range(config.experiment.epochs):
        sampler.set_epoch(epoch_index)
        model.train()
        iterator = iter(loader)
        consumed = 0
        totals = {'loss_sum': 0., 'images': 0}
        for update_index in range(steps_per_epoch):
            batches = list(islice(iterator, accumulation))
            assert batches
            consumed += len(batches)
            metrics = accumulated_update(ddp, tokenizer, optimizer, scaler, batches,
                                         max_gn=config.optimizer.max_gn)
            attempts += 1
            if metrics['optimizer_updated']:
                scheduler.step()
                step += 1
            else:
                skipped += 1
            totals['loss_sum'] += metrics['loss'] * metrics['global_images']
            totals['images'] += metrics['global_images']
            if not metrics['optimizer_updated'] and skipped > 25:
                save(epoch_index, consumed)
                raise FloatingPointError('More than 25 AMP-overflow skips; saved for inspection')
            epoch_fraction = epoch_index + (update_index + 1) / steps_per_epoch
            if rank == 0 and (attempts <= 10 or attempts % 10 == 0):
                record = {'phase': 'training', 'optimizer_step': step, 'attempted_updates': attempts,
                    'epoch': epoch_fraction, 'lr': optimizer.param_groups[0]['lr'],
                    'skipped_amp_updates': skipped, 'updated_unix': time.time(),
                    'elapsed_seconds': time.time() - started, 'pid': os.getpid(), **metrics}
                atomic_json(output / 'status.json', record)
                with (output / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                run.log({f'stage2/{k}': v for k, v in record.items() if isinstance(v, (int, float))})
                print(json.dumps(record), flush=True)
            signal_tensor = torch.tensor(int(stopping['requested']), device=device)
            dist.all_reduce(signal_tensor, op=dist.ReduceOp.MAX)
            if signal_tensor.item():
                save(epoch_index, consumed)
                if rank == 0:
                    atomic_json(output / 'status.json', {'phase': 'paused', 'epoch': epoch_fraction,
                                'optimizer_step': step, 'updated_unix': time.time()})
                    run.summary['training_status'] = 'paused_with_checkpoint'
                    run.finish()
                dist.destroy_process_group()
                return
            # All ranks make identical checkpoint decisions; rank-0 wall clock is broadcast.
            checkpoint_due = torch.tensor(int(rank == 0 and time.monotonic() - last_saved > 900), device=device)
            dist.broadcast(checkpoint_due, 0)
            if checkpoint_due.item():
                save(epoch_index, consumed)
                last_saved = time.monotonic()
        assert consumed == len(loader)
        epoch = epoch_index + 1
        if rank == 0:
            run.log({'stage2/epoch_complete': epoch, 'stage2/epoch_soft_ce': totals['loss_sum'] / totals['images']})
        save(epoch, 0)
        if epoch % config.experiment.save_ckpt_freq == 0:
            # Model-only archival checkpoints match upstream evaluation's state_dict format.
            if rank == 0:
                torch.save({'epoch': epoch, 'step': step, 'state_dict': model.state_dict(),
                            'tokenizer': completed}, output / f'epoch{epoch}_model.pt')
            dist.barrier()
        if epoch == 1 or epoch % config.experiment.test_freq == 0:
            validation = evaluate_validation(model, tokenizer, heldout, device, rank)
            if rank == 0:
                atomic_json(output / f'validation-epoch{epoch:03d}.json', {'epoch': epoch, **validation})
                run.log({'validation/epoch': epoch, **{f'validation/{k}': v for k, v in validation.items()}})
            n = 50000 if epoch % 50 == 0 else 4096
            score = evaluate_samples(model, tokenizer, inception, n, epoch, output,
                                     args.reference, device, rank, world)
            if rank == 0:
                run.log({f'generation/fid_{n}': score, 'generation/epoch': epoch,
                         'generation/samples': __import__('wandb').Image(str(output / f'samples-epoch{epoch:03d}.png'))})
    if rank == 0:
        atomic_json(output / 'status.json', {'phase': 'complete', 'epochs': 300,
                                            'optimizer_step': step, 'updated_unix': time.time()})
        run.summary['training_status'] = 'complete'
        run.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
