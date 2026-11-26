#!/usr/bin/env python3
"""CPU preflight against the actual recovered Church model and optimizer state."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    args = p.parse_args()
    base = args.base.resolve()
    sys.path[:0] = [str(base / 'source'), str(base / 'source/upstream')]
    from src.original_rq_training import file_sha256, state_sha256, load_stage2_config
    from rqvae.models import create_model
    from rqvae.utils.config import load_config, augment_arch_defaults
    from rqvae.optimizer import create_scheduler
    from rqvae.optimizer.optimizer import create_resnet_optimizer
    from church_control_resume_support import restore_training_state, validate_control_resume, ranked_candidates
    from prepare_church_control_cache import ChurchImages, verify_keys
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert payload['step'] == payload['attempts'] == 1984
    assert payload['epoch'] == 32 and payload['batch_in_epoch'] == 0
    tokenizer_path = base / 'assets/tokenizer/model.pt'
    tokenizer_config = base / 'assets/tokenizer/config.yaml'
    assert file_sha256(tokenizer_path) == payload['tokenizer']['checkpoint_sha256']
    assert file_sha256(tokenizer_config) == payload['tokenizer']['config_sha256']
    config = load_stage2_config(base / 'source/upstream', tokenizer_path)
    config.experiment.batch_size = 32
    config.experiment.total_batch_size = 2048
    config.experiment.sample.top_k = config.sampling.top_k = 1400
    config.experiment.sample.top_p = config.sampling.top_p = 1.
    # Meta construction checks every tensor shape without allocating a random full-size model.
    with torch.device('meta'):
        model, _ = create_model(config.arch, ema=False)
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    assert sum(p.numel() for p in model.parameters()) == 370087936
    optimizer = create_resnet_optimizer(model, config.optimizer)
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 62, 300)
    scaler = torch.amp.GradScaler('cpu')
    restore_training_state(payload, model, optimizer, scheduler, scaler)
    for parameter, state in optimizer.state.items():
        assert state['exp_avg'].shape == parameter.shape == state['exp_avg_sq'].shape
        assert int(state['step']) == 1984
    assert scaler.get_scale() == 65536
    lr = optimizer.param_groups[0]['lr']
    assert abs(lr - .0005 * (1 + math.cos(math.pi * 1984 / 18600)) / 2) < 1e-15
    optimizer.step()  # No gradients: no weight mutation, advances scheduler usage bookkeeping.
    scheduler.step()
    assert abs(optimizer.param_groups[0]['lr'] - .0005 * (1 + math.cos(math.pi * 1985 / 18600)) / 2) < 1e-15
    checked = []
    for world in (8, 16):
        config.training_world_size = world
        config.experiment.accumulation_steps = 2048 // (world * 32)
        batches = math.ceil(math.ceil(126227 / world) / 32)
        assert math.ceil(batches / config.experiment.accumulation_steps) == 62
        settings = OmegaConf.to_container(config, resolve=True)
        assert validate_control_resume(payload, config=settings, cache=payload['tokenizer'],
            protocol=payload['fid_protocol'], run_id=payload['run_id'], loader_batches=batches) == (32, 0)
        invalid = dict(payload, batch_in_epoch=1)
        try:
            validate_control_resume(invalid, config=settings, cache=payload['tokenizer'],
                protocol=payload['fid_protocol'], run_id=payload['run_id'], loader_batches=batches)
        except AssertionError:
            pass
        else:
            raise AssertionError('Mid-epoch migration was not rejected')
        checked.append(world)
    with torch.device('meta'):
        tokenizer, _ = create_model(augment_arch_defaults(load_config(tokenizer_config).arch), ema=False)
    tokenizer_state = torch.load(tokenizer_path, map_location='cpu', weights_only=False, mmap=True)
    tokenizer.load_state_dict(tokenizer_state['state_dict'], strict=True, assign=True)
    assert state_sha256(tokenizer) == payload['tokenizer']['frozen_state_sha256']
    data = {}
    for split, category in [('train', 'church'), ('val', 'church_val')]:
        dataset = ChurchImages(base / f'assets/data/church_outdoor_{split}_lmdb')
        key_hash = verify_keys(dataset, payload['tokenizer']['data_verification'][category])
        image, _ = dataset[0]
        assert image.shape == (3, 256, 256) and image.isfinite().all()
        data[split] = dict(images=len(dataset), key_order_sha256=key_hash)
    ranked = ranked_candidates(payload['ranked_fid_checkpoints'], fid=10., epoch=40, step=2480)
    assert len(ranked) == 3 and ranked[0]['step'] == 2480
    manifest = json.loads((base / 'runtime-manifest.json').read_text())
    for name, digest in manifest.items():
        assert file_sha256(base / 'source' / name) == digest
    report = dict(passed=True, checkpoint_epoch=32, checkpoint_step=1984, optimizer_entries=len(optimizer.state),
        checkpoint_lr=lr, next_lr=optimizer.param_groups[0]['lr'], scaler_scale=scaler.get_scale(),
        supported_gpu_counts=checked, global_batch=2048, steps_per_epoch=62,
        tokenizer_exact_hash_verified=True, full_model_strict_load_verified=True,
        optimizer_and_scheduler_restore_verified=True, mid_epoch_migration_rejected=True,
        dataset=data, source_files_verified=len(manifest), gpu_execution_pending=True)
    (base / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
