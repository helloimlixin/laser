#!/usr/bin/env python3
"""Check the actual recovered model, optimizer, LR control, tokenizer and data."""
import copy
import json
import math
from pathlib import Path
import sys

import resume_church_laser_ft3ep as recipe
from church_laser_continuation import ContinuationLearningRate
from church_laser_resume_state import validate_resume_payload, restore_training_state
from prepare_church_laser_cache import verify_data
import torch
from omegaconf import OmegaConf
from rqvae.models import create_model
from rqvae.optimizer import create_scheduler
from rqvae.optimizer.optimizer import create_resnet_optimizer


def main():
    base = recipe.BASE
    torch.set_num_threads(4)
    checkpoint = Path('/mnt/scratch/xl598/church-laser-resume/checkpoint/last.pt')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert (payload['epoch'], payload['batch_in_epoch'], payload['step']) == (91, 0, 5642)
    cache = json.loads((base/'recovery/cache/complete.json').read_text())
    cache.update(checkpoint=str(base/'recovery/tokenizer/epoch3-tokenizer.pt'),
                 codebook=str(base/'recovery/tokenizer/compact-codebook.pt'))
    calibration = json.loads((base/'recovery/tokenizer/temperature-calibration.json').read_text())
    assert calibration == payload['temperature_calibration']
    config = recipe.load_stage2_config(recipe.UPSTREAM, cache['checkpoint'])
    config.arch.vocab_size = config.dataset.vocab_size = 32769
    config.loss.temp = calibration['selected_temperature']
    for field in ('arch', 'optimizer', 'loss'):
        assert OmegaConf.to_container(config[field],resolve=True) == payload['config'][field], field
    with torch.device('meta'):
        model, _ = create_model(config.arch, ema=False)
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    assert sum(p.numel() for p in model.parameters()) == 386882561
    optimizer = create_resnet_optimizer(model, config.optimizer)
    scheduler = create_scheduler(optimizer, config.optimizer.warmup, 62, 300)
    scaler = torch.amp.GradScaler('cpu')
    restore_training_state(payload, model, optimizer, scheduler, scaler)
    for parameter, state in optimizer.state.items():
        assert parameter.shape == state['exp_avg'].shape == state['exp_avg_sq'].shape
        assert int(state['step']) == 5642
    assert scaler.get_scale() == 262144
    controller = ContinuationLearningRate(optimizer, scheduler, state=payload['fid_lr_control'])
    original_lr = optimizer.param_groups[0]['lr']
    assert controller.state_dict() == payload['fid_lr_control']
    assert math.isclose(original_lr, .00025*(1+math.cos(math.pi*5642/18600))/2, rel_tol=1e-10)
    optimizer.step()  # All gradients are absent; marks scheduler bookkeeping only.
    controller.step()
    next_lr = optimizer.param_groups[0]['lr']
    assert math.isclose(next_lr, .00025*(1+math.cos(math.pi*5643/18600))/2, rel_tol=1e-10)
    protocol = dict(payload['fid_protocol'], generation_batch_size=100,
                    seed_rule='independent global batches: seed=71000+batch_index')
    baseline = controller.observe(6200, 14., protocol)
    assert baseline['decision'] == 'baseline' and controller.controller.multiplier == .5
    assert controller.observe(6820, 14.2, protocol)['decision'] == 'watch'
    assert controller.observe(7440, 14.1, protocol)['decision'] == 'reduced'
    assert controller.controller.multiplier == .25
    checked = []
    for world in (4,8,16):
        for batch in (16,32):
            accumulation = 2048//(world*batch)
            batches = math.ceil(math.ceil(126227/world)/batch)
            assert math.ceil(batches/accumulation) == 62
            assert validate_resume_payload(payload,world=world,batch_size=batch,cache=cache,
                calibration=calibration,loader_batches=batches,accumulation=accumulation) == (91,0)
            try:
                validate_resume_payload(dict(payload,batch_in_epoch=4),world=world,batch_size=batch,
                    cache=cache,calibration=calibration)
            except ValueError:
                pass
            else:
                raise AssertionError('Mid-epoch batch migration accepted')
            # Exact FID population, independent of world size.
            indices = [i for rank in range(world) for i in range(rank,500,world)]
            assert sorted(indices) == list(range(500))
            checked.append(dict(gpus=world,batch_size=batch,accumulation=accumulation))
    assert recipe.file_sha256(cache['checkpoint']) == cache['checkpoint_sha256']
    assert recipe.file_sha256(cache['codebook']) == cache['codebook_sha256']
    tokenizer = recipe.load_frozen_tokenizer(cache)
    assert recipe.state_sha256(tokenizer) == cache['frozen_state_sha256']
    source = torch.load(cache['checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    assert source['epoch'] == 3
    assert recipe.file_sha256(base/'reference/real-statistics.npz') == payload['fid_protocol']['reference_sha256']
    report = dict(passed=True, checkpoint_epoch=91,checkpoint_step=5642,parameters=386882561,
        optimizer_entries=len(optimizer.state),checkpoint_lr=original_lr,next_lr=next_lr,
        saved_lr_multiplier=.5,scaler_scale=262144,stage1_epochs=3,stage1_metadata={
            k:v for k,v in source.items() if k!='state_dict'},supported_allocations=checked,
        data=verify_data(base),exact_frozen_tokenizer_verified=True,exact_real_fid_reference_verified=True,
        full_model_and_optimizer_strict_restore=True,lr_step_and_plateau_verified=True,
        fid_partition_exact_50000=True,latents_rebuild_required=True,gpu_execution_pending=True)
    recipe.atomic_json(base/'preflight.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
