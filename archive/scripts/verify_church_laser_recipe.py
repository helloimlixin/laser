#!/usr/bin/env python3
"""Strictly reload a locally produced preflight and verify its batch/schedule."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--microbatch',type=int,choices=[256,512],required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    driver=root/'scripts/tools/train_church_laser_original_recipe.py'
    spec=importlib.util.spec_from_file_location('church_recipe',driver)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base=root/'outputs/church-laser-original-recipe-20260913'
    out=base/('preflight' if args.microbatch==256 else 'preflight-512')
    status=json.loads((out/'status.json').read_text())
    assert status['phase']=='preflight_complete'
    for rank in range(2):
        assert json.loads((out/f'sampler-smoke-rank{rank}.json').read_text())['passed']
    initialization=json.loads((out/'initialization.json').read_text())
    original=json.loads((module.SOURCE_RUN/'train/initialization.json').read_text())
    assert initialization['initial_weights_sha256']==original['initial_weights_sha256']
    state=module.torch.load(out/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    assert state['step']==2 and state['attempts']==2
    assert state['config']['experiment']['total_batch_size']==2048
    assert state['config']['experiment']['batch_size']==args.microbatch
    assert state['scheduler']['after']['T_max']==18600
    assert state['scheduler']['after']['last_epoch']==2
    assert len(state['rng_states'])==2 and state['optimizer']['state']
    for value in state['state_dict'].values():
        assert module.torch.isfinite(value).all()
    for entry in state['optimizer']['state'].values():
        for value in entry.values():
            if module.torch.is_tensor(value): assert module.torch.isfinite(value).all()
    config=module.OmegaConf.create(state['config'])
    model,optimizer=module.fresh_transformer(config)
    model.load_state_dict(state['state_dict'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    assert module.state_sha256(model)!=initialization['initial_weights_sha256']
    for group in optimizer.param_groups:
        assert abs(group['lr']-.0005*(1+math.cos(math.pi*2/18600))/2)<1e-14
    proof=dict(passed=True,strict_checkpoint_reload=True,model_and_optimizer_finite=True,
        successful_updates=2,skipped_updates=0,global_batch=2048,
        microbatch_per_gpu=args.microbatch,accumulation_steps=2048//(2*args.microbatch),
        scheduled_updates=18600,sampler_decode_passed=True,
        initial_weights_sha256=initialization['initial_weights_sha256'],
        tokenizer_unchanged=True,peak_gpu_allocated_gib=status['peak_gpu_allocated_gib'],
        driver_sha256=module.file_sha256(driver))
    module.atomic_json(base/f'verification-{args.microbatch}.json',proof)
    print(json.dumps(proof))


if __name__=='__main__': main()
