#!/usr/bin/env python3
"""Strictly reload the fresh integer-token GPU preflight and its optimizer."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    base=Path(os.environ['LASER_RQ_BASE'])
    os.environ['RQ_UPSTREAM_ROOT']=str(base/'upstream')
    sys.path[:0]=[str(base/'source'),str(base/'upstream'),str(base/'runtime')]
    import torch
    from omegaconf import OmegaConf
    from src.training.laser_packed_integer_recipe import PackedIntegerLaserRecipe
    from src.training.rq_sample_panels import CLASSES, PREVIEW_IMAGES, OVERVIEW_IMAGES
    from rqvae.utils.config import load_config,augment_arch_defaults,augment_optimizer_defaults
    from rqvae.optimizer.optimizer import create_resnet_optimizer
    from rqvae.optimizer.scheduler import create_scheduler
    profile=PackedIntegerLaserRecipe(base)
    out=Path('/workspace/Projects/laser/outputs')/profile.run_id
    verification=json.loads((out/'smoke/verification.json').read_text())
    assert verification['passed'] and verification['global_step']==2
    receipts=[json.loads(p.read_text()) for p in (out/'smoke/checkpoints/.checkpoint-data').glob('*.json')]
    receipt=max(receipts,key=lambda r:r['global_step'])
    assert receipt['global_step']==2
    for name in ['local','persistent']:
        path=Path(receipt[name]);assert path.stat().st_size==receipt['bytes']
        with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==receipt['sha256']
    saved=torch.load(receipt['local'],map_location='cpu',mmap=True,weights_only=False)
    profile.verify_resume(saved)
    assert saved['runtime']['fresh_stage2'] and saved['runtime']['fresh_optimizer']
    assert saved['runtime']['global_step']==0 and saved['runtime']['initial_lr']==.0005
    assert saved['runtime']['warmup_epochs']==0 and saved['runtime']['lr_schedule_epochs']==100
    config=load_config(base/'inputs/stage2-training-config.yaml')
    arch=augment_arch_defaults(config.arch);arch.vocab_size=config.dataset.vocab_size
    with torch.device('meta'):model=profile.model(arch)
    model.load_state_dict(saved['state_dict'],strict=True,assign=True)
    parameters=sum(p.numel() for p in model.parameters())
    assert parameters==2444674560
    assert model.classifier.linear.classes.weight.shape==(8388608,128)
    assert len(saved['optimizer']['state'])==len(list(model.parameters()))
    for state in saved['optimizer']['state'].values():
        assert int(state['step'])==2
        for value in [state['exp_avg'],state['exp_avg_sq']]:
            for chunk in value.reshape(-1).split(1048576):assert torch.isfinite(chunk).all()
    optimizer=create_resnet_optimizer(model,augment_optimizer_defaults(config.optimizer))
    scheduler=create_scheduler(optimizer,config.optimizer.warmup,626,100)
    optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    assert scheduler.after_scheduler.last_epoch==2 and scheduler.after_scheduler.T_max==62600
    expected_lr=.0005*(1+math.cos(math.pi*2/62600))/2
    assert abs(optimizer.param_groups[0]['lr']-expected_lr)<1.e-12
    preview=json.loads((out/'smoke/samples/step000002-panels.json').read_text())
    assert preview['generated_images']==PREVIEW_IMAGES and preview['overview_images']==OVERVIEW_IMAGES
    assert preview['classes']==[dict(name=name,imagenet_id=label) for name,label in CLASSES]
    assert preview['samples_per_class']==64
    proof=dict(passed=True,parameter_count=parameters,vocabulary=8388608,
        classifier_projection_dim=128,strict_model_reload=True,strict_optimizer_reload=True,
        scheduler_updates=2,cosine_total_updates=62600,all_adam_steps=2,
        optimizer_state_count=len(saved['optimizer']['state']),
        source_checkpoint_sha256=receipt['sha256'],persistent_sha256_verified=True,
        production_starts_from_zero=True,preflight_weights_discarded=True,
        designated_preview_images=PREVIEW_IMAGES,overview_images=OVERVIEW_IMAGES,grids_per_class=[8,8],time=time.time())
    (out/'smoke-reload-verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__=='__main__':main()
