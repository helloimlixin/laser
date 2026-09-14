#!/usr/bin/env python3
"""Audit saved ImageNet training state against the published 480M RQ recipe."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM=ROOT/'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0]=[str(ROOT),str(UPSTREAM)]
from omegaconf import OmegaConf
import torch
from src.imagenet_scaled_stage2 import load_imagenet_config
from src.original_rq_training import atomic_json,file_sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    config=OmegaConf.load(args.run_dir/'config.yaml')
    original=load_imagenet_config(UPSTREAM,vocab_size=16384)
    native=OmegaConf.to_container(original,resolve=True)
    actual=OmegaConf.to_container(config,resolve=True)
    architecture=dict(actual['arch'])
    architecture['vocab_size']=16384
    assert architecture==native['arch'],'Transformer body/head differs from the original architecture'
    assert actual['optimizer']==native['optimizer'],'Optimizer recipe differs'
    assert config.experiment.total_batch_size==2048 and config.experiment.epochs==100
    assert config.experiment.amp and config.experiment.test_freq==2 and config.experiment.save_ckpt_freq==2
    assert config.loss.type=='soft_target_cross_entropy' and config.loss.stochastic_codes
    assert config.training_data.mode=='online-images' and not config.cache.training_latents_reused
    sampler=OmegaConf.to_container(config.generation_sampler)
    published_path=args.run_dir.parent/'original-published-stage2-config.yaml'
    published=OmegaConf.load(published_path)
    expected_sampler=dict(temp=float(published.sampling.temp),top_k=int(published.sampling.top_k[0]),
        top_p=float(published.sampling.top_p[0]))
    assert sampler==dict(name='published_imagenet_480m',mode='joint',temperature=expected_sampler['temp'],
        top_k=expected_sampler['top_k'],top_p=expected_sampler['top_p'])
    assert OmegaConf.to_container(config.sampling)==expected_sampler
    initialization=json.loads((args.run_dir/'initialization.json').read_text())
    batch=initialization['microbatch_per_gpu']*initialization['world_size']*initialization['gradient_accumulation']
    assert batch==2048==initialization['effective_batch_size']
    saved=torch.load(args.run_dir/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    assert saved['config']==actual,'Checkpoint configuration differs from runtime configuration'
    assert saved['world_size']==4 and len(saved['rng_states'])==4
    assert saved['initial_weights_sha256']==initialization['initial_weights_sha256']
    schedule=saved['scheduler']
    assert schedule['warmup'] is None
    schedule=schedule['after']
    assert schedule['T_max']==100*initialization['steps_per_epoch'] and schedule['eta_min']==0
    assert schedule['last_epoch']==saved['step']
    expected_lr=.0005*(1+math.cos(math.pi*saved['step']/schedule['T_max']))/2
    for group in saved['optimizer']['param_groups']:
        assert tuple(group['betas'])==(.9,.95) and group['weight_decay']==.0001 and group['eps']==1e-8
        assert group['initial_lr']==.0005 and math.isclose(group['lr'],expected_lr,rel_tol=1e-10,abs_tol=1e-12)
    calibration=json.loads((args.run_dir/'temperature-calibration.json').read_text())
    assert config.loss.temp==calibration['selected_temperature']
    compact_sweep=calibration.get('compact_rq_sweep',calibration.get('scaled_rq_sweep',[]))
    selected=next(row for row in compact_sweep if row['temperature']==config.loss.temp)
    assert selected['sampled_to_hard_residual_mse_ratio']<=calibration['allowed_mse_ratio']
    report=dict(published_training_hyperparameters_verified=True,exact_original_tokenizer=False,
        updated_unix=time.time(),checkpoint_step=saved['step'],checkpoint_epoch=saved['epoch'],
        global_batch=batch,microbatch_per_gpu=initialization['microbatch_per_gpu'],
        gradient_accumulation=initialization['gradient_accumulation'],world_size=4,
        published_microbatch=original.experiment.batch_size,
        optimizer=dict(type='AdamW',initial_lr=.0005,betas=[.9,.95],weight_decay=.0001,eps=1e-8,gradient_clip=1.),
        schedule=dict(type='cosine',epochs=100,warmup_epochs=0,min_lr=0.,optimizer_steps=schedule['T_max'],
            current_lr=expected_lr),sampler=sampler,
        augmentation='Released Resize256/RandomCrop256/RandomHorizontalFlip; fresh each image and epoch',
        sparse_adaptations=dict(vocabulary=actual['arch']['vocab_size'],original_vocabulary=16384,
            original_soft_target_temperature=.5,sparse_soft_target_temperature=float(config.loss.temp),
            sampled_hard_mse_ratio=selected['sampled_to_hard_residual_mse_ratio'],
            original_control_mse_ratio=calibration['original_rq_control']['sampled_to_hard_residual_mse_ratio']),
        checkpoint_rng_states=len(saved['rng_states']),resumed_from_step=initialization.get('resumed_from_step'),
        evidence=dict(recipe=str(UPSTREAM/'configs/imagenet256/stage2/in256-rqtransformer-8x8x4-480M.yaml'),
            published_checkpoint_configuration=str(published_path),
            published_checkpoint_configuration_sha256=file_sha256(published_path),
            runtime_config_sha256=file_sha256(args.run_dir/'config.yaml'),
            paper='https://arxiv.org/html/2203.01941v2#A3',
            repository='https://github.com/kakaobrain/rq-vae-transformer'))
    atomic_json(args.output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
