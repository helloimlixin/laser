#!/usr/bin/env python3
"""Validate the exact tokenizer/cache and run a fresh ImageNet compound prior."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'cache-check', 'smoke', 'train'])
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    sys.path[:0] = [str(base / 'source'), str(base / 'source/scripts/tools')]
    import torch
    from src.training import rqtransformer as recipe
    from imagenet_compound_launch_support import atomic_json
    config = json.loads((base / 'recipe.json').read_text())
    assert config['resume'] is False and config['resume_checkpoint'] is None
    assert config['init_stage2_checkpoint'] is None
    expected_hash = 'dd28db9306d526bdc9fbf8016403e106c4f317fd8f5c04792f0953e53310fdab'
    if args.action == 'preflight':
        assert sha(config['checkpoint']) == expected_hash
        with torch.device('meta'):
            model = recipe.build_model(18432, 16384, compound=True,
                coeff_vocab_size=2048, compound_micro_transformer_layers=2,
                compound_depth_specific_coeff_heads=True,
                compound_pair_autoregressive=True, sparsity_level=4,
                model_preset='imagenet-1400m')
        parameters = sum(p.numel() for p in model.parameters())
        report = dict(passed=True, stage2_from_scratch=True, stage1_frozen=True,
            tokenizer_sha256=expected_hash, source_rfid=4.210914134979248,
            source_artifact='helloimlixin-rutgers/laser/imga16384k4altbn64-b128-b300-20260830000755-stage1-checkpoints:v9',
            source_artifact_digest='c1d2d042e6876471ff28556cf1221932',
            source_file='best_rfid_slot1_model.pt', source_epoch=10,
            compound_reference='helloimlixin-rutgers/laser/church-laser-rfid421-ft3-compound-scratch90-20260918',
            original_recipe='https://github.com/kakaobrain/rq-vae-transformer/blob/main/configs/imagenet256/stage2/in256-rqtransformer-8x8x4-1400M.yaml',
            parameters=parameters, latent_shape=[8, 8, 4], embed_dim=1536,
            body_layers=42, depth_layers=6, attention_heads=24,
            compound_geometry_version='atom_conditional_v2',
            total_batch_size=2048, local_batch_size=8, world_size=config['world_size'],
            accumulation_steps=2048 // (8 * config['world_size']),
            epochs=100, reference_sha256=sha(config['fid_reference_stats']),
            adaptations=['LASER compound atom/coefficient events with two micro-transformer layers',
                'Church soft physical coefficient targets at temperature 0.125 and geometry objective',
                'Validated deterministic center-crop sparse cache; stochastic coefficient targets redrawn each visit',
                'ImageNet-specific maximum coefficient scales calibrated without clipping',
                f"{config['world_size']} GPUs with gradient accumulation; BF16 autocast",
                'FID50k and Inception Score against published ImageNet training statistics every two epochs'])
        atomic_json(base / 'preflight.json', report)
        print(json.dumps(report, indent=2), flush=True)
        return
    cache = Path(config['token_cache'])
    report = json.loads(cache.with_suffix('.validation.json').read_text())
    assert report['passed'] and report['atom_exact_fraction'] == 1.
    assert report['items'] == 1281167
    payload = torch.load(cache, map_location='cpu', weights_only=True, mmap=True)
    meta = payload['meta']
    assert meta['format'] == 'laser_compound_pairs_v1'
    assert meta['shape'] == [8, 8, 4] and meta['items'] == 1281167
    assert meta['stage1_checkpoint'] == config['checkpoint']
    assert payload['atoms'].shape == (1281167, 8, 8, 4)
    assert meta['auto_coeff_scales_percentile'] == 100.
    if args.action == 'cache-check':
        assert sha(config['checkpoint']) == expected_hash
        atomic_json(base / 'cache/ready.json', dict(passed=True,
            tokenizer_sha256=expected_hash, cache_sha256=sha(cache),
            meta=meta, validation=report))
        print(json.dumps(dict(phase='cache_verified', **report)), flush=True)
        return
    del payload
    ready = json.loads((base / 'cache/ready.json').read_text())
    local = Path(os.environ['LASER_LOCAL_SCRATCH'])
    local_cache = local / 'cache/compound-cache.pt'
    assert sha(local_cache) == ready['cache_sha256']
    assert sha(config['checkpoint']) == expected_hash
    world = int(os.environ['WORLD_SIZE'])
    assert world in (4, 8)
    assert 2048 % (world * config['batch_size']) == 0
    device = int(os.environ['LOCAL_RANK'])
    name = torch.cuda.get_device_name(device)
    assert 'A100' in name or 'L40S' in name, name
    phase = args.action
    os.environ['LASER_LAUNCH_PHASE'] = phase
    config.update(token_cache=str(local_cache), checkpoint_dir=str(local / phase / 'checkpoints'))
    if phase == 'smoke':
        config.update(output=str(local / 'smoke/output'), wandb_mode='disabled',
            upload_checkpoints=False, sample_grid_every=0, save_step_freq=0,
            fid_every=0, max_optimizer_steps=1, smoke_test=True,
            geometry_start_epoch=0., geometry_warmup_epochs=0.)
    assert not (Path(config['checkpoint_dir']) / 'last.pt').exists(), 'Refuse scratch overwrite'
    record = dict(rank=int(os.environ['RANK']), gpu=name, host=os.uname().nodename,
        world_size=world, accumulation_steps=2048 // (world * config['batch_size']),
        stage2_from_scratch=True, resume_checkpoint=None,
        initialization_checkpoint=None, tokenizer_sha256=expected_hash,
        cache_sha256=ready['cache_sha256'], coeff_scales=meta['coeff_scales'])
    atomic_json(base / phase / ('rank-%d.json' % record['rank']), record)
    print(json.dumps(dict(phase=phase, **record)), flush=True)
    argv = []
    for action in recipe.build_parser()._actions:
        if not action.option_strings or action.dest not in config:
            continue
        value = config[action.dest]
        if value is None:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            argv.append(flag if value else '--no-' + flag[2:])
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            argv.extend([flag, *map(str, value)])
        else:
            argv.extend([flag, str(value)])
    recipe.main(argv)


if __name__ == '__main__':
    main()
