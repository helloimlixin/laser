#!/usr/bin/env python3
"""Matched FID50k for integer-token samplers at one frozen checkpoint."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    base, out = Path(plan['base']), Path(plan['output'])
    source_epoch = int(plan.get('source_epoch', 1))
    source_step = int(plan.get('source_step', 626 * source_epoch))
    run_baseline_fid = plan.get('run_baseline_fid', True)
    baseline_top_k = int(plan.get('baseline_top_k', 512))
    baseline_fid = float(plan['baseline_fid'] if 'baseline_fid' in plan else plan['original_fid512'])
    top_k_values = plan.get('top_k_values', [512, 2048])
    assert top_k_values[0] == baseline_top_k
    os.environ['RQ_UPSTREAM_ROOT'] = str(base / 'upstream')
    os.environ['TORCH_HOME'] = str(base / 'torch-cache')
    sys.path[:0] = [str(base / 'source'), str(base / 'upstream'), str(base),
                   str(base / 'runtime')]
    import numpy as np
    from PIL import Image
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from src.training.laser_joint_recipe import JointLaserRecipe
    from src.training import rq_imagenet_shared as shared
    from src.training.rq_imagenet_runtime import write_json
    from src.training.rq_sample_panels import designated_preview, MEDIA_KEYS
    from src.training.rq_sampling_override import validated_sampling_recipe

    rank, world, local = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    assert world == plan['world_size'] == 4
    torch.set_num_threads(4)
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if plan.get('memory_gib') is not None:
        torch.cuda.set_per_process_memory_fraction(plan['memory_gib'] * 2**30 /
            torch.cuda.get_device_properties(device).total_memory, device)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=60))
    profile = JointLaserRecipe(base)
    saved = torch.load(plan['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    if plan.get('sampling_override_evidence'):
        evidence = json.loads(Path(plan['sampling_override_evidence']).read_text())
        assert saved['global_step'] >= evidence['source_step']
        saved['recipe'] = validated_sampling_recipe(
            OmegaConf.to_container(profile.recipe, resolve=True), saved['recipe'], evidence)
    profile.verify_resume(saved)
    assert saved['run_id'] == plan['run_id'] and saved['global_step'] == source_step
    assert saved[profile.checkpoint_key] == saved['recipe']['tokenizer']['sha256']
    assert saved['epoch'] == saved['next_epoch'] == source_epoch and saved['next_microbatch'] == 0
    assert len(saved['rng_state_by_rank']) == world
    assert all(int(value['step']) == source_step for value in saved['optimizer']['state'].values())
    assert saved['evaluation']['fid'] == baseline_fid
    assert profile.generation_batch == plan['batch_size']
    with torch.device('meta'):
        model = profile.model(OmegaConf.create(saved['config']['arch']))
    model.load_state_dict(saved['state_dict'], strict=True, assign=True)
    del saved
    model.requires_grad_(False).eval().to(device)
    aux, _ = profile.tokenizer(device)
    profile.adapt(model, aux)
    assert tuple(model.block_size) == (8, 8, 5)
    versions = [parameter._version for parameter in model.parameters()]
    tokenizer_digest = profile.digest(aux)
    original_recipe = OmegaConf.to_container(profile.recipe, resolve=True)
    shared.BASE, shared.RUN, shared.PROFILE = base, profile.run_id, profile
    shared.original_metrics.INCEPTION_PATH = base / 'upstream/rqvae/metrics/inception.py'
    results = dict(plan.get('prior_results', {}))
    for key, previous in results.items():
        assert previous['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert previous['source_step'] == source_step and previous['seed'] == plan['seed']
        assert previous['generated_images'] == 50000 and previous['reference_images'] == 1281167
        assert previous['reference_sha256'] == plan['reference_sha256']
        assert previous['temperature'] == 1. and previous['top_p'] == .95
        assert previous['world_size'] == world and previous['batch_size'] == plan['batch_size']
        assert int(key) == previous['top_k']
    with torch.no_grad():
        for top_k in top_k_values:
            destination = out / f'k{top_k}'
            profile.recipe.sampling = OmegaConf.create(dict(temperature=1., top_k=top_k, top_p=.95))
            shared.OUT = destination
            # Verify the actual training panel before using its baseline FID.
            designated_preview(profile, model, aux, device, rank, world, source_step, None, destination)
            if top_k == baseline_top_k and rank == 0:
                for key in MEDIA_KEYS:
                    name = key.split('/')[-1]
                    previous = Path(plan['training_output']) / f'samples/step{source_step:06d}-{name}.png'
                    current = destination / f'smoke/samples/step{source_step:06d}-{name}.png'
                    assert np.array_equal(np.asarray(Image.open(previous)), np.asarray(Image.open(current))), key
                write_json(out / 'checkpoint-verification.json', dict(passed=True,
                    checkpoint_sha256=plan['checkpoint_sha256'], source_step=source_step,
                    baseline_top_k=baseline_top_k,
                    baseline_preview_pixels_exact=True, preview_images=512, panels_verified=9,
                    strict_model_reload=True, optimizer_age=source_step, training_updates_added=0))
            dist.barrier()
            if rank == 0:
                write_json(out / 'progress.json', dict(phase='fid50k', top_k=top_k,
                    target=50000, completed_policies=list(results), time=time.time()))
            started = time.monotonic()
            if top_k == baseline_top_k and not run_baseline_fid:
                baseline = plan['baseline_evaluation']
                assert baseline['train/global_step'] == source_step
                assert baseline['generated_images'] == 50000 and baseline['real_images'] == 1281167
                assert baseline['real_split'] == 'train' and not baseline['rejection_sampling']
                result = dict(fid=baseline['eval/fid'],
                    inception_score=baseline['eval/inception_score'],
                    inception_score_std=baseline['eval/inception_score_std'],
                    reused_verified_baseline=True)
            else:
                result = shared.generation(model, aux, device, rank, world, source_epoch, None,
                    batch=plan['batch_size'], seed=plan['seed'])
            result.update(top_k=top_k, temperature=1., top_p=.95, seed=plan['seed'],
                generated_images=50000, reference_images=1281167, reference_split='train',
                reference_sha256=plan['reference_sha256'], batch_size=plan['batch_size'],
                source_step=source_step, checkpoint_sha256=plan['checkpoint_sha256'],
                world_size=world, elapsed_seconds=time.monotonic() - started,
                training_updates_added=0, rejection_sampling=False)
            if top_k == baseline_top_k:
                assert abs(result['fid'] - baseline_fid) < 1.e-5, result
                result['original_fid_reproduced'] = run_baseline_fid
            results[str(top_k)] = result
            assert versions == [parameter._version for parameter in model.parameters()]
            assert all(parameter.grad is None for parameter in model.parameters())
            if rank == 0:
                write_json(destination / 'fid50k.json', result)
                print(json.dumps(result), flush=True)
            dist.barrier()
    profile.recipe = OmegaConf.create(original_recipe)
    assert profile.digest(aux) == tokenizer_digest
    if rank == 0:
        report = dict(completed=True, run_id=plan['run_id'],
            source_epoch=source_epoch, source_step=source_step, checkpoint_sha256=plan['checkpoint_sha256'],
            results=results, baseline_top_k=baseline_top_k,
            lower_fid_policy=int(min(results, key=lambda key:results[key]['fid'])),
            fid_minus_baseline={key:result['fid'] - baseline_fid for key,result in results.items()},
            validation=dict(baseline_preview_pixels_exact=True, baseline_fid_reproduced=run_baseline_fid,
                baseline_fid_reused=not run_baseline_fid,
                identical_model_weights=True, frozen_tokenizer_unchanged=True,
                identical_generation_seed_labels_world_size_and_batch=True,
                same_temperature_and_nucleus_threshold=True,
                training_updates_added=0, training_recipe_modified=False),
            limits=['One checkpoint and one generation seed; no training comparison.',
                'Changing sampling top-k does not change the training objective or target noise.'],
            time=time.time())
        if '2048' in results:
            report['fid_minus_k2048'] = {key:result['fid'] - results['2048']['fid']
                                       for key,result in results.items()}
        if baseline_top_k == 512:
            report['validation']['original_fid512_reproduced'] = run_baseline_fid
        if '2048' in results and '512' in results:
            report['fid2048_minus_fid512'] = results['2048']['fid'] - results['512']['fid']
        write_json(out / 'report.json', report)
        write_json(out / 'progress.json', dict(phase='completed', time=time.time()))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
