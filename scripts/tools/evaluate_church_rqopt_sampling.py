#!/usr/bin/env python3
"""Compare coefficient samplers on a frozen Church checkpoint and FID protocol."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    directory = args.request.resolve().parent
    runtime = Path(request['runtime'])
    sys.path.insert(0, str(runtime))
    import numpy as np
    import torch
    import torch.distributed as dist
    from src.training import rqtransformer as training
    from src import rqvae_metrics

    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    assert world == request['world_size']
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(request['memory_fraction'], device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    manifest = json.loads(Path(request['runtime_manifest']).read_text())
    for relative, digest in manifest.items():
        assert sha(runtime / relative) == digest, relative
    checkpoint = Path(request['checkpoint'])
    if rank == 0:
        assert sha(checkpoint) == request['checkpoint_sha256']
        assert sha(request['stage1']) == request['stage1_sha256']
        assert sha(request['reference']) == request['reference_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = payload['config']
    assert payload['epoch'] == request['checkpoint_epoch']
    assert config['compound_pair_autoregressive']
    model = training.build_model(
        config['num_atoms'] + config['coeff_vocab_size'], config['num_atoms'],
        compound=True, coeff_vocab_size=config['coeff_vocab_size'],
        compound_refiner_layers=config['compound_refiner_layers'],
        compound_geometry_head=config['compound_distribution_geometry'],
        compound_micro_transformer_layers=config['compound_micro_transformer_layers'],
        compound_pair_attention=config['compound_pair_attention'],
        compound_depth_specific_coeff_heads=config['compound_depth_specific_coeff_heads'],
        compound_causal_prefix_state=config['causal_prefix_state'],
        compound_pair_autoregressive=config['compound_pair_autoregressive'],
        compound_mask_seen_atoms_training=config['compound_mask_seen_atoms_training'],
        coefficient_history_layers=config.get('coefficient_history_layers', 0),
        coefficient_history_width=config.get('coefficient_history_width', 512),
        sparsity_level=config['sparsity_level'], model_preset=config['model_preset'],
    )
    model.load_state_dict(payload['state_dict'], strict=True)
    model = model.eval().requires_grad_(False).to(device)
    del payload
    aux = training.LaserAux(
        Path(request['stage1']), config['num_atoms'], config['coeff_vocab_size'],
        config['coeff_max'], config['coeff_scale'], attn_resolutions=(8,),
        coeff_scales=config['coeff_scales'], soft_target_physical=True,
        clamp_coeffs=False, coeff_bin_centers=config.get('coeff_bin_centers'),
        sparsity_level=config['sparsity_level'],
    ).eval().requires_grad_(False).to(device)
    original_decode = aux.decode_compound
    original_metric = rqvae_metrics.DistributedOriginalRQVAEMetrics
    current_result = None
    seen = 0
    first_images = None
    previous_progress = 0.

    @torch.no_grad()
    def decode(atoms, coefficients):
        nonlocal seen, first_images, previous_progress
        images = torch.cat([
            original_decode(a, c) for a, c in zip(
                atoms.split(request['decode_batch_size']),
                coefficients.split(request['decode_batch_size']))
        ])
        assert images.dtype == torch.float32 and torch.isfinite(images).all()
        if first_images is None:
            first_images = ((images[:16] + 1.) * .5).clamp(0, 1).cpu()
        seen += len(images)
        if rank == 0 and time.monotonic() - previous_progress > 30:
            write(directory / 'progress.json', dict(
                evaluation=current_result, generated_at_least=seen * world,
                timestamp=time.time(), phase='generating', pid=os.getpid()))
            previous_progress = time.monotonic()
        return images

    aux.decode_compound = decode

    class Metrics(original_metric):
        @torch.no_grad()
        def update(self, images, *, real):
            for chunk in images.split(request['inception_batch_size']):
                super().update(chunk, real=real)

        def _reduce_fid_state(self):
            super()._reduce_fid_state()
            if rank == 0 and current_result is not None:
                mean, covariance = rqvae_metrics._mean_covariance(
                    self.fake_sum, self.fake_cross, int(self.fake_count.item()))
                np.savez(directory / (current_result + '-statistics.npz'),
                         mu=mean, sigma=covariance)

    rqvae_metrics.DistributedOriginalRQVAEMetrics = Metrics
    seed_streams = [{seed + r for r in range(world)} for seed in request['seeds']]
    assert all(a.isdisjoint(b) for i, a in enumerate(seed_streams) for b in seed_streams[i + 1:])
    sampler = dict(atom_temperature=1., atom_top_k=250, atom_top_p=1.,
                   coeff_temperature=1., coeff_top_k=0)
    with torch.no_grad():
        torch.manual_seed(2026092490 + rank)
        atoms, coefficients = model.sample_compound(
            request['batch_size'], aux,
            cond=torch.zeros(request['batch_size'], device=device, dtype=torch.long),
            amp=True, coeff_top_p=.85, **sampler)
        images = decode(atoms, coefficients)
        check_metric = Metrics(device, compute_inception_score=False,
                               reference_stats_path=request['reference'])
        check_metric.update(((images + 1.) * .5).clamp(0, 1), real=False)
        assert int(check_metric.fake_count.item()) == request['batch_size']
        assert torch.isfinite(check_metric.fake_sum).all()
        del check_metric, images, atoms, coefficients
    torch.cuda.synchronize()
    peak = torch.tensor(torch.cuda.max_memory_allocated(), device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    if rank == 0:
        write(directory / 'preflight.json', dict(
            passed=True, strict_checkpoint_load=True,
            parameters=sum(p.numel() for p in model.parameters()),
            generated=world * request['batch_size'],
            peak_gpu_bytes=int(peak.item()), checkpoint_sha256=request['checkpoint_sha256']))
        print(json.dumps({'event': 'preflight_passed', 'peak_gpu_bytes': int(peak.item())}), flush=True)
    if args.preflight_only:
        dist.destroy_process_group()
        return
    wb = None
    if rank == 0:
        import wandb
        wb = wandb.init(entity='helloimlixin-rutgers', project='laser',
                        id=request['run_id'], name=request['run_id'], resume='allow',
                        mode='online', job_type='controlled-sampling-evaluation',
                        config=request, dir=request['wandb_directory'])
        write(directory / 'wandb.json', {'url': wb.url, 'id': wb.id})
        wb.use_artifact('helloimlixin-rutgers/laser/church-laser-stage1-selection-20260920-selected-checkpoints:v0')
    dist.barrier()
    results = []
    for seed in request['seeds']:
        for name, top_p in [('baseline', .85), ('unfiltered', 1.)]:
            current_result = f'{name}-seed{seed}'
            seen, first_images = 0, None
            random.seed(seed + rank)
            np.random.seed(seed + rank)
            torch.manual_seed(seed + rank)
            torch.cuda.manual_seed(seed + rank)
            started = time.monotonic()
            dist.barrier()
            fid, _, _ = training.evaluate_generation_metrics(
                model, aux, None, request['samples'], batch_size=request['batch_size'],
                num_condition_classes=1, compute_inception_score=False,
                metric_backend='original-rqvae', fid_reference_stats=request['reference'],
                coeff_top_p=top_p, **sampler)
            assert seen == request['samples'] // world + (rank < request['samples'] % world)
            assert math.isfinite(fid)
            row = dict(setting=name, coeff_top_p=top_p, seed=seed, fid=fid,
                       samples=request['samples'], seconds=time.monotonic() - started)
            results.append(row)
            previews = [torch.empty_like(first_images, device=device) for _ in range(world)]
            dist.all_gather(previews, first_images.to(device))
            if rank == 0:
                image_path = directory / (current_result + '.png')
                training.save_unlabeled_grid(torch.cat(previews).cpu(), image_path, nrow=8)
                write(directory / (current_result + '.json'), row)
                write(directory / 'results.json', results)
                wb.log({**{f'evaluation/{k}': v for k, v in row.items()},
                        'evaluation/samples_grid': wandb.Image(str(image_path))})
                print(json.dumps({'event': 'evaluation_complete', **row}), flush=True)
            del previews
            dist.barrier()
    if rank == 0:
        means = {name: sum(r['fid'] for r in results if r['setting'] == name) / len(request['seeds'])
                 for name in ['baseline', 'unfiltered']}
        paired_gains = [next(r['fid'] for r in results if r['seed'] == seed and r['setting'] == 'baseline')
                        - next(r['fid'] for r in results if r['seed'] == seed and r['setting'] == 'unfiltered')
                        for seed in request['seeds']]
        selected = 'unfiltered' if all(gain > 0 for gain in paired_gains) else 'baseline'
        report = dict(results=results, mean_fid=means, paired_improvements=paired_gains,
                      selected=selected, selection_rule='Promote only when both seeds improve',
                      checkpoint_sha256=request['checkpoint_sha256'],
                      reference_sha256=request['reference_sha256'],
                      limitation='Two generation seeds on one checkpoint; no training or structural-quality claim.')
        write(directory / 'report.json', report)
        write(directory / 'selected-sampling.json', dict(
            **sampler, coeff_top_p=1. if selected == 'unfiltered' else .85,
            checkpoint=request['checkpoint'], checkpoint_sha256=request['checkpoint_sha256'],
            reference=request['reference'], report=str(directory / 'report.json')))
        wb.summary.update({'selected': selected, 'mean_baseline_fid': means['baseline'],
                           'mean_unfiltered_fid': means['unfiltered'],
                           'mean_fid_improvement': means['baseline'] - means['unfiltered']})
        artifact = wandb.Artifact(request['run_id'] + '-results', type='evaluation')
        for path in directory.iterdir():
            if path.suffix in {'.json', '.png', '.npz', '.py'}:
                artifact.add_file(str(path), name=path.name)
        wb.log_artifact(artifact, aliases=['latest']).wait()
        write(directory / 'complete.json', dict(passed=True, selected=selected,
                                                artifact=artifact.qualified_name, completed_unix=time.time()))
        wb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
