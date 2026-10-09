"""Evaluate a fixed physical-pair ImageNet checkpoint with the historical K2 sampler."""

import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--wandb-id', required=True)
    parser.add_argument('--generation-batch-size', type=int, default=128)
    parser.add_argument('--memory-limit-gib', type=float, default=36.)
    parser.add_argument('--seed', type=int, default=261001)
    args = parser.parse_args()
    if args.generation_batch_size < 64 or args.memory_limit_gib <= 0:
        parser.error('Require generation batch >=64 and a positive GPU memory cap')
    sys.path[:0] = [str(args.runtime_root), str(args.runtime_root / 'runtime')]

    import torch
    import torch.distributed as dist
    from src.training import rqtransformer as training
    from src.training.comparison_fid import ComparisonFIDMetrics
    from src.training.fid_reference import fixed_evaluation_rng
    from src import rqvae_metrics

    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('Use the same four ranks as the baseline evaluation')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_gib * 2**30 / total, device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))

    baseline = json.loads(args.baseline.read_text())
    payload = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    epoch, step = int(payload['epoch']), int(payload['global_step'])
    if (epoch, step) != (6, 3756) or baseline['train/global_step'] != step:
        raise ValueError('Expected the epoch-6 checkpoint and its matching baseline')
    config = payload['config']
    if not config['physical_pair_context'] or config['compound_tokens']:
        raise ValueError('Expected physical-pair scalar checkpoint')
    with torch.device('meta'):
        model = training.build_model(config['num_atoms'] + config['coeff_vocab_size'],
            config['num_atoms'], coeff_vocab_size=config['coeff_vocab_size'],
            sparsity_level=config['sparsity_level'], physical_pair_context=True,
            model_preset=config['model_preset'])
    model.load_state_dict(payload['state_dict'], strict=True, assign=True)
    model.requires_grad_(False).to(device).eval()
    del payload
    assert sum(p.numel() for p in model.parameters()) == 1391864832
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    aux = training.LaserAux(args.inputs / 'stage1-tokenizer.pt', config['num_atoms'],
        config['coeff_vocab_size'], config['coeff_max'], config['coeff_scale'],
        attn_resolutions=(8,), coeff_scales=config['coeff_scales'],
        soft_target_physical=False, clamp_coeffs=False,
        sparsity_level=config['sparsity_level']).to(device).eval().requires_grad_(False)

    comparisons = {}
    def comparison_log(values):
        comparisons.update(values)
        record(args.output / 'comparison.json', dict(values, checkpoint_epoch=epoch,
            checkpoint_global_step=step, sampling='k2'))

    def metrics(*values, **kwargs):
        return ComparisonFIDMetrics(*values, **kwargs,
            validation_reference=args.inputs / 'imagenet_val_original.npz',
            torchmetrics_reference=args.inputs / 'imagenet_val_torchmetrics.pt',
            selection_metric='torchmetrics_val50k', on_comparison=comparison_log)
    rqvae_metrics.DistributedOriginalRQVAEMetrics = metrics
    sampling = dict(atom_temperature=1., atom_top_k=0, atom_top_p=.92,
        coeff_temperature=1., coeff_top_k=0, coeff_top_p=.92)

    # Exercise the exact production batch and decoder chunk under the allocator
    # cap before starting the full corpus. The fixed RNG context restores all
    # random streams, so this probe cannot change the evaluation samples.
    with torch.no_grad(), fixed_evaluation_rng(args.seed, device, rank=rank):
        labels = (torch.arange(args.generation_batch_size, device=device) * world + rank) % 1000
        tokens = model.sample_sparse(args.generation_batch_size, aux, cond=labels, **sampling, amp=True)
        atoms = tokens[..., 0::2]
        assert bool((atoms.sort(-1).values.diff(dim=-1) > 0).all())
        images = aux.decode_tokens(tokens[:64]).float().add(1).mul(.5).clamp(0, 1)
        assert bool(torch.isfinite(images).all())
        probe = metrics(device, compute_inception_score=True,
            reference_stats_path=args.inputs / 'imagenet_256_train.npz')
        probe.update(images, real=False)
        assert int(probe.fake_count) == 64
        assert int(probe.companion.fake_features_num_samples) == 64
        del probe, tokens, atoms, images
    record(args.output / f'preflight-rank{rank}.json', dict(passed=True, pid=os.getpid(),
        checkpoint_epoch=epoch, checkpoint_global_step=step, strict_state_restore=True,
        finite_model=True, generation_batch=args.generation_batch_size, decode_batch=64,
        memory_limit_gib=args.memory_limit_gib,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30, time=time.time()))
    torch.cuda.empty_cache()
    dist.barrier()

    run = None
    if rank == 0:
        provenance = dict(parent_run='helloimlixin-rutgers/laser/imagenet-rfid421-sigma200bins-scratch-4h200-20261008',
            checkpoint=str(args.checkpoint), checkpoint_epoch=epoch, checkpoint_global_step=step,
            checkpoint_sha256=sha256(args.checkpoint), evaluator_sha256=sha256(Path(__file__)),
            runtime_root=str(args.runtime_root), seed=args.seed, world_size=world,
            generated_images=50000, validation_real_images=50000,
            generation_batch_per_rank=args.generation_batch_size, decode_batch=64,
            sampling=sampling, baseline_fid=baseline['eval/fid_torchmetrics_val50k'],
            reference_sha256=sha256(args.inputs / 'imagenet_val_torchmetrics.pt'),
            concurrent_training=True, memory_limit_gib=args.memory_limit_gib)
        record(args.output / 'provenance.json', provenance)
        os.environ['WANDB_API_KEY'] = args.key_file.read_text().strip()
        import wandb
        run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=args.wandb_id,
            resume='allow', mode='online', job_type='sampler-evaluation',
            dir=str(args.checkpoint.parent), config=provenance)
        record(args.output / 'status.json', dict(state='evaluating', wandb_url=run.url,
            checkpoint_epoch=epoch, checkpoint_global_step=step, time=time.time()))
        run.summary['execution/state'] = 'evaluating'
    dist.barrier()

    started = time.monotonic()
    fid, score, std = training.evaluate_generation_metrics(model, aux, None, 50000,
        args.generation_batch_size, num_condition_classes=1000, **sampling,
        metric_backend='original-rqvae', compute_inception_score=True,
        fid_reference_stats=args.inputs / 'imagenet_256_train.npz', fid_seed=args.seed)
    if rank == 0:
        assert comparisons['eval/comparison_generated_images'] == 50000
        assert comparisons['eval/comparison_real_images'] == 50000
        assert math.isfinite(fid) and fid == comparisons['eval/fid_torchmetrics_val50k']
        old_fid = baseline['eval/fid_torchmetrics_val50k']
        k2_fid = 47.88251876831055
        result = dict(checkpoint_epoch=epoch, checkpoint_global_step=step, sampling=sampling,
            **comparisons, fid=fid, inception_score=score, inception_score_std=std,
            baseline_fid=old_fid, fid_change_from_baseline=fid - old_fid,
            historical_k2_epoch6_fid=k2_fid, fid_gap_to_historical_k2=fid - k2_fid,
            seed=args.seed, generation_batch_per_rank=args.generation_batch_size,
            elapsed_seconds=time.monotonic() - started, completed_unix=time.time(), wandb_url=run.url)
        record(args.output / 'result.json', result)
        run.log(dict(comparisons, **{'eval/fid':fid, 'eval/inception_score':score,
            'eval/inception_score_std':std, 'comparison/baseline_fid':old_fid,
            'comparison/fid_change':fid - old_fid, 'comparison/k2_epoch6_fid':k2_fid,
            'comparison/k2_epoch6_gap':fid - k2_fid}))
        for path in [Path(__file__), args.output / 'provenance.json', args.output / 'result.json',
                     args.output / 'comparison.json', *args.output.glob('preflight-rank*.json')]:
            if path.parent != args.output:
                run.save(str(path), base_path=str(path.parent), policy='now')
            else:
                run.save(str(path), base_path=str(args.output), policy='now')
        run.summary['execution/state'] = 'completed'
        run.finish()
        record(args.output / 'status.json', dict(state='completed', time=time.time(),
            fid=fid, checkpoint_epoch=epoch, checkpoint_global_step=step, wandb_url=result['wandb_url']))
        print(json.dumps(result), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
