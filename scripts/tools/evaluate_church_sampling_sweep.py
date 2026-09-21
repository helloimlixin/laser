#!/usr/bin/env python3
"""Compare samplers on one fixed checkpoint, using the frozen training runtime."""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/church-consistent-rqvae-20260914'
SNAPSHOT = BASE / 'stage2-source'
UPSTREAM = SNAPSHOT / 'outputs/church-rq-baseline-scratch-20260912/upstream-source'
sys.path[:0] = [str(UPSTREAM), str(SNAPSHOT), str(ROOT)]
import src
src.__path__ = [str(SNAPSHOT / 'src')]
os.environ.setdefault('TORCH_HOME', '/workspace/tmp/official-rqvae-eval-cache')

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchvision.utils import save_image
from rqvae.models import create_model
from rqvae.metrics.fid import get_inception_model
import rqvae.models.rqtransformer.transformers as transformer_module
from rqvae.utils.utils import top_k_logits, top_p_probs
from src.compact_rq_training import FrozenCompactTokenizer
from src.original_rq_training import atomic_json, file_sha256, FeatureMoments, fid_from_moments


SETTINGS = {
    'baseline': dict(temperature=1., top_k=1400, top_p=1.),
    'wider_2800': dict(temperature=1., top_k=2800, top_p=1.),
    'wider_5600': dict(temperature=1., top_k=5600, top_p=1.),
    'cooler': dict(temperature=.9, top_k=1400, top_p=1.),
    'warmer': dict(temperature=1.1, top_k=1400, top_p=1.),
    'nucleus_95': dict(temperature=1., top_k=None, top_p=.95),
    'nucleus_98': dict(temperature=1., top_k=None, top_p=.98),
    'depth_dependent_k': dict(temperature=1., top_k=[1400, 1400, 2800, 2800], top_p=1.),
}


def validate_setting(setting):
    if not math.isfinite(setting['temperature']) or setting['temperature'] <= 0:
        raise ValueError('Sampling temperature must be finite and positive')
    if not 0 < setting['top_p'] <= 1:
        raise ValueError('top_p must be in (0, 1]')
    ks = setting['top_k']
    if isinstance(ks, list) and len(ks) != 4:
        raise ValueError('A depth-specific top_k must have four entries')
    for k in ks if isinstance(ks, list) else [ks]:
        if k is not None and (not isinstance(k, int) or k < 1):
            raise ValueError('top_k must be positive or null')


@contextmanager
def observe_filter(statistics):
    """Observe generated prefixes without changing the released draw or its RNG."""
    original = transformer_module.sample_from_logits
    position = 0

    def observed(logits, temperature=1., top_k=None, top_p=None):
        nonlocal position
        if statistics is not None:
            scores = logits.float() / temperature
            full = scores.softmax(-1)
            filtered = (top_k_logits(scores, top_k) if top_k is not None else scores).softmax(-1)
            if top_p is not None:
                filtered = top_p_probs(filtered, top_p)
            kept = filtered > 0
            statistics[position % len(statistics)] += torch.stack([
                (full * kept).sum(),
                kept.sum().to(full.dtype),
                -(full * full.clamp_min(1e-30).log()).sum(),
                -(filtered * filtered.clamp_min(1e-30).log()).sum(),
                full.new_tensor(len(logits)),
            ]).double()
        position += 1
        return original(logits, temperature=temperature, top_k=top_k, top_p=top_p)

    transformer_module.sample_from_logits = observed
    try:
        yield
    finally:
        transformer_module.sample_from_logits = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--diagnostic', action='store_true',
                        help='Allow a smaller sample count for an explicitly labeled diagnostic')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--decode-batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=71000)
    parser.add_argument('--settings', nargs='+', choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument('--memory-fraction', type=float, default=.08)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    if args.samples != 50000 and not args.diagnostic:
        parser.error('Reported FID requires 50000 generated samples; use --diagnostic for a small check')
    for name in args.settings:
        validate_setting(SETTINGS[name])
    if args.samples < 2 or min(args.batch_size, args.decode_batch_size) < 1:
        raise ValueError('At least two samples and positive batch sizes required')
    if not 0 < args.memory_fraction <= .15:
        raise ValueError('Use a memory fraction in (0, .15] alongside training')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    if args.samples < world * args.batch_size:
        raise ValueError('Each rank must have a full first batch for the shared grid')
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dist.init_process_group('nccl', timeout=timedelta(hours=1))
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    started = time.time()
    run = None

    def status(phase, **values):
        if rank == 0:
            row = dict(phase=phase, pid=os.getpid(), updated_unix=time.time(),
                       elapsed_seconds=time.time()-started, **values)
            atomic_json(output / 'status.json', row)
            print(json.dumps(row), flush=True)

    try:
        manifest = json.loads((BASE / 'stage2-source-manifest.json').read_text())
        if rank == 0:
            for name, digest in manifest.items():
                assert file_sha256(SNAPSHOT / name) == digest, name
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        config = OmegaConf.create(payload['config'])
        cache = payload['tokenizer']
        model, _ = create_model(config.arch, ema=False)
        model.load_state_dict(payload['state_dict'], strict=True)
        epoch, step = payload['epoch'], payload['step']
        del payload
        reference = BASE / 'reference/real-statistics.npz'
        protocol = BASE / 'reference/data-protocol.json'
        reference_receipt = json.loads((BASE / 'reference/complete.json').read_text())
        data_protocol = json.loads(protocol.read_text())
        real_samples = data_protocol['cache_stage2_fid']['datasets']['church']['images']
        assert data_protocol['verified'] and reference_receipt['phase'] == 'complete'
        assert reference_receipt['images'] == real_samples == cache['images']
        assert reference_receipt['padded_or_dropped_images'] == 0
        metric_name = (f'diagnostic/fid_{args.samples}_vs_full_train' if args.diagnostic
                       else 'fid_50000_vs_full_train')
        if rank == 0:
            assert file_sha256(cache['checkpoint']) == cache['checkpoint_sha256']
            assert file_sha256(cache['codebook']) == cache['codebook_sha256']
            assert file_sha256(reference) == reference_receipt['sha256']
            assert file_sha256(protocol) == reference_receipt['data_protocol_sha256']
        dist.barrier()
        model.requires_grad_(False).to(device).eval()
        tokenizer = FrozenCompactTokenizer(cache['checkpoint'], cache['codebook']).to(device).eval()
        inception = get_inception_model().requires_grad_(False).to(device).eval()
        specification = None
        if rank == 0:
            specification = dict(
                checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_sha256(args.checkpoint),
                checkpoint_epoch=epoch, checkpoint_step=step, tokenizer_checkpoint=cache['checkpoint'],
                tokenizer_sha256=cache['checkpoint_sha256'], codebook_sha256=cache['codebook_sha256'],
                reference=str(reference), reference_sha256=reference_receipt['sha256'],
                real_samples=real_samples, reference_split='entire training set',
                diagnostic=args.diagnostic, metric_name=metric_name,
                data_protocol_sha256=reference_receipt['data_protocol_sha256'],
                samples=args.samples, seed=args.seed, seed_rule='seed + rank, reset for each setting',
                world_size=world, batch_size_per_rank=args.batch_size,
                decode_and_inception_batch_size=args.decode_batch_size,
                memory_fraction_per_process=args.memory_fraction,
                settings={name: SETTINGS[name] for name in args.settings},
                precision='original sampler FP16 autocast; decoder/Inception FP32; FP64 moments; TF32 off',
                pixels='continuous RGB [0,1] clamp; no additional resize/crop; released Inception preprocessing',
                evaluator_sha256=file_sha256(__file__), frozen_source_manifest=manifest,
                diagnostics='generated prefixes from the first batch on each rank',
            )
            spec_path = output / 'specification.json'
            if spec_path.exists():
                assert json.loads(spec_path.read_text()) == specification, 'Cannot mix sweep protocols'
            else:
                atomic_json(spec_path, specification)
                (output / 'evaluator.py').write_text(Path(__file__).read_text())
            if not args.offline:
                import wandb
                run = wandb.init(entity='helloimlixin-rutgers', project='laser', id=args.run_id,
                                 name=args.run_id, resume='allow', dir=str(output), config=specification)
                atomic_json(output / 'wandb.json', dict(id=run.id, url=run.url))
        dist.barrier()
        results_path = output / 'results.json'
        results = json.loads(results_path.read_text())['results'] if results_path.exists() else {}
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for index, name in enumerate(args.settings):
                if name in results:
                    continue
                setting = SETTINGS[name]
                folder = output / name
                if rank == 0:
                    folder.mkdir(exist_ok=True)
                dist.barrier()
                moments = FeatureMoments(device)
                usage = torch.zeros(4, tokenizer.quantizer.vocab_size, dtype=torch.long, device=device)
                diagnostics = torch.zeros(4, 5, dtype=torch.float64, device=device)
                saved_codes = []
                local_total = len(range(rank, args.samples, world))
                torch.manual_seed(args.seed + rank)
                setting_started = time.time()
                status('sampling', setting=name, setting_index=index, samples=args.samples, completed=0)
                for offset in range(0, local_total, args.batch_size):
                    size = min(args.batch_size, local_total-offset)
                    empty = torch.zeros(size, 8, 8, 4, dtype=torch.long, device=device)
                    with observe_filter(diagnostics if offset == 0 else None):
                        codes = model.sample(empty, model_aux=tokenizer, amp=True,
                                             cached=True, is_tqdm=False, **setting)
                    saved_codes.append(codes.cpu().to(torch.int32))
                    for depth in range(4):
                        usage[depth] += torch.bincount(codes[..., depth].flatten(), minlength=usage.shape[1])
                    previews = []
                    for code_batch in codes.split(args.decode_batch_size):
                        images = tokenizer.decode_code(code_batch).mul(.5).add(.5).clamp(0, 1)
                        if not torch.isfinite(images).all():
                            raise FloatingPointError('Nonfinite decoded samples')
                        moments.update(inception(images))
                        if offset == 0:
                            previews.append(images)
                    if offset == 0:
                        local_images = torch.cat(previews)
                        images_by_rank = [torch.empty_like(local_images) for _ in range(world)]
                        dist.all_gather(images_by_rank, local_images)
                        if rank == 0:
                            ordered = torch.stack(images_by_rank, 1).flatten(0, 1)
                            save_image(ordered[:64], folder / 'samples.png', nrow=8)
                        del previews, local_images, images_by_rank
                    if offset == 0 or (offset // args.batch_size) % 2 == 1:
                        status('sampling', setting=name, setting_index=index, samples=args.samples,
                               completed=min(args.samples, (offset+size)*world),
                               setting_elapsed_seconds=time.time()-setting_started,
                               peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                               peak_gpu_reserved_gib=torch.cuda.max_memory_reserved(device)/1024**3)
                torch.save(dict(codes=torch.cat(saved_codes),
                                sample_indices=torch.arange(rank, args.samples, world)),
                           folder / f'codes-rank{rank}.pt')
                measured = moments.finish()
                dist.all_reduce(usage)
                dist.all_reduce(diagnostics)
                assert measured[0] == args.samples
                if rank == 0:
                    status('computing_fid', setting=name, samples=args.samples)
                    score = fid_from_moments(measured, reference)
                    np.savez(folder / 'statistics.npz', mu=measured[1], sigma=measured[2], samples=args.samples)
                    torch.save(usage.cpu(), folder / 'token-usage.pt')
                    counts = diagnostics[:, 4]
                    by_depth = [dict(depth=d, positions=int(counts[d]),
                        retained_probability_mass=float(diagnostics[d, 0]/counts[d]),
                        mean_candidate_count=float(diagnostics[d, 1]/counts[d]),
                        entropy_before_filter=float(diagnostics[d, 2]/counts[d]),
                        entropy_after_filter=float(diagnostics[d, 3]/counts[d]),
                        zero_fraction=float(usage[d, 0]/usage[d].sum()),
                        distinct_tokens=int((usage[d] > 0).sum())) for d in range(4)]
                    results[name] = dict(fid=score, samples=args.samples, real_samples=real_samples,
                                         metric_name=metric_name, settings=setting,
                                         elapsed_seconds=time.time()-setting_started, depths=by_depth)
                    atomic_json(results_path, dict(specification=specification, results=results))
                    status('setting_complete', setting=name, fid=score, samples=args.samples,
                           setting_elapsed_seconds=time.time()-setting_started)
                    if run:
                        import wandb
                        run.log(dict(setting_index=index, setting=name,
                                     **{metric_name: score, f'{metric_name}/{name}': score},
                                     samples=wandb.Image(str(folder / 'samples.png'), caption=json.dumps(setting))))
                        run.summary['results'] = results
                        run.summary['best_setting'] = min(results, key=lambda k: results[k]['fid'])
                        run.summary['best_fid'] = min(row['fid'] for row in results.values())
                dist.barrier()
                # Every rank loads the same completed journal before another setting.
                results = json.loads(results_path.read_text())['results']
                del moments, usage, diagnostics, saved_codes
        status('complete', checkpoint_epoch=epoch, checkpoint_step=step,
               results={k: v['fid'] for k, v in results.items()})
        if run:
            run.summary['status'] = 'complete'
            run.finish()
    except BaseException as error:
        atomic_json(output / f'failure-rank{rank}.json',
                    dict(type=type(error).__name__, error=str(error), updated_unix=time.time()))
        if run:
            run.summary['status'] = 'failed'
            run.finish(exit_code=1)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
