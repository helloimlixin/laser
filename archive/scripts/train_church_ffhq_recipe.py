#!/usr/bin/env python3
"""Train Church compound priors from scratch with the recovered FFHQ-v4 recipe."""
import argparse
import codecs
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file
from src.training.rqtransformer import (
    LaserAux, atomic_torch_save, compound_objective, scheduled_geometry_weight,
)
from src.church_ffhq_recipe import ARCHITECTURES, make_prior, early_decay_lr, recipe_targets
from src.coefficient_history_training import EpochStream
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))
    tmp.replace(path)


def objective(model, aux, atoms, physical, geometry_weight, stochastic=True):
    packed, probs = recipe_targets(aux, atoms, physical, stochastic=stochastic)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(packed, model_aux=aux, amp=True)
    target_physical = aux.dictionary.t()[atoms.long()] * physical[..., None]
    loss, values = compound_objective(
        outputs['atom_logits'], outputs['coeff_logits'], None, atoms, probs, target_physical,
        atom_weight=1.5, geometry_weight=geometry_weight, accumulation=1,
        distribution_geometry=True, geometry_dictionary=aux.dictionary,
        geometry_coeff_bins=aux.coeff_bins, geometry_coeff_scales=aux.coeff_scales,
        geometry_top_k=4,
    )
    with torch.no_grad():
        entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
        cp = outputs['coeff_logits'].float().softmax(-1)
        coeff_pred = (cp * aux.coeff_bins).sum(-1) * aux.coeff_scales
        coeff_log = outputs['coeff_logits'].float().log_softmax(-1)
        nearest = packed.remainder(aux.coeff_vocab_size)
        hard_nll = -coeff_log.gather(-1, nearest[..., None]).squeeze(-1)
        sign_prob = cp[..., aux.coeff_vocab_size // 2:].sum(-1)
        metrics = {'loss': loss.detach(), 'atom_nll': values['atom_nll'].mean(),
                   'coeff_cross_entropy': values['coeff_cross_entropy'].mean(),
                   'coeff_target_entropy': entropy.mean(),
                   'coeff_kl': (values['coeff_cross_entropy'] - entropy).mean(),
                   'coefficient_sample_nll' if stochastic else 'coefficient_nearest_nll': hard_nll.mean(),
                   'coefficient_mean_mae': (coeff_pred - physical).abs().mean(),
                   'sign_accuracy': ((sign_prob >= .5) == (physical >= 0)).float().mean(),
                   'geometry': values['geometry'].detach()}
        for d in range(atoms.shape[-1]):
            metrics[f'atom_nll_d{d}'] = values['atom_nll'][..., d].mean()
            metrics[f'coeff_kl_d{d}'] = (values['coeff_cross_entropy'] - entropy)[..., d].mean()
    return loss, {k: float(v) for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, aux, data, batch=16):
    model.eval()
    total = {}
    for first in range(0, len(data['atoms']), batch):
        atoms = data['atoms'][first:first + batch].cuda().long()
        physical = data['coefficients'][first:first + batch].cuda()
        _, metrics = objective(model, aux, atoms, physical, .05, stochastic=False)
        for key, value in metrics.items():
            total[key] = total.get(key, 0.) + value * len(atoms)
    return {k: v / len(data['atoms']) for k, v in total.items()}


@torch.no_grad()
def generate_metric(model, aux, args, output, count, seed, log):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    metric = DistributedOriginalRQVAEMetrics('cuda', reference_stats_path=args.fid_stats)
    saved_atoms, saved_ids = [], []
    start = time.monotonic()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        for first in range(0, count, args.generation_batch):
            batch = min(args.generation_batch, count - first)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                atoms, ids = model.sample_compound(
                    batch, aux, atom_top_k=args.atom_top_k, atom_top_p=1.,
                    coeff_top_k=0, coeff_top_p=.85, atom_temperature=1.,
                    coeff_temperature=1., amp=True,
                )
            saved_atoms.append(atoms.cpu().short())
            saved_ids.append(ids.cpu().short())
            for offset in range(0, batch, 32):
                images = (aux.decode_compound(atoms[offset:offset + 32], ids[offset:offset + 32]).float() + 1) / 2
                metric.update(images.clamp(0, 1), real=False)
                if first == 0 and offset == 0:
                    save_image(images.clamp(0, 1), output / 'samples.png', nrow=8)
            if first == 0 or (first // args.generation_batch) % 5 == 0 or first + batch == count:
                log({'phase': 'generation', 'generated': first + batch, 'target_samples': count,
                     'generation_seconds': time.monotonic() - start})
    fid, _, _ = metric.compute()
    result = {'fid': float(fid), 'samples': count, 'seed': seed, 'atom_top_k': args.atom_top_k,
              'atom_top_p': 1., 'coeff_top_p': .85, 'temperature': 1.,
              'precision': 'BF16 AR; FP32 decoder; original RQ-VAE Inception',
              'seconds': time.monotonic() - start}
    atomic_torch_save({'atoms': torch.cat(saved_atoms), 'coefficient_ids': torch.cat(saved_ids)}, output / 'generated-codes.pt')
    write_json(output / 'metrics.json', result)
    del metric
    model.init_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--architecture', choices=list(ARCHITECTURES), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    p.add_argument('--fid-stats', type=Path, default=ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--microbatch', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--knee-lr', type=float, default=5e-5)
    p.add_argument('--min-lr', type=float, default=2e-6)
    p.add_argument('--warmup-epochs', type=float, default=1.)
    p.add_argument('--knee-epoch', type=float, default=30.)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--full-fid-every', type=int, default=50)
    p.add_argument('--fid-samples', type=int, default=4096)
    p.add_argument('--full-fid-samples', type=int, default=50000)
    p.add_argument('--generation-batch', type=int, default=128)
    p.add_argument('--atom-top-k', type=int, default=2048)
    p.add_argument('--seed', type=int, default=5701)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-step', type=int, default=0)
    p.add_argument('--smoke-evaluate', action='store_true')
    p.add_argument('--wandb-id')
    args = p.parse_args()
    early_decay_lr(0, args.epochs, args.lr, args.knee_lr, args.min_lr, args.warmup_epochs, args.knee_epoch)
    if not (0 < args.microbatch <= 64 and args.batch_size > 0):
        p.error('microbatch must be 1..64; larger BF16 classifier backward GEMMs fail on this runtime')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    raw = torch.load(args.cache, weights_only=False, map_location='cpu')
    assert raw['meta']['format'] == 'church_ffhq_continuous_v1'
    assert raw['meta']['checkpoint_sha256'] == sha256_file(args.stage1)
    model = make_prior(args.architecture).cuda()
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=False).cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95),
                                 weight_decay=args.weight_decay, fused=True)
    stream = EpochStream(len(raw['train']['atoms']), args.seed + 1)
    config = json.loads(json.dumps({**vars(args), 'model': ARCHITECTURES[args.architecture],
        'parameters': sum(p.numel() for p in model.parameters()),
        'initialization': 'random; no stage-2 checkpoint loaded', 'tokenizer': raw['meta'],
        'recipe_source': 'helloimlixin-rutgers/laser/ffhqcmp0804205803',
        'recipe': 'full pair AR; normalized soft targets T=.5; stochastic context; atom weight 1.5; distribution geometry .05, delay 2/ramp 3',
        'train_images': stream.size, 'holdout_images': len(raw['holdout']['atoms']),
        'planned_steps': args.epochs * math.ceil(stream.size / args.batch_size),
        'cache_sha256': sha256_file(args.cache), 'fid_stats_sha256': sha256_file(args.fid_stats),
        'selection': 'best fixed-seed FID-4096, confirmed by fresh-seed FID-50000; full FID also every 50 epochs',
        'holdout_note': 'stage-2-disjoint 1024 images; frozen stage-1 saw training population',
        'source_hashes': {name: sha256_file(ROOT / name) for name in [
            'scripts/train_church_ffhq_recipe.py', 'src/church_ffhq_recipe.py',
            'src/coefficient_history_training.py', 'scripts/train_official_rqtransformer_laser_stage2.py',
            'src/models/rqtransformer/transformers.py', 'src/models/rqtransformer/attentions.py']},
    }, default=str))
    step, best_fid, best_epoch, elapsed = 0, None, None, 0.
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=False)
        ignored = {'resume', 'stop_after_step', 'smoke_evaluate', 'wandb_id'}
        for key in config.keys() - ignored:
            if config[key] != saved['config'][key]:
                raise ValueError(f'Resume setting changed: {key}')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        stream.load_state_dict(saved['stream'])
        step, best_fid, best_epoch, elapsed = [saved[k] for k in ('step', 'best_fid', 'best_epoch', 'elapsed_seconds')]
        del saved
    write_json(args.output / 'config.json', config)
    wb = None
    if args.wandb_id:
        import wandb
        wb = wandb.init(entity='helloimlixin-rutgers', project='laser', id=args.wandb_id,
                        name=args.wandb_id, group='church-ffhq-recipe-20260911',
                        job_type='compound-stage2-scratch', dir=str(args.output), config=config,
                        resume='must' if args.resume else 'never')
    start = time.monotonic()
    stopping = {'signal': None}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopping.update(signal=number))

    def log(row):
        full = {'optimizer_step': step, 'epoch_progress': stream.epoch + stream.position / stream.size,
                'elapsed_seconds': elapsed + time.monotonic() - start, **row}
        print(json.dumps(full, allow_nan=False), flush=True)
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(full, allow_nan=False) + '\n')
        write_json(args.output / 'status.json', {'pid': os.getpid(), 'planned_epochs': args.epochs,
                    'best_screen_fid': best_fid, 'best_epoch': best_epoch, **full})
        if wb:
            wb.log(full)

    def save_last():
        model.init_cache()
        atomic_torch_save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(), 'config': config, 'step': step, 'best_fid': best_fid,
            'best_epoch': best_epoch, 'elapsed_seconds': elapsed + time.monotonic() - start}, args.output / 'last.pt')
        log({'phase': 'checkpoint_saved'})

    def validate(epoch):
        with torch.random.fork_rng(devices=[0]):
            values = {s: evaluate(model, aux, raw[s]) for s in ('holdout', 'validation')}
            probe = {k: v[:256] for k, v in raw['train'].items() if k != 'keys'}
            values['train_probe'] = evaluate(model, aux, probe)
        directory = args.output / f'evaluations/epoch-{epoch:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'teacher-forcing.json', values)
        log({'phase': 'validation', **{f'{s}/{k}': v for s, rows in values.items() for k, v in rows.items()}})
        return directory

    log({'phase': 'ready', 'parameters': config['parameters'], 'resumed': args.resume})
    try:
        if args.smoke_evaluate:
            data = {k: v[:16] for k, v in raw['holdout'].items() if k != 'keys'}
            log({'phase': 'smoke_validation', **evaluate(model, aux, data)})
            result = generate_metric(model, aux, args, args.output / 'smoke-generation', 64, 15701, log)
            log({'phase': 'smoke_fid', **result})
        while stream.epoch + stream.position / stream.size < args.epochs:
            indices, progress, epoch_end = stream.next(args.batch_size)
            step += 1
            lr = early_decay_lr(progress, args.epochs, args.lr, args.knee_lr, args.min_lr, args.warmup_epochs, args.knee_epoch)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            model.train()
            geometry_weight = scheduled_geometry_weight(.05, progress, 2., 3.)
            totals = {}
            iteration = time.monotonic()
            for micro_index, chunk in enumerate(indices.split(args.microbatch)):
                # Per-step seeds isolate stochastic contexts/dropout from evaluations and resumes.
                torch.manual_seed(args.seed + step * 100 + micro_index)
                a = raw['train']['atoms'][chunk].cuda().long()
                c = raw['train']['coefficients'][chunk].cuda()
                loss, metrics = objective(model, aux, a, c, geometry_weight)
                weight = len(chunk) / len(indices)
                (loss * weight).backward()
                for k, v in metrics.items():
                    totals[k] = totals.get(k, 0.) + v * weight
                del loss, a, c
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            optimizer.step()
            if step == 1 or step % 10 == 0 or epoch_end:
                seconds = time.monotonic() - iteration
                log({'phase': 'train', **{f'train/{k}': v for k, v in totals.items()},
                     'train/lr': lr, 'train/gradient_norm': float(norm), 'train/geometry_weight': geometry_weight,
                     'train/step_seconds': seconds, 'train/images_seen': stream.epoch * stream.size + stream.position,
                     'gpu/peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30})
            if stopping['signal'] or (args.stop_after_step and step >= args.stop_after_step):
                save_last()
                log({'phase': 'paused', 'signal': stopping['signal']})
                if wb:
                    wb.finish()
                return
            if epoch_end:
                epoch = stream.epoch + 1
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                save_last()
                if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
                    directory = validate(epoch)
                    if args.fid_samples:
                        result = generate_metric(model, aux, args, directory / 'screen', args.fid_samples, args.seed + 10000, log)
                        if best_fid is None or result['fid'] < best_fid:
                            best_fid, best_epoch = result['fid'], epoch
                            atomic_torch_save({'state_dict': model.state_dict(), 'config': config,
                                               'epoch': epoch, 'step': step, 'fid': best_fid}, args.output / 'best-screen.pt')
                        log({'phase': 'screen_complete', 'screen/fid': result['fid'], 'screen/samples': args.fid_samples})
                        if wb:
                            wb.log({'samples': wandb.Image(str(directory / 'screen/samples.png')), 'optimizer_step': step})
                    if args.full_fid_samples and (epoch % args.full_fid_every == 0 or epoch == args.epochs):
                        result = generate_metric(model, aux, args, directory / 'full', args.full_fid_samples, args.seed + 20000, log)
                        log({'phase': 'full_fid_complete', 'full/fid': result['fid'], 'full/samples': args.full_fid_samples})
                    save_last()
        atomic_torch_save({'state_dict': model.state_dict(), 'config': config, 'epoch': args.epochs, 'step': step}, args.output / 'final.pt')
        del optimizer
        if (args.output / 'best-screen.pt').exists():
            saved = torch.load(args.output / 'best-screen.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(saved['state_dict'], strict=True)
            del saved
        final = {'selected_epoch': best_epoch, 'selected_screen_fid': best_fid}
        if args.full_fid_samples:
            final['generation'] = generate_metric(model, aux, args, args.output / 'selected-independent-fid',
                                                  args.full_fid_samples, args.seed + 30000, log)
        final['holdout'] = evaluate(model, aux, raw['holdout'])
        write_json(args.output / 'results.json', final)
        log({'phase': 'complete', **final})
        if wb:
            wb.summary['results'] = final
            artifact = wandb.Artifact(args.wandb_id + '-selected', type='model', metadata={'epoch': best_epoch})
            artifact.add_file(str(args.output / 'best-screen.pt'), policy='immutable', skip_cache=True)
            artifact.add_file(str(args.output / 'results.json'), policy='immutable', skip_cache=True)
            wb.log_artifact(artifact).wait()
            wb.finish()
    except Exception as error:
        log({'phase': 'failed', 'error': repr(error)})
        if wb:
            wb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
