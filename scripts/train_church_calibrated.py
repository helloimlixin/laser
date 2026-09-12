#!/usr/bin/env python3
"""Train calibrated Church compound priors with fresh pixel views and early stopping."""
import argparse
import codecs
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file
from scripts.train_official_rqtransformer_laser_stage2 import (
    LaserAux, atomic_torch_save, compound_objective, scheduled_geometry_weight,
)
from scripts.train_church_ffhq_recipe import write_json
from src.church_calibrated_training import (
    calibrated_prior, physical_targets, AugmentedChurch, PendingEpochBatches,
    HeldoutStop, optimizer_groups,
)
from src.church_ffhq_recipe import early_decay_lr
from src.coefficient_history_training import EpochStream
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


def objective(model, aux, atoms, physical, mode, sigma, geometry_weight, stochastic=True):
    packed, probabilities = physical_targets(aux, atoms, physical, mode, sigma, stochastic)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(packed, model_aux=aux, amp=True)
    loss, values = compound_objective(
        outputs['atom_logits'], outputs['coeff_logits'], None, atoms, probabilities,
        aux.dictionary.t()[atoms] * physical[..., None], atom_weight=1.5,
        geometry_weight=geometry_weight, accumulation=1, distribution_geometry=True,
        geometry_dictionary=aux.dictionary, geometry_coeff_bins=aux.coeff_bins,
        geometry_coeff_scales=aux.coeff_scales, geometry_top_k=4,
    )
    with torch.no_grad():
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        log_probs = outputs['coeff_logits'].float().log_softmax(-1)
        distribution = log_probs.exp()
        bins = aux.coeff_bins * aux.coeff_scales[..., None]
        nearest = (physical[..., None] - bins).abs().argmin(-1)
        coefficient_nll = -log_probs.gather(-1, nearest[..., None]).squeeze(-1)
        prediction = (distribution * bins).sum(-1)
        sign = distribution[..., aux.coeff_vocab_size // 2:].sum(-1) >= .5
        atom_nll = values['atom_nll'].mean()
        kl = (values['coeff_cross_entropy'] - entropy).mean()
        metrics = {'loss': loss.detach(), 'atom_nll': atom_nll,
            'coeff_cross_entropy': values['coeff_cross_entropy'].mean(),
            'coeff_target_entropy': entropy.mean(), 'coeff_kl': kl,
            'selection_score': (1.5 * atom_nll + kl) / 2.5,
            'coefficient_nearest_nll': coefficient_nll.mean(),
            'coefficient_mean_mae': (prediction - physical).abs().mean(),
            'sign_accuracy': (sign == (physical >= 0)).float().mean(),
            'geometry': values['geometry'].detach(),
            'coefficient_out_of_range': (physical.abs() > aux.coeff_scales * 3).float().mean()}
        for d in range(4):
            metrics[f'atom_nll_d{d}'] = values['atom_nll'][..., d].mean()
            metrics[f'coeff_kl_d{d}'] = (values['coeff_cross_entropy'] - entropy)[..., d].mean()
    return loss, {k: float(v) for k, v in metrics.items()}


@torch.no_grad()
def evaluate(model, aux, data, args, stochastic=True):
    model.eval()
    totals = {}
    # A fixed draw per image batch estimates the training-context objective.
    # Clean-context diagnostics are logged separately for direct comparisons.
    with torch.random.fork_rng(devices=[0]):
        for first in range(0, len(data['atoms']), 16):
            torch.manual_seed(29701 + first)
            atoms = data['atoms'][first:first + 16].cuda().long()
            physical = data['coefficients'][first:first + 16].cuda()
            _, metrics = objective(model, aux, atoms, physical, args.target_mode,
                                   args.sigma, .05, stochastic=stochastic)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + value * len(atoms)
    return {k: v / len(data['atoms']) for k, v in totals.items()}


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
                atoms, ids = model.sample_compound(batch, aux,
                    atom_top_k=args.atom_top_k, atom_top_p=1., coeff_top_k=0,
                    coeff_top_p=args.coeff_top_p, atom_temperature=1.,
                    coeff_temperature=1., amp=True)
            saved_atoms.append(atoms.cpu().short())
            saved_ids.append(ids.cpu().short())
            for offset in range(0, batch, 32):
                images = ((aux.decode_compound(atoms[offset:offset + 32], ids[offset:offset + 32]) + 1) / 2).clamp(0, 1)
                metric.update(images, real=False)
                if first == 0 and offset == 0:
                    save_image(images, output / 'samples.png', nrow=8)
            if first == 0 or (first // args.generation_batch) % 8 == 0 or first + batch == count:
                log({'phase': 'generation', 'generated': first + batch, 'target_samples': count,
                     'generation_seconds': time.monotonic() - start})
    fid, _, _ = metric.compute()
    result = {'fid': float(fid), 'samples': count, 'seed': seed, 'atom_top_k': args.atom_top_k,
              'coeff_top_p': args.coeff_top_p, 'temperature': 1.,
              'precision': 'BF16 AR; FP32 decoder; original RQ-VAE Inception',
              'seconds': time.monotonic() - start}
    atomic_torch_save({'atoms': torch.cat(saved_atoms), 'coefficient_ids': torch.cat(saved_ids)}, output / 'generated-codes.pt')
    write_json(output / 'metrics.json', result)
    model.init_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-mode', choices=['soft', 'hard'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=Path, default=ROOT / 'outputs/church-calibrated-20260911/calibration/calibration.json')
    p.add_argument('--cache', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911/continuous-cache.pt')
    p.add_argument('--data', type=Path, default=Path('/tmp/laser-sign-data/church/church_outdoor_train_lmdb'))
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    p.add_argument('--fid-stats', type=Path, default=ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--microbatch', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--knee-lr', type=float, default=2e-5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--warmup-epochs', type=float, default=1.)
    p.add_argument('--knee-epoch', type=float, default=10.)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--dropout', type=float, default=.15)
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--minimum-stop-epoch', type=int, default=8)
    p.add_argument('--fid-samples', type=int, default=4096)
    p.add_argument('--full-fid-samples', type=int, default=50000)
    p.add_argument('--full-fid-every', type=int, default=25)
    p.add_argument('--generation-batch', type=int, default=128)
    p.add_argument('--atom-top-k', type=int, default=2048)
    p.add_argument('--coeff-top-p', type=float, default=.5)
    p.add_argument('--seed', type=int, default=8701)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-step', type=int, default=0)
    p.add_argument('--smoke-evaluate', action='store_true')
    p.add_argument('--smoke-population', type=int, default=0,
                   help='Test-only limit on each split; never used by the production launcher')
    p.add_argument('--wandb-id')
    args = p.parse_args()
    early_decay_lr(0, args.epochs, args.lr, args.knee_lr, args.min_lr, args.warmup_epochs, args.knee_epoch)
    if not (1 <= args.microbatch <= 64 and args.batch_size > 0 and args.eval_every > 0 and args.patience > 0):
        p.error('Invalid batch sizes or evaluation/stopping intervals')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    calibration = json.loads(args.calibration.read_text())
    args.sigma = float(calibration['selected_sigma'])
    raw = torch.load(args.cache, weights_only=False, map_location='cpu')
    assert sha256_file(args.cache) == calibration['cache_sha256']
    assert sha256_file(args.stage1) == raw['meta']['checkpoint_sha256'] == calibration['checkpoint_sha256']
    keys = [set(raw[s]['keys']) for s in ('train', 'holdout', 'validation')]
    assert not any(keys[i] & keys[j] for i in range(3) for j in range(i))
    if args.smoke_population:
        if args.smoke_population < 2 or args.wandb_id:
            p.error('Smoke populations require at least two images and cannot use a production W&B run')
        for split in ('train', 'holdout', 'validation'):
            raw[split] = {k: v[:args.smoke_population] for k, v in raw[split].items()}
    model = calibrated_prior(args.dropout).cuda()
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=True, clamp_coeffs=False).cuda().eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.weight_decay), lr=args.lr,
                                 betas=(.9, .95), fused=True)
    stream = EpochStream(len(raw['train']['atoms']), args.seed + 1)
    monitor = HeldoutStop(args.patience, .01, args.minimum_stop_epoch)
    config = json.loads(json.dumps({**vars(args), 'architecture': 'balanced',
        'parameters': sum(p.numel() for p in model.parameters()),
        'initialization': 'random; no stage-2 checkpoint loaded', 'tokenizer': raw['meta'],
        'target_coordinate': 'physical', 'target_temperature': 2 * args.sigma ** 2,
        'calibration': calibration, 'train_images': stream.size,
        'augmentation': 'fresh per-epoch Resize(256), RandomCrop(256), horizontal flip .5, then frozen BF16 encoder/FP32 OMP',
        'maximum_steps': args.epochs * math.ceil(stream.size / args.batch_size),
        'stopping': f'{args.patience} consecutive held-out checks without .01 improvement in (1.5 atom NLL + coefficient KL)/2.5, after epoch {args.minimum_stop_epoch}',
        'selection': 'best FID-4096 retained and confirmed by independent-seed FID-50000; best holdout checkpoint retained separately',
        'holdout_context': 'fixed-seed stochastic context matching target recipe; clean contexts logged separately',
        'source_hashes': {name: sha256_file(ROOT / name) for name in [
            'scripts/train_church_calibrated.py', 'src/church_calibrated_training.py',
            'src/church_ffhq_recipe.py', 'src/coefficient_history_training.py',
            'scripts/train_church_ffhq_recipe.py', 'scripts/train_official_rqtransformer_laser_stage2.py',
            'src/models/rqtransformer/transformers.py', 'src/models/rqtransformer/attentions.py']},
    }, default=str))
    step, elapsed, best_fid, best_fid_epoch, pending_epoch, early_stop = 0, 0., None, None, None, False
    if args.resume:
        saved = torch.load(args.output / 'last.pt', weights_only=False, map_location='cpu')
        ignored = {'resume', 'stop_after_step', 'smoke_evaluate', 'wandb_id', 'workers'}
        for key in config.keys() - ignored:
            if config[key] != saved['config'][key]:
                raise ValueError(f'Resume setting changed: {key}')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        stream.load_state_dict(saved['stream'])
        monitor = HeldoutStop(**saved['monitor'])
        step, elapsed, best_fid, best_fid_epoch, pending_epoch, early_stop = [saved[k] for k in
            ('step', 'elapsed_seconds', 'best_fid', 'best_fid_epoch', 'pending_epoch', 'early_stop')]
        del saved
    write_json(args.output / 'config.json', config)
    dataset = AugmentedChurch(args.data, raw['train']['keys'], args.seed + 2)
    loader = DataLoader(dataset, batch_sampler=PendingEpochBatches(stream, args.batch_size),
                        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(args.seed + 3))
    wb = None
    if args.wandb_id:
        import wandb
        wb = wandb.init(entity='helloimlixin-rutgers', project='laser', id=args.wandb_id,
            name=args.wandb_id, group='church-calibrated-20260911', job_type='calibrated-stage2-scratch',
            dir=str(args.output), config=config, resume='must' if args.resume else 'never')
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
        write_json(args.output / 'status.json', {'pid': os.getpid(), 'target_mode': args.target_mode,
            'maximum_epochs': args.epochs, 'best_screen_fid': best_fid,
            'best_fid_epoch': best_fid_epoch, 'monitor': monitor.state_dict(), **full})
        if wb:
            wb.log(full)

    def save_last():
        model.init_cache()
        atomic_torch_save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(), 'config': config, 'step': step, 'best_fid': best_fid,
            'best_fid_epoch': best_fid_epoch, 'monitor': monitor.state_dict(),
            'pending_epoch': pending_epoch, 'early_stop': early_stop,
            'elapsed_seconds': elapsed + time.monotonic() - start}, args.output / 'last.pt')
        log({'phase': 'checkpoint_saved'})

    def save_model(name, epoch):
        model.init_cache()
        atomic_torch_save({'state_dict': model.state_dict(), 'config': config, 'step': step,
                           'epoch': epoch, 'screen_fid': best_fid}, args.output / name)

    def validate_epoch(epoch):
        nonlocal best_fid, best_fid_epoch, early_stop, pending_epoch
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        values = {s: evaluate(model, aux, raw[s], args) for s in ('holdout', 'validation')}
        values['holdout_clean'] = evaluate(model, aux, raw['holdout'], args, stochastic=False)
        probe = {k: v[:256] for k, v in raw['train'].items() if k != 'keys'}
        values['train_probe'] = evaluate(model, aux, probe, args)
        improved, early_stop = monitor.observe(values['holdout']['selection_score'], epoch)
        directory = args.output / f'evaluations/epoch-{epoch:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'teacher-forcing.json', values)
        log({'phase': 'validation', **{f'{s}/{k}': v for s, metrics in values.items() for k, v in metrics.items()},
             'early_stop/bad_checks': monitor.bad_checks, 'early_stop/triggered': early_stop})
        if improved:
            save_model('best-holdout.pt', epoch)
        if args.fid_samples:
            result = generate_metric(model, aux, args, directory / 'screen', args.fid_samples, 18701, log)
            if best_fid is None or result['fid'] < best_fid:
                best_fid, best_fid_epoch = result['fid'], epoch
                save_model('best-screen.pt', epoch)
            log({'phase': 'screen_complete', 'screen/fid': result['fid'], 'screen/samples': args.fid_samples})
            if wb:
                wb.log({'samples': wandb.Image(str(directory / 'screen/samples.png')), 'optimizer_step': step})
        if args.full_fid_samples and epoch % args.full_fid_every == 0 and not early_stop:
            result = generate_metric(model, aux, args, directory / 'full', args.full_fid_samples, 28701, log)
            log({'phase': 'full_fid_complete', 'full/fid': result['fid'], 'full/samples': args.full_fid_samples})
        pending_epoch = None
        save_last()

    def pause_if_requested():
        if stopping['signal'] or (args.stop_after_step and step >= args.stop_after_step):
            save_last()
            log({'phase': 'paused', 'signal': stopping['signal']})
            if wb:
                wb.summary['training_status'] = 'paused'
                wb.finish()
            return True
        return False

    log({'phase': 'ready', 'parameters': config['parameters'], 'sigma_physical': args.sigma, 'resumed': args.resume})
    try:
        if args.smoke_evaluate:
            data = {k: v[:16] for k, v in raw['holdout'].items() if k != 'keys'}
            log({'phase': 'smoke_validation', **evaluate(model, aux, data, args)})
            log({'phase': 'smoke_fid', **generate_metric(model, aux, args, args.output / 'smoke', 64, 18701, log)})
        if pending_epoch is not None:
            validate_epoch(pending_epoch)
        while stream.epoch + stream.position / stream.size < args.epochs and not early_stop:
            for images, actual_indices in loader:
                indices, progress, epoch_end = stream.next(args.batch_size)
                if not torch.equal(indices, actual_indices):
                    raise RuntimeError('Prefetched images differ from the committed epoch stream')
                step += 1
                lr = early_decay_lr(progress, args.epochs, args.lr, args.knee_lr, args.min_lr,
                                    args.warmup_epochs, args.knee_epoch)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                model.train()
                weight_geo = scheduled_geometry_weight(.05, progress, 2., 3.)
                totals = {}
                iteration = time.monotonic()
                for micro, batch in enumerate(images.split(args.microbatch)):
                    torch.manual_seed(args.seed + step * 100 + micro)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        atoms, normalized = aux.encode_sparse_components(batch.cuda(non_blocking=True))
                    physical = normalized.float() * aux.coeff_scales
                    loss, metrics = objective(model, aux, atoms.long(), physical, args.target_mode,
                                               args.sigma, weight_geo)
                    if metrics['coefficient_out_of_range'] > .01:
                        raise RuntimeError('More than 1% of augmented coefficients exceed the retained bin range')
                    weight = len(batch) / len(images)
                    (loss * weight).backward()
                    for key, value in metrics.items():
                        totals[key] = totals.get(key, 0.) + value * weight
                    del loss, atoms, normalized, physical
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f'Nonfinite gradient at step {step}')
                optimizer.step()
                if step == 1 or step % 10 == 0 or epoch_end:
                    log({'phase': 'train', **{f'train/{k}': v for k, v in totals.items()},
                        'train/lr': lr, 'train/gradient_norm': float(norm), 'train/geometry_weight': weight_geo,
                        'train/step_seconds': time.monotonic() - iteration,
                        'train/images_seen': stream.epoch * stream.size + stream.position,
                        'gpu/peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30})
                if epoch_end:
                    epoch = stream.epoch + 1
                    if epoch == 1 or epoch % args.eval_every == 0 or epoch % args.full_fid_every == 0 or epoch == args.epochs:
                        pending_epoch = epoch
                if pause_if_requested():
                    return
                if epoch_end:
                    save_last()
                    if pending_epoch is not None:
                        validate_epoch(pending_epoch)
                    if pause_if_requested() or early_stop:
                        break
                elif step % 250 == 0:
                    save_last()
            if stopping['signal']:
                return
        save_last()
        save_model('final.pt', stream.epoch + stream.position / stream.size)
        del optimizer, loader
        gc.collect()
        torch.cuda.empty_cache()
        selected_path = args.output / ('best-screen.pt' if (args.output / 'best-screen.pt').exists() else 'best-holdout.pt')
        saved = torch.load(selected_path, weights_only=False, map_location='cpu')
        model.load_state_dict(saved['state_dict'], strict=True)
        selected_epoch = saved['epoch']
        del saved
        result = {'reason': 'heldout_early_stop' if early_stop else 'maximum_epochs',
                  'selected_checkpoint': str(selected_path), 'selected_epoch': selected_epoch,
                  'best_screen_fid': best_fid, 'monitor': monitor.state_dict(),
                  'holdout': evaluate(model, aux, raw['holdout'], args)}
        if args.full_fid_samples:
            result['generation'] = generate_metric(model, aux, args, args.output / 'selected-independent-fid',
                                                   args.full_fid_samples, 38701, log)
        write_json(args.output / 'results.json', result)
        log({'phase': 'complete', **result})
        if wb:
            wb.summary['training_status'] = result['reason']
            wb.summary['results'] = result
            artifact = wandb.Artifact(args.wandb_id + '-selected', type='model', metadata={'epoch': selected_epoch})
            artifact.add_file(str(selected_path), policy='immutable', skip_cache=True)
            artifact.add_file(str(args.output / 'results.json'), policy='immutable', skip_cache=True)
            wb.log_artifact(artifact).wait()
            wb.finish()
    except Exception as error:
        log({'phase': 'failed', 'error': repr(error)})
        if wb:
            wb.summary['training_status'] = 'failed'
            wb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
