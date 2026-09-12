#!/usr/bin/env python3
"""Train a matched Church pattern-first or support-first five-event prior."""
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
    LaserAux, atomic_torch_save,
)
from scripts.train_church_ffhq_recipe import write_json
from src.church_calibrated_training import (
    AugmentedChurch, PendingEpochBatches,
    HeldoutStop, optimizer_groups,
)
from src.church_ffhq_recipe import early_decay_lr
from src.coefficient_history_training import EpochStream
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics
from src.church_pattern_order import pattern_order_prior, order_objective
from src.support_pattern_integer_codec import pack_support_pattern


@torch.no_grad()
def evaluate(model, aux, data, args):
    model.eval()
    totals = {}
    for first in range(0, len(data['atoms']), 16):
        atoms = data['atoms'][first:first + 16].cuda().long()
        physical = data['coefficients'][first:first + 16].cuda()
        _, metrics = order_objective(model, aux, atoms, physical)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + value * len(atoms)
    return {k: v / len(data['atoms']) for k, v in totals.items()}


@torch.no_grad()
def generate_metric(model, aux, args, output, count, seed, log):
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    metric = DistributedOriginalRQVAEMetrics('cuda', reference_stats_path=args.fid_stats)
    saved_atoms, saved_ids, complete_integers = [], [], []
    start = time.monotonic()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        for first in range(0, count, args.generation_batch):
            batch = min(args.generation_batch, count - first)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                atoms, ids = model.sample_compound(batch, aux,
                    atom_top_k=args.atom_top_k, atom_top_p=None, coeff_top_k=0,
                    coeff_top_p=args.coeff_top_p, atom_temperature=1.,
                    coeff_temperature=1., amp=True)
            saved_atoms.append(atoms.cpu().short())
            saved_ids.append(ids.cpu().short())
            complete_integers.extend(pack_support_pattern(atoms, ids, num_patterns=len(aux.coefficient_patterns)))
            for offset in range(0, batch, 32):
                images = ((aux.decode_coefficient_patterns(atoms[offset:offset + 32], ids[offset:offset + 32]) + 1) / 2).clamp(0, 1)
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
    atomic_torch_save({'atoms': torch.cat(saved_atoms), 'pattern_ids': torch.cat(saved_ids), 'complete_site_integers': complete_integers, 'integer_grid_shape': [count, 8, 8]}, output / 'generated-codes.pt')
    write_json(output / 'metrics.json', result)
    model.init_cache()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=Path, default=ROOT / 'outputs/church-support-pattern-integer-20260911/results.json')
    p.add_argument('--augmentation-check', type=Path, default=ROOT / 'outputs/church-support-pattern-integer-20260911/augmentation-check.json')
    p.add_argument('--cache', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911/continuous-cache.pt')
    p.add_argument('--data', type=Path, default=Path('/tmp/laser-sign-data/church/church_outdoor_train_lmdb'))
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    p.add_argument('--fid-stats', type=Path, default=ROOT / 'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--schedule-epochs', type=int, default=60)
    p.add_argument('--ordering', choices=['pattern-first', 'support-first'], required=True)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--microbatch', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=8e-5)
    p.add_argument('--knee-lr', type=float, default=1e-5)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--warmup-epochs', type=float, default=.5)
    p.add_argument('--knee-epoch', type=float, default=5.)
    p.add_argument('--weight-decay', type=float, default=.05)
    p.add_argument('--dropout', type=float, default=.15)
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--minimum-stop-epoch', type=int, default=4)
    p.add_argument('--fid-samples', type=int, default=4096)
    p.add_argument('--full-fid-samples', type=int, default=0)
    p.add_argument('--confirmation-samples', type=int, default=4096)
    p.add_argument('--full-fid-every', type=int, default=20)
    p.add_argument('--generation-batch', type=int, default=128)
    p.add_argument('--atom-top-k', type=int, default=2048)
    p.add_argument('--coeff-top-p', type=float, default=.5)
    p.add_argument('--seed', type=int, default=9901)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-step', type=int, default=0)
    p.add_argument('--smoke-evaluate', action='store_true')
    p.add_argument('--smoke-population', type=int, default=0,
                   help='Test-only limit on each split; never used by the production launcher')
    p.add_argument('--wandb-id')
    args = p.parse_args()
    early_decay_lr(0, args.schedule_epochs, args.lr, args.knee_lr, args.min_lr, args.warmup_epochs, args.knee_epoch)
    if not (1 <= args.microbatch <= 64 and args.batch_size > 0 and args.eval_every > 0 and args.patience > 0
            and 0 < args.epochs <= args.schedule_epochs and args.confirmation_samples >= 0
            and args.full_fid_every > 0 and args.fid_samples >= 0 and args.full_fid_samples >= 0
            and args.generation_batch > 0 and args.workers >= 0 and 0 < args.coeff_top_p <= 1):
        p.error('Invalid batch sizes or evaluation/stopping intervals')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.serialization.add_safe_globals([codecs.encode])
    torch.manual_seed(args.seed)
    calibration = json.loads(args.calibration.read_text())
    size = calibration['selected_vocabulary']
    if calibration['phase'] != 'complete' or size is None or not calibration['candidates'][str(size)]['passes_calibration']:
        raise ValueError('Joint coefficient vocabulary must pass calibration before training')
    codebook_path = Path(calibration['selected_codebook'])
    codebook_hash = sha256_file(codebook_path)
    if codebook_hash != calibration['candidates'][str(size)]['codebook_sha256']:
        raise ValueError('Selected coefficient codebook hash mismatch')
    book = torch.load(codebook_path, weights_only=True, map_location='cpu')
    augmentation = json.loads(args.augmentation_check.read_text())
    if not augmentation['passes'] or augmentation['codebook_sha256'] != codebook_hash:
        raise ValueError('Selected codebook must pass the fresh-view reconstruction check')
    raw = torch.load(args.cache, weights_only=False, map_location='cpu')
    assert sha256_file(args.cache) == calibration['cache_sha256']
    assert sha256_file(args.stage1) == raw['meta']['checkpoint_sha256'] == calibration['checkpoint_sha256'] == book['checkpoint_sha256']
    keys = [set(raw[s]['keys']) for s in ('train', 'holdout', 'validation')]
    assert not any(keys[i] & keys[j] for i in range(3) for j in range(i))
    if args.smoke_population:
        if args.smoke_population < 2 or args.wandb_id:
            p.error('Smoke populations require at least two images and cannot use a production W&B run')
        for split in ('train', 'holdout', 'validation'):
            raw[split] = {k: v[:args.smoke_population] for k, v in raw[split].items()}
    model = pattern_order_prior(size, args.dropout, args.ordering).cuda()
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=True, clamp_coeffs=False,
                   coefficient_patterns=book['coefficient_patterns']).cuda().eval().requires_grad_(False)
    torch.testing.assert_close(aux.coefficient_patterns, aux.coeff_bins[book['pattern_coefficient_ids'].cuda()] * aux.coeff_scales, atol=0, rtol=0)
    frozen = [(value, value._version) for value in (*aux.parameters(), *aux.buffers())]
    optimizer = torch.optim.AdamW(optimizer_groups(model, args.weight_decay), lr=args.lr,
                                 betas=(.9, .95), fused=True)
    stream = EpochStream(len(raw['train']['atoms']), args.seed + 1)
    monitor = HeldoutStop(args.patience, .01, args.minimum_stop_epoch)
    generation_monitor = HeldoutStop(args.patience, .25, args.minimum_stop_epoch)
    config = json.loads(json.dumps({**vars(args), 'architecture': 'balanced',
        'parameters': sum(p.numel() for p in model.parameters()),
        'initialization': 'random; no stage-2 checkpoint loaded', 'tokenizer': raw['meta'],
        'target_coordinate': 'physical; hard nearest joint pattern under exact support Gram metric',
        'coefficient_vocabulary': size, 'codebook_path': str(codebook_path), 'codebook_sha256': codebook_hash,
        'complete_site_integer_bits': book['nominal_bits_per_site'], 'augmentation_check': augmentation,
        'calibration': calibration, 'train_images': stream.size,
        'augmentation': 'fresh per-epoch Resize(256), RandomCrop(256), horizontal flip .5, then frozen BF16 encoder/FP32 OMP',
        'maximum_steps': args.epochs * math.ceil(stream.size / args.batch_size),
        'transport_fields': 5, 'serialized_complete_site_integer_unchanged': True,
        'stopping': f'{args.patience} consecutive checks without .01 held-out NLL improvement OR .25 FID-4096 improvement, after epoch {args.minimum_stop_epoch}',
        'objective': '(sum of four atom NLLs + joint pattern NLL)/5; selection uses unweighted joint NLL per site',
        'depth_context': 'pattern-first: known physical pattern and signed weighted support prefix; support-first: unweighted support prefix',
        'selection': 'best screening FID retained; independent-seed selected-checkpoint screen after bounded pilot; no automatic long-run promotion',
        'holdout_context': 'deterministic nearest joint patterns, same construction as generation',
        'source_hashes': {name: sha256_file(ROOT / name) for name in [
            'scripts/train_church_pattern_order.py', 'src/church_pattern_order.py', 'src/church_support_pattern_training.py',
            'src/church_calibrated_training.py', 'src/coefficient_pattern_codec.py', 'src/support_pattern_integer_codec.py',
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
        generation_monitor = HeldoutStop(**saved['generation_monitor'])
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
            name=args.wandb_id, group='church-pattern-order-20260911', job_type='matched-pattern-order-scratch',
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
        write_json(args.output / 'status.json', {'pid': os.getpid(), 'target_mode': args.ordering,
            'maximum_epochs': args.epochs, 'best_screen_fid': best_fid,
            'best_fid_epoch': best_fid_epoch, 'monitor': monitor.state_dict(), 'generation_monitor': generation_monitor.state_dict(), **full})
        if wb:
            wb.log(full)

    def save_last():
        assert all(value._version == version and value.grad is None for value, version in frozen)
        model.init_cache()
        atomic_torch_save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(), 'config': config, 'step': step, 'best_fid': best_fid,
            'best_fid_epoch': best_fid_epoch, 'monitor': monitor.state_dict(),
            'generation_monitor': generation_monitor.state_dict(),
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
            _, generation_stop = generation_monitor.observe(result['fid'], epoch)
            early_stop = early_stop or generation_stop
            log({'phase': 'screen_complete', 'screen/fid': result['fid'], 'screen/samples': args.fid_samples,
                 'generation_stop/bad_checks': generation_monitor.bad_checks, 'early_stop/triggered': early_stop})
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

    log({'phase': 'ready', 'parameters': config['parameters'], 'coefficient_patterns': size, 'resumed': args.resume})
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
                lr = early_decay_lr(progress, args.schedule_epochs, args.lr, args.knee_lr, args.min_lr,
                                    args.warmup_epochs, args.knee_epoch)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                model.train()
                totals = {}
                iteration = time.monotonic()
                for micro, batch in enumerate(images.split(args.microbatch)):
                    torch.manual_seed(args.seed + step * 100 + micro)
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        atoms, normalized = aux.encode_sparse_components(batch.cuda(non_blocking=True))
                    physical = normalized.float() * aux.coeff_scales
                    loss, metrics = order_objective(model, aux, atoms.long(), physical)
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
                        'train/lr': lr, 'train/gradient_norm': float(norm),
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
                elif step == 1 or step % 250 == 0:
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
        result = {'reason': 'heldout_or_generation_early_stop' if early_stop else 'maximum_epochs',
                  'selected_checkpoint': str(selected_path), 'selected_epoch': selected_epoch,
                  'best_screen_fid': best_fid, 'monitor': monitor.state_dict(),
                  'generation_monitor': generation_monitor.state_dict(), 'frozen_stage1_verified': True,
                  'holdout': evaluate(model, aux, raw['holdout'], args)}
        if args.confirmation_samples:
            result['generation'] = generate_metric(model, aux, args, args.output / 'selected-independent-fid',
                                                   args.confirmation_samples, 38701, log)
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
