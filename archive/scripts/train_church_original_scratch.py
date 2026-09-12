#!/usr/bin/env python3
"""Train the original Church compound architecture from random stage-2 initialization."""
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
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file
from src.training.rqtransformer import LaserAux, atomic_torch_save
from archive.scripts.train_church_ffhq_recipe import write_json
from src.church_epoch50_loop import cached_splits, pair_objective
from src.church_original_scratch import scratch_prior, initialization_audit, FullBatchEpochStream
from src.church_ffhq_recipe import early_decay_lr
from src.coefficient_history_training import EpochStream
from src.rqvae_metrics import DistributedOriginalRQVAEMetrics


@torch.no_grad()
def evaluate(model, aux, data):
    model.eval()
    totals = {}
    for chunk in data.split(16):
        _, metrics = pair_objective(model, aux, chunk.cuda())
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + len(chunk) * value
    return {key: value / len(data) for key, value in totals.items()}


@torch.no_grad()
def generate(model, aux, args, directory, count, seed, log):
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    metric = DistributedOriginalRQVAEMetrics('cuda', reference_stats_path=args.fid_stats)
    atoms_all, ids_all = [], []
    started = time.monotonic()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        for first in range(0, count, args.generation_batch):
            batch = min(args.generation_batch, count - first)
            # Preserve the source evaluator's nested precision scopes: cached
            # backbone disables autocast; sampled readouts use outer BF16.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                atoms, ids = model.sample_compound(batch, aux, atom_top_k=250,
                    atom_top_p=1., coeff_top_k=0, coeff_top_p=None,
                    atom_temperature=1., coeff_temperature=1., amp=False)
            atoms_all.append(atoms.cpu().short())
            ids_all.append(ids.cpu().short())
            for offset in range(0, batch, 32):
                images = ((aux.decode_compound(atoms[offset:offset+32], ids[offset:offset+32]) + 1) / 2).clamp(0, 1)
                metric.update(images, real=False)
                if first == 0 and offset == 0:
                    save_image(images, directory / 'samples.png', nrow=8)
            if first == 0 or first // args.generation_batch % 8 == 0 or first + batch == count:
                log({'phase': 'generation', 'generated': first + batch, 'target_samples': count,
                     'generation_seconds': time.monotonic() - started})
    fid, _, _ = metric.compute()
    result = {'fid': float(fid), 'samples': count, 'seed': seed,
              'atom_top_k': 250, 'coefficient_sampling': 'untruncated', 'temperature': 1.,
              'precision': 'source sampler: FP32/TF32 cached backbone, BF16 readouts; FP32 decoder; original RQ-VAE Inception',
              'seconds': time.monotonic() - started}
    atomic_torch_save({'atoms': torch.cat(atoms_all), 'coefficient_ids': torch.cat(ids_all)}, directory / 'generated-codes.pt')
    write_json(directory / 'metrics.json', result)
    model.init_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['looped', 'control'], required=True)
    parser.add_argument('--stage1', type=Path, default=ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    parser.add_argument('--cache', type=Path, default=ROOT/'outputs/lsun-church-bar-20260910/church-cache.pt')
    parser.add_argument('--fid-stats', type=Path, default=ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--minimum-stop-epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--microbatch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--gate-lr-multiplier', type=float, default=1.)
    parser.add_argument('--warmup-epochs', type=float, default=1.)
    parser.add_argument('--knee-epoch', type=float, default=10.)
    parser.add_argument('--knee-lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--fid-samples', type=int, default=4096)
    parser.add_argument('--confirmation-samples', type=int, default=50000)
    parser.add_argument('--generation-batch', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-step', type=int, default=0)
    parser.add_argument('--smoke-population', type=int, default=0)
    parser.add_argument('--wandb-id')
    args = parser.parse_args()
    early_decay_lr(0, args.epochs, args.lr, args.knee_lr, args.min_lr, args.warmup_epochs, args.knee_epoch)
    if min(args.batch_size, args.microbatch, args.eval_every, args.patience, args.generation_batch) < 1:
        parser.error('Expected positive batch sizes and intervals')
    if args.fid_samples < 0 or args.confirmation_samples < 0 or args.gate_lr_multiplier <= 0:
        parser.error('Invalid evaluation count or gate rate')
    if args.smoke_population and (args.smoke_population < 2 or args.wandb_id):
        parser.error('Smoke populations cannot be production W&B runs')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    # Retain the source implementation's CUDA cumsum and TF32 setting. PyTorch
    # does not certify cumsum as deterministic; resume equivalence is tested.
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.serialization.add_safe_globals([codecs.encode])
    torch.manual_seed(args.seed)
    raw = torch.load(args.cache, map_location='cpu', weights_only=False)
    splits = cached_splits(raw)
    model = scratch_prior(args.variant)
    initialization = initialization_audit(model, args.seed)
    write_json(args.output/'initialization.json', initialization)
    model = model.cuda()
    scales = [1.109375, 0.6234375238418579, 0.3685546815395355, 0.24746093153953552]
    assert scales == raw['meta']['coeff_scales'] and raw['meta']['coeff_max'] == 20
    stage1_hash = sha256_file(args.stage1)
    assert stage1_hash == raw['meta']['checkpoint_sha256']
    aux = LaserAux(args.stage1, 16384, 2048, 20., coeff_scales=scales,
                   sparsity_level=4).cuda().eval().requires_grad_(False)
    frozen = [(value, value._version) for value in (*aux.parameters(), *aux.buffers())]
    if args.smoke_population:
        splits = {k: v[:args.smoke_population] for k, v in splits.items()}
    del raw
    groups = [{'params': [p for n, p in model.named_parameters() if not n.endswith('loop_gates')],
               'weight_decay': 1e-4, 'lr_multiplier': 1.}]
    if args.variant == 'looped':
        groups.append({'params': [model.head_transformer.loop_gates], 'weight_decay': 0.,
                       'lr_multiplier': args.gate_lr_multiplier})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(.9, .95), fused=True)
    assert len(optimizer.state) == 0
    stream = FullBatchEpochStream(len(splits['train']), args.seed + 1)
    if stream.size < args.batch_size:
        parser.error('Training population is smaller than the full effective batch')
    config = json.loads(json.dumps({**vars(args), 'parameters': sum(p.numel() for p in model.parameters()),
        'initialization': initialization, 'stage2_checkpoint_loaded': None,
        'initial_optimizer_state_entries': 0, 'steps_per_epoch': stream.size // args.batch_size,
        'maximum_steps': args.epochs * (stream.size // args.batch_size),
        'stage1_sha256': stage1_hash, 'cache_sha256': sha256_file(args.cache), 'train_images': stream.size,
        'architecture': 'original 1024-wide 24-spatial/4-depth compound full-pair RQ; two coefficient micro blocks',
        'loop_update': 'h <- h + tanh(gate)*(F(h)-h); two extra shared four-block passes; gates initialized zero' if args.variant == 'looped' else 'original four-block depth head',
        'objective': '(mean atom NLL + mean coefficient NLL)/2; original hard targets; no geometry loss',
        'data': 'cached center crops, original bins/scales; train and former holdout rejoined; no added augmentation',
        'validation_note': '300 official images reused in earlier experiments; train_probe belongs to this run training population',
        'optimizer': 'fresh empty AdamW state; original weight decay 1e-4 and betas (.9,.95)',
        'selection': 'best checkpoint from this scratch run only; six unimproved screens trigger stopping after epoch 50; selected checkpoint gets FID-50k',
        'baseline_reference': {'unguided_fid50k': 13.704673116414568, 'sample_seed': 17701},
        'source_hashes': {name: sha256_file(ROOT/name) for name in [
            'scripts/train_church_original_scratch.py', 'src/church_original_scratch.py', 'src/church_epoch50_loop.py', 'src/church_ffhq_recipe.py',
            'scripts/train_church_bar_coefficients.py', 'scripts/train_church_ffhq_recipe.py',
            'scripts/train_official_rqtransformer_laser_stage2.py', 'src/coefficient_history_training.py',
            'src/models/rqtransformer/attentions.py', 'src/models/rqtransformer/transformers.py']},
    }, default=str))
    step, elapsed, best_fid, best_step, bad_checks, pending_eval, initialized = 0, 0., None, 0, 0, None, False
    if args.resume:
        saved = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
        for key in config.keys() - {'resume', 'stop_after_step', 'wandb_id'}:
            if config[key] != saved['config'][key]:
                raise ValueError(f'Resume setting changed: {key}')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        stream.load_state_dict(saved['stream'])
        step, elapsed, best_fid, best_step, bad_checks, pending_eval, initialized = [saved[k] for k in
            ('step', 'elapsed_seconds', 'best_fid', 'best_step', 'bad_checks', 'pending_eval', 'initialized')]
        del saved
    write_json(args.output/'config.json', config)
    wb = None
    if args.wandb_id:
        import wandb
        wb = wandb.init(entity='helloimlixin-rutgers', project='laser', id=args.wandb_id,
            name=f'Church original 405M {args.variant} FROM SCRATCH', group='church-original-scratch-20260911',
            job_type='stage2-from-scratch', dir=str(args.output), config=config, resume='must' if args.resume else 'never')
    started = time.monotonic()
    stopping = {'signal': None}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopping.update(signal=number))

    def log(row):
        full = {'optimizer_step': step, 'epoch_progress': stream.epoch + stream.position/stream.size,
                'elapsed_seconds': elapsed + time.monotonic() - started, **row}
        print(json.dumps(full, allow_nan=False), flush=True)
        with (args.output/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(full, allow_nan=False)+'\n')
        write_json(args.output/'status.json', {'pid': os.getpid(), 'variant': args.variant,
                   'best_fid': best_fid, 'best_step': best_step, 'bad_checks': bad_checks, **full})
        if wb:
            wb.log(full)

    def save_model(name):
        model.init_cache()
        atomic_torch_save({'state_dict': model.state_dict(), 'config': config, 'step': step,
                           'screen_fid': best_fid}, args.output/name)

    def save_last():
        assert all(v._version == version and v.grad is None for v, version in frozen)
        model.init_cache()
        atomic_torch_save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(), 'config': config, 'step': step, 'best_fid': best_fid,
            'best_step': best_step, 'bad_checks': bad_checks, 'pending_eval': pending_eval,
            'initialized': initialized, 'elapsed_seconds': elapsed + time.monotonic() - started}, args.output/'last.pt')
        log({'phase': 'checkpoint_saved'})

    def validate():
        nonlocal best_fid, best_step, bad_checks, pending_eval, initialized
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        directory = args.output/f'evaluations/step-{step:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        values = {s: evaluate(model, aux, splits[s]) for s in ('validation', 'train_probe')}
        write_json(directory/'teacher-forcing.json', values)
        log({'phase': 'validation', **{f'{s}/{k}': v for s, metrics in values.items() for k, v in metrics.items()}})
        if args.fid_samples:
            result = generate(model, aux, args, directory/'screen', args.fid_samples, 12701, log)
            improved = best_fid is None or result['fid'] < best_fid - .25
            if improved:
                best_fid, best_step, bad_checks = result['fid'], step, 0
                save_model('best-screen.pt')
            else:
                bad_checks += 1
            log({'phase': 'screen_complete', 'screen/fid': result['fid'], 'screen/samples': args.fid_samples,
                 'selection/improved': improved, 'selection/bad_checks': bad_checks})
            if wb:
                wb.log({'samples': wandb.Image(str(directory/'screen/samples.png')), 'optimizer_step': step})
        elif not (args.output/'best-screen.pt').exists():
            save_model('best-screen.pt')
        initialized, pending_eval = True, None
        save_last()

    def pause():
        if stopping['signal'] or (args.stop_after_step and step >= args.stop_after_step):
            save_last()
            log({'phase': 'paused', 'signal': stopping['signal']})
            if wb:
                wb.summary['training_status'] = 'paused'
                wb.finish()
            return True
        return False

    log({'phase': 'ready', 'parameters': config['parameters'], 'initialization_kind': 'random', 'resumed': args.resume})
    try:
        if pending_eval is not None:
            validate()
        elif not initialized:
            initialized = True
            save_last()
        if pause():
            return
        while stream.epoch + stream.position/stream.size < args.epochs and not (bad_checks >= args.patience and stream.epoch + stream.position/stream.size >= args.minimum_stop_epoch):
            indices, progress, epoch_end = stream.next(args.batch_size)
            step += 1
            lr = early_decay_lr(progress, args.epochs, args.lr, args.knee_lr, args.min_lr, args.warmup_epochs, args.knee_epoch)
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_multiplier']
            optimizer.zero_grad(set_to_none=True)
            model.train()
            totals, iteration = {}, time.monotonic()
            for micro, chunk in enumerate(splits['train'][indices].split(args.microbatch)):
                torch.manual_seed(args.seed + step * 10000 + micro)
                loss, metrics = pair_objective(model, aux, chunk.cuda())
                (loss * (len(chunk)/len(indices))).backward()
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value * len(chunk)/len(indices)
                del loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            gates = model.head_transformer.loop_gates.detach().tanh().tolist() if args.variant == 'looped' else []
            if step == 1 or step % 4 == 0:
                log({'phase': 'train', **{f'train/{k}': v for k, v in totals.items()},
                     'train/lr': lr, 'train/gradient_norm': float(norm), 'train/loop_gates': gates,
                     'train/step_seconds': time.monotonic()-iteration,
                     'gpu/peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30})
            if epoch_end:
                epoch = stream.epoch + 1
                if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
                    pending_eval = step
            if pause():
                return
            if pending_eval is not None:
                save_last()
                validate()
                if pause():
                    return
            elif step == 1 or step % 61 == 0 or epoch_end:
                save_last()
        save_last()
        save_model('final.pt')
        selected = torch.load(args.output/'best-screen.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(selected['state_dict'], strict=True)
        selected_step = selected['step']
        del selected, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        result = {'reason': 'screen_plateau' if bad_checks >= args.patience and stream.epoch + stream.position/stream.size >= args.minimum_stop_epoch else 'maximum_epochs',
                  'selected_step': selected_step, 'best_screen_fid': best_fid,
                  'trained_from_scratch': True, 'initialization': initialization,
                  'frozen_stage1_verified': True}
        if args.confirmation_samples:
            result['generation'] = generate(model, aux, args, args.output/'selected-fid', args.confirmation_samples, 17701, log)
        write_json(args.output/'results.json', result)
        log({'phase': 'complete', **result})
        if wb:
            wb.summary['results'] = result
            wb.summary['training_status'] = 'complete'
            wb.finish()
    except Exception as error:
        log({'phase': 'failed', 'error': repr(error)})
        if wb:
            wb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
