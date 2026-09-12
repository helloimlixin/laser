#!/usr/bin/env python3
"""Train a complete-sparse-code tokenizer through the frozen Church image decoder."""
import argparse
import codecs
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, atomic_torch_save
from scripts.tools.build_sign_probe_cache import sha256_file
from src.coefficient_history_training import EpochStream
from src.learned_sparse_site_codec import LearnedSparseSiteCodec, SparseSiteProjector
from src.models.lpips import LPIPS


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911/continuous-cache.pt')
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    p.add_argument('--initial-codebook', type=Path, default=ROOT / 'outputs/church-complete-site-token-20260911/vocab-16384/codebook.pt')
    p.add_argument('--split-indices', type=Path, default=ROOT / 'outputs/church-complete-site-token-20260911/fit-indices.pt')
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--microbatch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--codeword-lr', type=float, default=5e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=250)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--minimum-steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=10701)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after-step', type=int, default=0)
    p.add_argument('--smoke-eval-images', type=int, default=0)
    p.add_argument('--wandb-id')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.microbatch, args.eval_every, args.patience) < 1:
        raise ValueError('Expected positive training and evaluation sizes')
    if args.microbatch > 8 or not 0 < args.lr <= .001 or not 0 < args.codeword_lr <= .01:
        raise ValueError('Unsupported microbatch or learning rate')
    if args.smoke_eval_images and args.wandb_id:
        raise ValueError('Production W&B runs must use the complete evaluation population')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.manual_seed(args.seed)
    torch.serialization.add_safe_globals([codecs.encode])
    raw = torch.load(args.cache, weights_only=False, map_location='cpu')
    initial = torch.load(args.initial_codebook, weights_only=True, map_location='cpu')
    split_indices = torch.load(args.split_indices, weights_only=True, map_location='cpu')
    stage1_hash = sha256_file(args.stage1)
    assert raw['meta']['checkpoint_sha256'] == initial['checkpoint_sha256'] == stage1_hash
    key_sets = {name: set(raw[name]['keys']) for name in ('train', 'holdout', 'validation')}
    assert not key_sets['train'] & key_sets['holdout'] and not key_sets['train'] & key_sets['validation']
    assert not key_sets['holdout'] & key_sets['validation']
    calibration = split_indices['calibration_image_indices']
    train_indices = torch.arange(len(raw['train']['atoms']))
    train_indices = train_indices[~torch.isin(train_indices, calibration)]
    stream = EpochStream(len(train_indices), args.seed + 1)
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
        sparsity_level=4, soft_target_physical=True, clamp_coeffs=False).cuda().eval().requires_grad_(False)
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    frozen_versions = {n: value._version for n, value in list(aux.named_parameters()) + list(aux.named_buffers())}
    projector = SparseSiteProjector(aux.dictionary.t(), aux.coeff_bins, aux.coeff_scales).cuda()
    model = LearnedSparseSiteCodec(initial['latents']).cuda()

    @torch.no_grad()
    def target_latent(split, indices):
        atoms = raw[split]['atoms'][indices].cuda().long()
        coefficients = raw[split]['coefficients'][indices].cuda()
        return (aux.dictionary.t()[atoms] * coefficients[..., None]).sum(-2).permute(0, 3, 1, 2).contiguous()

    model.set_normalization(target_latent('train', train_indices[:1024]))
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.lr, 'weight_decay': .01},
        {'params': [model.codewords], 'lr': args.codeword_lr, 'weight_decay': 0.},
    ], betas=(.9, .95), fused=True)
    assert not {id(p) for p in model.parameters()} & {id(p) for p in aux.parameters()}
    source_files = ['scripts/train_church_learned_site_codec.py', 'src/learned_sparse_site_codec.py',
                    'src/coefficient_history_training.py', 'src/coefficient_pattern_codec.py',
                    'scripts/train_official_rqtransformer_laser_stage2.py']
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'architecture': model.config, 'parameters': sum(p.numel() for p in model.parameters()),
        'stage1_sha256': stage1_hash, 'cache_sha256': sha256_file(args.cache),
        'initial_codebook_sha256': sha256_file(args.initial_codebook),
        'source_sha256': {name: sha256_file(ROOT / name) for name in source_files},
        'training_images': len(train_indices), 'calibration_images': len(calibration),
        'representation': 'one categorical ID selects four atom IDs and four signed coefficient-bin IDs',
        'training_views': 'existing center-crop continuous sparse cache; no image augmentation in this pilot',
        'selection': 'minimum calibration LPIPS; stop after four checks without .001 improvement after step 2000',
        'projection_gradient': 'identity surrogate through hard OMP/bin projection; image gradients to encoder and selected codewords',
        'objective': 'LPIPS + .5 image_L1 + .1 relative_latent_MSE + .25 commitment + .1 alignment + .1 projection'})
    step, pending, best, best_step, stop_best, bad_checks, elapsed = 0, 0, None, None, None, 0, 0.
    if args.resume:
        saved = torch.load(args.output / 'last.pt', weights_only=False, map_location='cpu')
        ignore = {'resume', 'stop_after_step', 'wandb_id'}
        if {k: v for k, v in config.items() if k not in ignore} != {k: v for k, v in saved['config'].items() if k not in ignore}:
            raise ValueError('Source, data, or training configuration changed on resume')
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        stream.load_state_dict(saved['stream'])
        step, pending, best, best_step, stop_best, bad_checks, elapsed = [saved[k] for k in ('step', 'pending', 'best', 'best_step', 'stop_best', 'bad_checks', 'elapsed_seconds')]
    write_json(args.output / 'config.json', config)
    wb = None
    if args.wandb_id:
        import wandb
        wb = wandb.init(entity='helloimlixin-rutgers', project='laser', id=args.wandb_id,
            name=args.wandb_id, group='church-learned-site-codec-20260911', job_type='learned-complete-sparse-tokenizer',
            config=config, dir=str(args.output), resume='must' if args.resume else 'never')
    stopping = {'signal': None}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, frame: stopping.update(signal=number))
    started = time.monotonic()

    def log(row):
        full = {'optimizer_step': step, 'epoch_progress': stream.epoch + stream.position / stream.size,
                'elapsed_seconds': elapsed + time.monotonic() - started, **row}
        print(json.dumps(full, allow_nan=False), flush=True)
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(full, allow_nan=False) + '\n')
        write_json(args.output / 'status.json', {'pid': os.getpid(), 'best_calibration_lpips': best,
            'best_step': best_step, 'bad_checks': bad_checks, **full})
        if wb:
            wb.log(full)

    def save_last():
        atomic_torch_save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'stream': stream.state_dict(), 'config': config, 'step': step, 'pending': pending,
            'best': best, 'best_step': best_step, 'stop_best': stop_best, 'bad_checks': bad_checks,
            'elapsed_seconds': elapsed + time.monotonic() - started}, args.output / 'last.pt')

    def decode(z):
        # Frozen weights still propagate the reconstruction gradient to the codec.
        return aux.decoder(aux.post_quant_conv(z.float())).clamp(-1, 1)

    def check_frozen():
        assert all(value._version == frozen_versions[name] for name, value in list(aux.named_parameters()) + list(aux.named_buffers()))
        assert all(p.grad is None for p in aux.parameters())

    @torch.no_grad()
    def evaluate():
        nonlocal best, best_step, stop_best, bad_checks, pending
        model.eval()
        output = args.output / f'evaluations/step-{step:06d}'
        output.mkdir(parents=True, exist_ok=True)
        evaluation_indices = {
            'calibration': ('train', calibration),
            'holdout': ('holdout', torch.randperm(len(raw['holdout']['atoms']), generator=torch.Generator().manual_seed(9703))[:256]),
            'validation': ('validation', torch.arange(len(raw['validation']['atoms']))),
        }
        results = {}
        for name, (split, indices) in evaluation_indices.items():
            if args.smoke_eval_images:
                indices = indices[:args.smoke_eval_images]
            rows, token_grids = [], []
            for first in range(0, len(indices), 8):
                z = target_latent(split, indices[first:first + 8])
                target = decode(z)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    compressed, ids, _, projection = model(z, projector)
                recovered = model.decode_ids(ids, projector)['latents'].permute(0, 3, 1, 2)
                assert torch.equal(compressed, recovered)
                image = decode(compressed)
                rows.append({'lpips': perceptual(image, target).flatten().cpu(),
                    'psnr': (-10 * ((image - target) / 2).square().mean((1, 2, 3)).clamp_min(1e-12).log10()).cpu(),
                    'relative_latent_mse': ((compressed - z).square().mean((1, 2, 3)) / z.square().mean((1, 2, 3))).cpu()})
                token_grids.append(ids.cpu())
                if first == 0:
                    save_image((torch.stack([target, image], 1).flatten(0, 1) + 1) / 2, output / f'{name}.png', nrow=2)
            metrics = {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
            tokens = torch.cat(token_grids)
            count = torch.bincount(tokens.flatten(), minlength=model.config['vocabulary']).double()
            probability = count[count > 0] / count.sum()
            result = {k: float(v.mean()) for k, v in metrics.items()}
            result.update({'lpips_upper_2se': float(metrics['lpips'].mean() + 2 * metrics['lpips'].std(unbiased=len(indices) > 1) / len(indices) ** .5),
                'used_codes': int((count > 0).sum()), 'marginal_entropy_bits': float(-(probability * probability.log2()).sum()),
                'images': len(indices), 'tokens_per_image': 64})
            results[name] = result
            atomic_torch_save({'per_image': metrics, 'image_indices': indices, 'site_ids': tokens}, output / f'{name}.pt')
        score = results['calibration']['lpips']
        significant = stop_best is None or score < stop_best - .001
        bad_checks = 0 if significant else bad_checks + 1
        if significant:
            stop_best = score
        if best is None or score < best:
            best, best_step = score, step
            atomic_torch_save({'state_dict': model.state_dict(), 'step': step, 'config': config,
                              'metrics': results}, args.output / 'best.pt')
        results['passes_calibration'] = results['calibration']['lpips_upper_2se'] <= .01 and results['calibration']['relative_latent_mse'] <= .005
        write_json(output / 'metrics.json', results)
        pending = None
        check_frozen()
        save_last()
        log({'phase': 'evaluation', **{f'{name}/{k}': v for name, row in results.items() if isinstance(row, dict) for k, v in row.items()},
             'passes_calibration': results['passes_calibration']})
        if wb:
            wb.log({'reconstructions': wandb.Image(str(output / 'validation.png')), 'optimizer_step': step})
        return results

    log({'phase': 'ready', 'parameters': config['parameters'], 'vocabulary': model.config['vocabulary'], 'resumed': args.resume})
    try:
        if not args.resume:
            save_last()
        if pending is not None:
            evaluate()
        while step < args.steps and not (step >= args.minimum_steps and bad_checks >= args.patience):
            if stopping['signal'] or (args.stop_after_step and step >= args.stop_after_step):
                save_last()
                log({'phase': 'paused'})
                if wb:
                    wb.summary['training_status'] = 'paused'
                    wb.finish()
                return
            indices, progress, _ = stream.next(args.batch_size)
            indices = train_indices[indices]
            step += 1
            torch.manual_seed(args.seed + step)
            if args.warmup and step <= args.warmup:
                multiplier = .01 + .99 * step / args.warmup
            else:
                fraction = max(0., min(1., (step - args.warmup) / max(1, args.steps - args.warmup)))
                multiplier = .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
            for group, peak in zip(optimizer.param_groups, (args.lr, args.codeword_lr)):
                group['lr'] = peak * multiplier
            optimizer.zero_grad(set_to_none=True)
            model.train()
            totals, usage = {}, []
            iteration = time.monotonic()
            for local in indices.split(args.microbatch):
                with torch.no_grad():
                    z = target_latent('train', local)
                    target = decode(z)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    reconstructed, ids, penalties, projection = model(z, projector)
                image = decode(reconstructed)
                lpips = perceptual(image, target).mean()
                pixel = (image - target).abs().mean()
                relative = ((reconstructed - z).square().mean((1, 2, 3)) / z.square().mean((1, 2, 3))).mean()
                loss = lpips + .5 * pixel + .1 * relative + .25 * penalties['commitment'] + .1 * penalties['alignment'] + .1 * penalties['projection']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite codec loss at step {step}')
                (loss * len(local) / len(indices)).backward()
                values = {'loss': loss, 'lpips': lpips, 'image_l1': pixel, 'relative_latent_mse': relative,
                          'clipped_fraction': projection['clipped_fraction'], **penalties}
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) * len(local) / len(indices)
                usage.append(ids.detach().flatten().cpu())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError(f'Nonfinite codec gradient at step {step}')
            optimizer.step()
            if step == 1 or step % 10 == 0:
                check_frozen()
                log({'phase': 'train', **{f'train/{k}': v for k, v in totals.items()},
                    'train/gradient_norm': float(norm), 'train/encoder_lr': optimizer.param_groups[0]['lr'],
                    'train/codeword_lr': optimizer.param_groups[1]['lr'], 'train/used_codes_batch': int(torch.unique(torch.cat(usage)).numel()),
                    'train/step_seconds': time.monotonic() - iteration, 'gpu/peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30})
            if step % args.eval_every == 0 or step == args.steps:
                pending = step
                save_last()
                evaluate()
            elif step % 100 == 0:
                save_last()
        check_frozen()
        save_last()
        selected = torch.load(args.output / 'best.pt', weights_only=False, map_location='cpu')
        model.load_state_dict(selected['state_dict'])
        exported = model.export_codebook(projector)
        exported.update({'coefficient_bins': aux.coeff_bins.cpu(), 'coefficient_scales': aux.coeff_scales.cpu(),
                         'checkpoint_sha256': stage1_hash, 'selected_step': best_step, 'format': 'learned_complete_sparse_site_v1'})
        exported['atoms'], exported['coefficient_ids'] = exported['atoms'].short(), exported['coefficient_ids'].short()
        atomic_torch_save(exported, args.output / 'best-codebook.pt')
        result = {'reason': 'early_stop' if step < args.steps else 'step_limit', 'selected_step': best_step,
            'metrics': selected['metrics'], 'passes_calibration': selected['metrics']['calibration']['lpips_upper_2se'] <= .01 and selected['metrics']['calibration']['relative_latent_mse'] <= .005,
            'frozen_stage1_verified': True, 'unconditional_quality_tested': False}
        write_json(args.output / 'results.json', result)
        log({'phase': 'complete', 'selected_step': best_step, 'passes_calibration': result['passes_calibration']})
        if wb:
            wb.summary['training_status'] = result['reason']
            wb.summary['selected_step'] = best_step
            wb.summary['passes_calibration'] = result['passes_calibration']
            artifact = wandb.Artifact(args.wandb_id + '-codec', type='model')
            for name in ('best.pt', 'best-codebook.pt', 'results.json'):
                artifact.add_file(str(args.output / name))
            wb.log_artifact(artifact)
            wb.finish()
    except BaseException as exc:
        log({'phase': 'failed', 'error': str(exc)})
        if wb:
            wb.summary['training_status'] = 'failed'
            wb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
