#!/usr/bin/env python3
"""Select a larger joint coefficient vocabulary before stage-2 training."""
import argparse
import codecs
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file
from src.training.rqtransformer import LaserAux, atomic_torch_save
from src.coefficient_pattern_codec import assign_coefficient_patterns, fit_coefficient_patterns, selected_support_grams
from src.complete_sparse_codec import sparse_latents
from src.models.lpips import LPIPS
from src.support_pattern_integer_codec import pack_support_pattern, decode_support_pattern_integers


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sizes', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192])
    p.add_argument('--iterations', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--cache', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911/continuous-cache.pt')
    p.add_argument('--split-indices', type=Path, default=ROOT / 'outputs/church-complete-site-token-20260911/fit-indices.pt')
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    args = p.parse_args()
    if min(args.sizes + [args.iterations, args.batch_size]) < 1:
        p.error('Sizes and counts must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.3)
    torch.serialization.add_safe_globals([codecs.encode])
    started = time.monotonic()

    def log(row):
        row = {'elapsed_seconds': time.monotonic() - started, **row}
        print(json.dumps(row, allow_nan=False), flush=True)
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')

    def save():
        (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    raw = torch.load(args.cache, map_location='cpu', weights_only=False)
    split = torch.load(args.split_indices, map_location='cpu', weights_only=True)
    fit_indices, calibration = split['fit_site_indices'], split['calibration_image_indices']
    assert not torch.isin(fit_indices // 64, calibration).any()
    keys = [set(raw[s]['keys']) for s in ('train', 'holdout', 'validation')]
    assert not any(keys[i] & keys[j] for i in range(3) for j in range(i))
    checkpoint_hash = sha256_file(args.stage1)
    assert checkpoint_hash == raw['meta']['checkpoint_sha256']
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
        sparsity_level=4, soft_target_physical=True, clamp_coeffs=False).cuda().eval().requires_grad_(False)
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    frozen = [(t, t._version) for m in (aux, perceptual) for t in (*m.parameters(), *m.buffers())]
    dictionary, bins, scales = aux.dictionary.t(), aux.coeff_bins, aux.coeff_scales
    atoms = raw['train']['atoms'].reshape(-1, 4)[fit_indices].cuda().long()
    physical = raw['train']['coefficients'].reshape(-1, 4)[fit_indices].cuda()
    grams = selected_support_grams(atoms, dictionary)
    result = {'format': 'church_support_pattern_integer_v1', 'phase': 'calibration',
        'checkpoint_sha256': checkpoint_hash, 'cache_sha256': sha256_file(args.cache),
        'split_indices_sha256': sha256_file(args.split_indices), 'fit_sites': len(fit_indices),
        'calibration_excluded_from_fit': True, 'fitting_seed': 9721, 'neural_weights_updated': False,
        'selection': 'smallest tested vocabulary with calibration LPIPS mean+2SE <= .01 AND relative latent MSE <= .005',
        'reference': 'continuous frozen LASER reconstructions; not unconditional generation',
        'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'candidates': {}, 'selected_vocabulary': None, 'source_sha256': {}}
    source_dir = args.output / 'source'
    source_dir.mkdir()
    for name in ('scripts/calibrate_church_support_patterns.py', 'src/support_pattern_integer_codec.py',
                 'src/coefficient_pattern_codec.py', 'src/complete_sparse_codec.py',
                 'scripts/train_official_rqtransformer_laser_stage2.py'):
        result['source_sha256'][name] = sha256_file(ROOT / name)
        shutil.copy2(ROOT / name, source_dir / Path(name).name)
    save()

    def decode(z):
        return aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)

    def evaluate(label, split_name, indices, pattern_bins, output):
        patterns = bins[pattern_bins] * scales
        rows, integer_rows, token_rows = [], [], []
        for first in range(0, len(indices), args.batch_size):
            selected = indices[first:first + args.batch_size]
            atoms = raw[split_name]['atoms'][selected].cuda().long()
            coefficients = raw[split_name]['coefficients'][selected].cuda()
            true_z = (dictionary[atoms] * coefficients[..., None]).sum(-2)
            ids, _ = assign_coefficient_patterns(coefficients.reshape(-1, 4), patterns,
                grams=selected_support_grams(atoms.reshape(-1, 4), dictionary), chunk_size=4096)
            ids = ids.reshape(atoms.shape[:-1])
            integers = pack_support_pattern(atoms, ids, num_patterns=len(patterns))
            recovered_atoms, recovered_bins = decode_support_pattern_integers(integers, pattern_bins.cpu())
            recovered_atoms = recovered_atoms.cuda().reshape_as(atoms)
            recovered_bins = recovered_bins.cuda().reshape_as(atoms)
            assert torch.equal(recovered_atoms, atoms)
            assert torch.equal(recovered_bins, pattern_bins[ids])
            z = sparse_latents(recovered_atoms, recovered_bins, dictionary, bins, scales)
            image, target = decode(z), decode(true_z)
            values = {'lpips': perceptual(image, target).flatten(),
                'psnr': -10 * ((image-target)/2).square().mean((1,2,3)).clamp_min(1e-12).log10(),
                'relative_latent_mse': (z-true_z).square().mean((1,2,3))/true_z.square().mean((1,2,3)),
                'coefficient_sign_error': ((patterns[ids] >= 0) != (coefficients >= 0)).float().mean((1,2,3))}
            rows.append({k:v.cpu() for k,v in values.items()})
            integer_rows.extend(integers)
            token_rows.append(ids.cpu().int())
            if first == 0:
                save_image((torch.stack([target, image],1).flatten(0,1)+1)/2, output/f'{label}-reconstructions.png', nrow=2)
        rows = {k:torch.cat([r[k] for r in rows]) for k in rows[0]}
        metrics = {k:float(v.mean()) for k,v in rows.items()}
        metrics['lpips_upper_2se'] = float(rows['lpips'].mean()+2*rows['lpips'].std()/len(indices)**.5)
        tokens = torch.cat(token_rows)
        counts = torch.bincount(tokens.flatten().long(), minlength=len(patterns)).double()
        probabilities = counts[counts>0]/counts.sum()
        metrics.update({'images':len(indices), 'used_patterns':int((counts>0).sum()),
            'marginal_pattern_entropy_bits':float(-(probabilities*probabilities.log2()).sum())})
        atomic_torch_save({'per_image':rows, 'image_indices':indices, 'pattern_ids':tokens,
            'complete_site_integers':integer_rows, 'integer_grid_shape':list(tokens.shape)}, output/f'{label}.pt')
        log({'phase':'evaluation', 'vocabulary':len(patterns), 'split':label, 'metrics':metrics})
        return metrics

    for size in sorted(set(args.sizes)):
        output = args.output / f'patterns-{size}'
        output.mkdir()
        log({'phase':'fit_start', 'vocabulary':size})
        patterns = fit_coefficient_patterns(physical, num_patterns=size, grams=grams,
            iterations=args.iterations, seed=9721, chunk_size=4096,
            progress=lambda iteration,error,empty: log({'phase':'lloyd','vocabulary':size,'iteration':iteration,
                'mean_squared_latent_error':error,'empty_clusters':empty}) if iteration % 4 == 0 else None)
        normalized = patterns/scales
        pattern_bins = torch.bucketize(normalized.contiguous(), (bins[1:]+bins[:-1])/2)
        patterns = bins[pattern_bins]*scales
        nominal_bits = (16384**4*size-1).bit_length()
        book = {'format':result['format'], 'checkpoint_sha256':checkpoint_hash,
            'coefficient_patterns':patterns.cpu(), 'pattern_coefficient_ids':pattern_bins.cpu(),
            'coefficient_bins':bins.cpu(), 'coefficient_scales':scales.cpu(), 'num_atoms':16384,
            'depth':4, 'nominal_bits_per_site':nominal_bits, 'fit_site_indices':fit_indices,
            'calibration_image_indices':calibration}
        atomic_torch_save(book, output/'codebook.pt')
        metrics = evaluate('calibration','train',calibration,pattern_bins,output)
        passed = metrics['lpips_upper_2se'] <= .01 and metrics['relative_latent_mse'] <= .005
        result['candidates'][str(size)] = {'nominal_bits_per_site':nominal_bits, 'calibration':metrics,
            'clipped_pattern_coefficient_fraction':float(((normalized < -3)|(normalized > 3)).float().mean()),
            'passes_calibration':passed, 'codebook_sha256':sha256_file(output/'codebook.pt')}
        save()
        if passed:
            result['selected_vocabulary'] = size
            result['selection_fixed_before_holdout_evaluation'] = True
            result['selected_codebook'] = str((output/'codebook.pt').resolve())
            save()
            heldout = torch.randperm(len(raw['holdout']['atoms']),generator=torch.Generator().manual_seed(9703))[:256]
            for label, indices in [('holdout',heldout), ('validation',torch.arange(len(raw['validation']['atoms'])))]:
                result['candidates'][str(size)][label] = evaluate(label,label,indices,pattern_bins,output)
                save()
            break
    assert all(t._version == version and t.grad is None for t,version in frozen)
    result.update({'phase':'complete','frozen_weight_versions_verified':True,'elapsed_seconds':time.monotonic()-started})
    save()
    log({'phase':'complete','selected_vocabulary':result['selected_vocabulary']})


if __name__ == '__main__':
    main()
