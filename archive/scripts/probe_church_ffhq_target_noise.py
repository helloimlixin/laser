#!/usr/bin/env python3
"""Decode oracle supports with the transferred soft targets; no weights change."""
import argparse
import codecs
import json
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.rqtransformer import LaserAux, atomic_torch_save
from scripts.tools.build_sign_probe_cache import sha256_file
from src.models.lpips import LPIPS
from src.models.rqtransformer.transformers import _top_p_probs


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=ROOT / 'outputs/church-ffhq-recipe-20260911/continuous-cache.pt')
    p.add_argument('--stage1', type=Path, default=ROOT / 'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.manual_seed(6701)
    raw = torch.load(args.cache, map_location='cpu', weights_only=False)
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=False).cuda().eval()
    perceptual = LPIPS().cuda().eval()
    train = raw['train']['coefficients']
    rms = train.square().mean((0, 1, 2)).sqrt()
    result = {'stage1_sha256': sha256_file(args.stage1),
              'physical_coefficient_rms': rms.tolist(),
              'soft_target_untruncated_std': (aux.coeff_scales.cpu() * .5).tolist(),
              'noise_std_over_coefficient_rms': (aux.coeff_scales.cpu() * .5 / rms).tolist(),
              'seed': 6701, 'reference': 'continuous clean frozen-tokenizer reconstruction',
              'scope': 'oracle supports and known coefficients; this is not unconditional FID'}
    conditions = [('nearest', False, 1.), ('normalized_p1', False, 1.),
                  ('normalized_p085', False, .85), ('physical_p085', True, .85)]
    for split, count in [('validation', 300), ('holdout', 256)]:
        records = {name: [] for name, _, _ in conditions}
        uniform = torch.rand((count, 8, 8, 4), generator=torch.Generator().manual_seed(6701))
        for start in range(0, count, 16):
            atoms = raw[split]['atoms'][start:start + 16].cuda().long()
            coefficients = raw[split]['coefficients'][start:start + 16].cuda()
            z = (aux.dictionary.t()[atoms] * coefficients[..., None]).sum(-2)
            target = aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)
            grid = [target[:8]]
            for name, physical, top_p in conditions:
                aux.soft_target_physical = physical
                ids, probs = aux.compound_coeff_ids(coefficients / aux.coeff_scales, stochastic=False, temp=.5)
                if name != 'nearest':
                    flattened = probs.flatten(0, -2)
                    if top_p < 1:
                        flattened = _top_p_probs(flattened, top_p)
                    flattened = flattened / flattened.sum(-1, keepdim=True)
                    cdf = flattened.cumsum(-1).contiguous()
                    u = uniform[start:start + len(atoms)].cuda().flatten()[:, None].contiguous()
                    ids = torch.searchsorted(cdf, u).clamp_max(2047).reshape_as(atoms)
                sampled_coeffs = aux.coeff_bins[ids] * aux.coeff_scales
                image = aux.decode_compound(atoms, ids)
                mse = ((image - target) / 2).square().mean((1, 2, 3))
                row = {'psnr': -10 * mse.clamp_min(1e-12).log10(),
                       'lpips': perceptual(image, target).flatten(),
                       'coefficient_mae': (sampled_coeffs - coefficients).abs().mean((1, 2, 3)),
                       'sign_flip_fraction': ((sampled_coeffs >= 0) != (coefficients >= 0)).float().mean((1, 2, 3)),
                       'relative_latent_mse': (aux.compound_embeddings(atoms, ids).sum(-2) - z).square().mean((1, 2, 3)) / z.square().mean((1, 2, 3))}
                records[name].append({k: v.cpu() for k, v in row.items()})
                grid.append(image[:8])
            if start == 0:
                save_image((torch.stack(grid, 1).flatten(0, 1) + 1) / 2,
                           args.output / f'{split}-oracle-targets.png', nrow=len(grid))
        per_image = {name: {k: torch.cat([r[k] for r in rows]) for k in rows[0]}
                     for name, rows in records.items()}
        atomic_torch_save(per_image, args.output / f'{split}-per-image.pt')
        result[split] = {name: {k: float(v.mean()) for k, v in rows.items()} for name, rows in per_image.items()}
        delta = per_image['normalized_p085']['lpips'] - per_image['physical_p085']['lpips']
        result[split]['paired_lpips_normalized_minus_physical'] = {
            'mean': float(delta.mean()), 'standard_error': float(delta.std() / len(delta) ** .5)}
        print(json.dumps({'split': split, 'images': count, 'results': result[split]}), flush=True)
    (args.output / 'results.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
