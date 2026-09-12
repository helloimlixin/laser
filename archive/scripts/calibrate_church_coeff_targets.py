#!/usr/bin/env python3
"""Choose physical target width from frozen-decoder distortion on training images."""
import argparse
import codecs
import json
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.rqtransformer import LaserAux
from scripts.tools.build_sign_probe_cache import sha256_file
from src.church_calibrated_training import physical_targets
from src.models.lpips import LPIPS


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
    raw = torch.load(args.cache, weights_only=False, map_location='cpu')
    assert sha256_file(args.stage1) == raw['meta']['checkpoint_sha256']
    aux = LaserAux(args.stage1, 16384, 2048, 3., coeff_scales=raw['meta']['coeff_scales'],
                   sparsity_level=4, soft_target_physical=True).cuda().eval()
    lpips = LPIPS().cuda().eval()
    indices = torch.randperm(len(raw['train']['atoms']), generator=torch.Generator().manual_seed(7701))[:256]
    results = []
    for sigma in (.5, .25, .125):
        errors, psnrs, latents = [], [], []
        for first in range(0, len(indices), 16):
            selected = indices[first:first + 16]
            atoms = raw['train']['atoms'][selected].cuda().long()
            physical = raw['train']['coefficients'][selected].cuda()
            z = (aux.dictionary.t()[atoms] * physical[..., None]).sum(-2)
            clean = aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)
            torch.manual_seed(17701 + first)
            packed, _ = physical_targets(aux, atoms, physical, 'soft', sigma)
            image = aux.decode_compound(atoms, packed % 2048)
            errors.append(lpips(image, clean).flatten().cpu())
            psnrs.append((-10 * ((image - clean) / 2).square().mean((1, 2, 3)).clamp_min(1e-12).log10()).cpu())
            latents.append(((aux.compound_embeddings(atoms, packed % 2048).sum(-2) - z).square().mean((1, 2, 3)) / z.square().mean((1, 2, 3))).cpu())
            if first == 0:
                save_image((torch.stack([clean[:8], image[:8]], 1).flatten(0, 1) + 1) / 2,
                           args.output / f'sigma-{sigma}.png', nrow=2)
        e = torch.cat(errors)
        upper = float(e.mean() + 2 * e.std() / len(e) ** .5)
        row = {'sigma': sigma, 'temperature': 2 * sigma ** 2, 'train_lpips': float(e.mean()),
               'lpips_upper_2se': upper, 'train_psnr': float(torch.cat(psnrs).mean()),
               'relative_latent_mse': float(torch.cat(latents).mean())}
        row['passes'] = upper <= .01 and row['relative_latent_mse'] <= .005
        results.append(row)
        print(json.dumps(row), flush=True)
    passed = [r for r in results if r['passes']]
    if not passed:
        raise RuntimeError('No tested width passed the predeclared distortion thresholds')
    result = {'selected_sigma': max(r['sigma'] for r in passed), 'candidates': results,
              'selection': 'largest sigma with LPIPS mean + 2 SE <= .01 and relative latent MSE <= .005',
              'fit_split': 'train', 'images': len(indices), 'indices': indices.tolist(),
              'samples': 'full untruncated target distribution, matching training',
              'checkpoint_sha256': raw['meta']['checkpoint_sha256'], 'cache_sha256': sha256_file(args.cache)}
    (args.output / 'calibration.json').write_text(json.dumps(result, indent=2))
    print('Selected sigma', result['selected_sigma'], flush=True)


if __name__ == '__main__':
    main()
