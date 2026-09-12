#!/usr/bin/env python3
"""Verify selected coefficient patterns on reproducible fresh training views."""
import argparse
import codecs
import json
import os
from pathlib import Path
import sys
import time

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tools.build_sign_probe_cache import sha256_file
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux, atomic_torch_save
from src.church_calibrated_training import AugmentedChurch
from src.church_support_pattern_training import pattern_targets
from src.models.lpips import LPIPS


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', type=Path, default=ROOT/'outputs/church-support-pattern-integer-20260911/results.json')
    args = p.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_per_process_memory_fraction(.3)
    started = time.monotonic()
    calibration = json.loads(args.calibration.read_text())
    book_path = Path(calibration['selected_codebook'])
    book = torch.load(book_path, map_location='cpu', weights_only=True)
    raw = torch.load(calibration['settings']['cache'], map_location='cpu', weights_only=False)
    aux = LaserAux(Path(calibration['settings']['stage1']), 16384, 2048, 3.,
        coeff_scales=raw['meta']['coeff_scales'], sparsity_level=4, soft_target_physical=True,
        clamp_coeffs=False, coefficient_patterns=book['coefficient_patterns']).cuda().eval().requires_grad_(False)
    perceptual = LPIPS().cuda().eval().requires_grad_(False)
    versions = [(t,t._version) for m in (aux,perceptual) for t in (*m.parameters(),*m.buffers())]
    dataset = AugmentedChurch('/tmp/laser-sign-data/church/church_outdoor_train_lmdb', raw['train']['keys'], 9803)
    indices = book['calibration_image_indices']
    rows = []
    for epoch in (0, 1, 2):
        for first in range(0,len(indices),8):
            selected = indices[first:first+8]
            images = torch.stack([dataset[int(index),epoch][0] for index in selected]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                atoms, normalized = aux.encode_sparse_components(images)
            physical = normalized.float()*aux.coeff_scales
            pattern_ids = pattern_targets(aux, atoms.long(), physical)
            continuous = (aux.dictionary.t()[atoms.long()]*physical[...,None]).sum(-2)
            quantized = aux.coefficient_pattern_latents(atoms,pattern_ids)
            target = aux.decoder(aux.post_quant_conv(continuous.permute(0,3,1,2).contiguous())).clamp(-1,1)
            actual = aux.decode_coefficient_patterns(atoms,pattern_ids)
            rows.append({'lpips':perceptual(actual,target).flatten().cpu(),
                'relative_latent_mse':((quantized-continuous).square().mean((1,2,3))/continuous.square().mean((1,2,3))).cpu(),
                'out_of_range':(physical.abs()>aux.coeff_scales*3).float().mean((1,2,3)).cpu()})
            if epoch == 0 and first == 0:
                save_image((torch.stack([target,actual],1).flatten(0,1)+1)/2,args.calibration.parent/'augmentation-reconstructions.png',nrow=2)
        print(json.dumps({'phase':'view_check','epoch_view':epoch,'images_done':(epoch+1)*len(indices),'seconds':time.monotonic()-started}),flush=True)
    per_image = {k:torch.cat([r[k] for r in rows]).reshape(3,len(indices)).mean(0) for k in rows[0]}
    # Average each image's three correlated views before estimating SE.
    metrics = {k:float(v.mean()) for k,v in per_image.items()}
    metrics['lpips_upper_2se'] = float(per_image['lpips'].mean()+2*per_image['lpips'].std()/len(indices)**.5)
    assert all(t._version==version and t.grad is None for t,version in versions)
    result = {'passes':metrics['lpips_upper_2se']<=.01 and metrics['relative_latent_mse']<=.005,
        'codebook_sha256':sha256_file(book_path),'codebook_path':str(book_path),'unique_images':len(indices),
        'views_per_image':3,'view_seed':9803,'epoch_views':[0,1,2], 'metrics':metrics,
        'standard_error_unit':'image, after averaging its three views','frozen_weights_verified':True,
        'seconds':time.monotonic()-started}
    atomic_torch_save({'view_rows':rows,'image_indices':indices},args.calibration.parent/'augmentation-per-image.pt')
    (args.calibration.parent/'augmentation-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
