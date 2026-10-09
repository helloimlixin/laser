"""Check FFHQ-style percentile scaling and clipping on fresh ImageNet views."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--distribution', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--images', type=int, default=4096)
    args = p.parse_args()
    sys.path[:0] = [str(args.base/'source'), str(args.base/'source/runtime')]
    import numpy as np
    import torch
    from src.training.fresh_images import EpochImageFolder
    from src.training.rqtransformer import LaserAux, image_transform
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(12*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    source = json.loads(args.distribution.read_text())
    old_scales = [d['coefficient_scale'] for d in source['depths']]
    scales = [d['coefficient_scale']*d['normalized_absolute_quantiles']['0.995']/3
              for d in source['depths']]
    aux = LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
        coeff_scales=old_scales, clamp_coeffs=False, soft_target_physical=False,
        sparsity_level=4).cuda().eval()
    new_aux = LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
        coeff_scales=scales, clamp_coeffs=True, soft_target_physical=False,
        sparsity_level=4).cuda().eval()
    dataset = EpochImageFolder(args.base/'imagenet/train', transform=image_transform(), augmentation_seed=261001)
    dataset.set_epoch(0)
    indices = np.random.default_rng(261095).choice(len(dataset), args.images, replace=False).tolist()
    coefficients, latent_energy, clipping_error, quantization_error = [], 0., 0., 0.
    noise_error, sites, pixel_error, pixel_values = 0., 0, 0., 0
    clipped_latent_energy = 0.
    started = time.time()
    ratio = torch.tensor(old_scales, device='cuda')/torch.tensor(scales, device='cuda')
    with torch.inference_mode():
        gram = aux.dictionary.T @ aux.dictionary
        for start in range(0, args.images, 32):
            images = torch.stack([dataset[i][0] for i in indices[start:start+32]]).cuda()
            atoms, c = aux.encode_sparse_components(images, dictionary_gram=gram)
            coefficients.append(c.cpu().reshape(-1, 4))
            raw = c*ratio
            clipped = raw.clamp(-3, 3)
            z = aux.physical_contributions(atoms, c).sum(-2)
            limited_z = new_aux.physical_contributions(atoms, clipped).sum(-2)
            ids, probabilities = new_aux.compound_coeff_ids(clipped, temp=.5, stochastic=False)
            quantized = new_aux.coeff_bins[ids]
            quantized_z = new_aux.physical_contributions(atoms, quantized).sum(-2)
            latent_energy += z.square().sum().item()
            clipped_latent_energy += limited_z.square().sum().item()
            clipping_error += (z-limited_z).square().sum().item()
            quantization_error += (z-quantized_z).square().sum().item()
            vectors = new_aux.dictionary.T[atoms]*torch.tensor(scales, device='cuda')[..., None]
            mean = (probabilities*new_aux.coeff_bins).sum(-1)
            variance = (probabilities*(new_aux.coeff_bins-mean[..., None]).square()).sum(-1)
            bias = (vectors*(mean-clipped)[..., None]).sum(-2)
            noise_error += (bias.square().sum()+(vectors.square().sum(-1)*variance).sum()).item()
            sites += z.numel()
            if start < 128:
                for offset in range(0, len(images), 8):
                    def decode(latent):
                        return aux.decoder(aux.post_quant_conv(latent[offset:offset+8].permute(0, 3, 1, 2).contiguous())).clamp(-1, 1)
                    native, limited = decode(z), decode(quantized_z)
                    pixel_error += (native-limited).abs().sum().item()*127.5
                    pixel_values += native.numel()
            print(json.dumps(dict(images=min(start+32,args.images),total=args.images,elapsed_seconds=time.time()-started)), flush=True)
    coefficients = torch.cat(coefficients)
    depths = []
    for d in range(4):
        c = coefficients[:, d]
        normalized = c*old_scales[d]/scales[d]
        clipped = normalized.clamp(-3, 3)
        depths.append(dict(depth=d, previous_scale=old_scales[d], calibrated_scale=scales[d],
            previous_normalized_min=c.min().item(), previous_normalized_max=c.max().item(),
            physical_min=c.min().item()*old_scales[d], physical_max=c.max().item()*old_scales[d],
            unclipped_new_normalized_min=normalized.min().item(), unclipped_new_normalized_max=normalized.max().item(),
            clipped_fraction=(normalized.abs()>3).float().mean().item(),
            clipped_normalized_rms=clipped.square().mean().sqrt().item(),
            clipped_normalized_range=[clipped.min().item(),clipped.max().item()],
            noise_sigma_normalized=.5, noise_sigma_bins=.5/(6/2047),
            sigma_over_clipped_rms=.5/clipped.square().mean().sqrt().item(),
            physical_clamping_limit=3*scales[d], physical_noise_sigma=.5*scales[d]))
    clipped_fraction=sum(d['clipped_fraction'] for d in depths)/4
    report = dict(passed=clipped_fraction<=.02, calibration_images=source['images'], validation_images=args.images,
        validation_seed=261095, validation_augmentation_epoch=0, coefficients_per_depth=args.images*64,
        coefficient_scales=scales, uniform_bins=dict(count=2048,range=[-3,3],width=6/2047),
        target_temperature=.5, target_sigma_normalized=.5, target_sigma_bins=.5/(6/2047),
        clipping=True, clipping_fraction=clipped_fraction, depths=depths,
        clipping_latent_relative_rms=(clipping_error/latent_energy)**.5,
        clipping_and_quantization_latent_relative_rms=(quantization_error/latent_energy)**.5,
        expected_fixed_atom_noise_relative_rms=(noise_error/clipped_latent_energy)**.5,
        clipped_quantized_decode_mae_255=pixel_error/pixel_values, decoded_validation_images=128,
        ffhq_cache_clipped_fraction=.004971986636519432,
        matching_rule='q99.5(abs(physical coefficient))/3 per depth; clamp normalized coefficients to [-3,3]; Gaussian temperature .5 -> nominal sigma .5',
        limitations='ImageNet calibration and validation are independent 4096-image subsets, not the full corpus. Noise energy is a codec diagnostic and does not establish prior FID.',
        elapsed_seconds=time.time()-started, peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    assert report['passed']
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__=='__main__':
    main()
