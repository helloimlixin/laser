"""Measure the live ImageNet tokenizer distribution without logging to W&B."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('--memory-limit-gib', type=float, default=8.)
    args = parser.parse_args()
    sys.path[:0] = [str(args.base/'source'), str(args.base/'source/runtime')]
    import numpy as np
    import torch
    from src.training.fresh_images import EpochImageFolder
    from src.training.rqtransformer import LaserAux, image_transform

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(
        args.memory_limit_gib*2**30/torch.cuda.get_device_properties(device).total_memory,
        device)
    torch.backends.cuda.matmul.allow_tf32 = False
    scales = [8.203365325927734, 4.265638828277588,
              3.0662174224853516, 1.8273425102233887]
    aux = LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
                   coeff_scales=scales, soft_target_physical=False,
                   clamp_coeffs=False, sparsity_level=4).to(device).eval()
    dataset = EpochImageFolder(args.base/'imagenet/train', transform=image_transform(),
                               augmentation_seed=261001)
    dataset.set_epoch(args.epoch)
    assert len(dataset) == 1281167
    indices = np.random.default_rng(261009).choice(len(dataset), args.images,
                                                replace=False).tolist()
    atoms_all, coefficients_all = [], []
    latent_energy = 0.
    expected_noise_energy = {25.5875: 0., 200.: 0.}
    sites = 0
    width = 6/2047
    with torch.inference_mode():
        dictionary = aux.dictionary.float()
        gram = dictionary.T @ dictionary
        for start in range(0, len(indices), 32):
            images = torch.stack([dataset[i][0] for i in indices[start:start+32]]).to(device)
            atoms, coeffs = aux.encode_sparse_components(images, dictionary_gram=gram)
            assert coeffs.dtype == torch.float32 and torch.isfinite(coeffs).all()
            atoms_all.append(atoms.cpu().reshape(-1, 4))
            coefficients_all.append(coeffs.cpu().reshape(-1, 4))
            vectors = dictionary.T[atoms] * aux.coeff_scales[None, None, None, :, None]
            clean = (vectors * coeffs[..., None]).sum(-2)
            latent_energy += clean.square().sum().item()
            sites += clean.numel()
            for sigma in expected_noise_energy:
                _, probs = aux.compound_coeff_ids(coeffs, stochastic=False,
                                                  temp=2*(sigma*width)**2, hard=False)
                mean = (probs*aux.coeff_bins).sum(-1)
                variance = (probs*(aux.coeff_bins-mean[..., None]).square()).sum(-1)
                bias = (vectors*(mean-coeffs)[..., None]).sum(-2)
                expected_noise_energy[sigma] += (bias.square().sum()
                    + (vectors.square().sum(-1)*variance).sum()).item()
            print(json.dumps(dict(images=min(start+32, len(indices)), total=args.images,
                elapsed_seconds=time.time()-started,
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)

    atoms = torch.cat(atoms_all)
    coefficients = torch.cat(coefficients_all)
    assert atoms.shape == coefficients.shape == (args.images*64, 4)
    quantiles = torch.tensor([.5, .9, .95, .99, .995, 1.])
    depths = []
    histogram = []
    for depth, scale in enumerate(scales):
        c = coefficients[:, depth]
        rms = c.square().mean().sqrt().item()
        counts = torch.bincount(atoms[:, depth], minlength=16384)
        histogram.append(counts.tolist())
        probability = counts.double()/counts.sum()
        positive = probability[probability > 0]
        entropy = -(positive*positive.log()).sum().item()
        largest, ids = counts.topk(10)
        depths.append(dict(depth=depth, observations=c.numel(), coefficient_scale=scale,
            normalized_mean=c.mean().item(), normalized_std=c.std(correction=0).item(),
            normalized_rms=rms, physical_rms=rms*scale,
            normalized_absolute_quantiles=dict(zip(['0.5', '0.9', '0.95', '0.99', '0.995', '1.0'],
                torch.quantile(c.abs(), quantiles).tolist())),
            negative_fraction=(c < 0).float().mean().item(),
            zero_fraction=(c == 0).float().mean().item(),
            outside_uniform_range_fraction=(c.abs() > 3).float().mean().item(),
            noise_sigma_over_coefficient_rms={str(sigma): sigma*width/rms
                for sigma in expected_noise_energy},
            observed_unique_atoms=int((counts > 0).sum()),
            empirical_atom_entropy_nats=entropy,
            empirical_effective_atoms=float(np.exp(entropy)),
            most_common_atoms=[dict(id=int(atom), count=int(count),
                fraction=float(count/c.numel())) for atom, count in zip(ids, largest)],
            top10_atom_fraction=largest.sum().item()/c.numel()))
    report = dict(images=args.images, spatial_sites=args.images*64, indices=indices,
        index_seed=261009, augmentation_seed=261001, augmentation_epoch=args.epoch,
        data='ImageNet training: live deterministic per-epoch random crop and flip',
        tokenizer=str(args.base/'inputs/stage1-tokenizer.pt'),
        encoding='Live FP32 dictionary OMP, chunk size 32, matmul TF32 disabled; no clipping',
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        support=dict(depths=4, atoms=16384),
        uniform_bins=dict(count=2048, minimum=-3., maximum=3., width=width),
        depths=depths, clean_latent_rms=(latent_energy/sites)**.5,
        expected_fixed_atom_latent_relative_rms={str(sigma): (energy/latent_energy)**.5
            for sigma, energy in expected_noise_energy.items()},
        noise_energy_method='Exact independent Gaussian soft-target moments on the finite bins, '
            'including range truncation and mean bias; clean fixed atoms, no top-p filtering',
        ffhq_reference=dict(depths=2, atoms=2048, coefficient_scales=[36.208333333333336,
            8.583333333333334], fixed_center_crop_cache=True,
            validation_clipped_fraction=.004971986636519432,
            validation_images=256, calibration_quantile=.995,
            empirical_distribution_available_locally=False),
        limitations=f'Random {args.images}-image subset, not a full-cache census. Empirical atom entropy '
            'and observed vocabulary depend on sample size. No FFHQ coefficient or atom histogram '
            'was measured; different scales alone do not establish normalized distributions. '
            'Fixed-atom noise energy does not predict trained-prior FID.',
        atom_sample_sha256=hashlib.sha256(atoms.numpy().tobytes()).hexdigest(),
        coefficient_sample_sha256=hashlib.sha256(coefficients.numpy().tobytes()).hexdigest(),
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.time()-started, wandb_metrics_logged=False,
        training_modified=False)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output/'atom-histograms.json').write_text(json.dumps(histogram)+'\n')
    print(json.dumps(dict(output=str(args.output), depths=depths,
        expected_fixed_atom_latent_relative_rms=report['expected_fixed_atom_latent_relative_rms'])),
        flush=True)


if __name__ == '__main__':
    main()
