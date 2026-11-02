#!/usr/bin/env python3
"""Confirm the queued LASER teacher on its actual FP32 training-image path."""
import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('/tmp/laser-rq-minimal-imagenet-20261014'))
    args = parser.parse_args()
    base = args.base.resolve()
    os.environ['RQ_UPSTREAM_ROOT'] = str(base/'upstream')
    sys.path[:0] = [str(base/'source'), str(base/'upstream')]
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from src.models.laser_tokenizer_adapter import FrozenLaserTokenizer
    from src.stochastic_compound import stochastic_omp
    from src.training.rq_imagenet_runtime import TarImageNet, sha256, write_json
    from src.training.stochastic_image_pairs import omp_precision, stochastic_components
    from rqvae.img_datasets.transforms import create_transforms
    from rqvae.utils.config import load_config

    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(261014)
    device = torch.device('cuda:0')
    recipe = OmegaConf.load(base/'inputs/recipe.yaml')
    options = recipe.tokenizer
    checkpoint = base/'inputs/stage1-model.pt'
    assert sha256(checkpoint) == options.sha256
    aux = FrozenLaserTokenizer(checkpoint, num_atoms=options.num_atoms,
        coeff_vocab_size=options.coefficient_bins, coeff_max=options.coefficient_max,
        coeff_scales=list(options.coefficient_scales), sparsity_level=options.depth).to(device)
    config = load_config(base/'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(base/'data', 'train', create_transforms(config.dataset, split='train'), seed=261001)
    assert len(dataset) == 1281167
    indices = np.random.default_rng(261014).choice(len(dataset), 128, replace=False)
    with torch.inference_mode(), torch.autocast('cuda', enabled=False):
        chunks = []
        for start in range(0, len(indices), int(recipe.teacher.encoder_batch_size)):
            images = torch.stack([dataset[int(i)][0] for i in indices[start:start+int(recipe.teacher.encoder_batch_size)]]).to(device)
            chunks.append(aux.quant_conv(aux.encoder(images.float())).permute(0, 2, 3, 1).float())
        signals = torch.cat(chunks)
        with omp_precision(device):
            gram = aux.dictionary.float().T @ aux.dictionary.float()
        chunk_size = int(recipe.teacher.site_chunk_size)

        def profile(z, temperature, seed):
            with omp_precision(device):
                if temperature == 0:
                    rows = [stochastic_omp(chunk, aux.dictionary.float(), depth=aux.sparsity_level, gram=gram)
                        for chunk in z.reshape(-1, 256).split(chunk_size)]
                    atoms = torch.cat([row['atoms'] for row in rows]).reshape(*z.shape[:-1], aux.sparsity_level)
                    coefficients = torch.cat([row['coefficients'] for row in rows]).reshape_as(atoms)/aux.coeff_scales
                    entropy = torch.zeros_like(coefficients)
                else:
                    sampled = stochastic_components(z, aux.dictionary, aux.coeff_scales,
                        temperature=temperature, gram=gram, site_chunk_size=chunk_size,
                        generator=torch.Generator(device=device).manual_seed(seed))
                    atoms, coefficients, entropy = sampled['atoms'], sampled['coefficients'], sampled['support_selection_entropy']
                vectors = aux.dictionary.T[atoms]*aux.coeff_scales[:, None]
                reconstruction = (vectors*coefficients[..., None]).sum(-2)
                noise = []
                for a, centers, latent in zip(vectors.reshape(-1, aux.sparsity_level, 256).split(chunk_size),
                        coefficients.reshape(-1, aux.sparsity_level).split(chunk_size), z.reshape(-1, 256).split(chunk_size)):
                    probabilities = (-(centers[..., None]-aux.coeff_bins).square()/float(recipe.teacher.coefficient_temperature)).softmax(-1)
                    mean = (probabilities*aux.coeff_bins).sum(-1)
                    variance = (probabilities*(aux.coeff_bins-mean[..., None]).square()).sum(-1)
                    expected = (a*mean[..., None]).sum(-2)
                    noise.append(((latent-expected).square().sum(-1)+(variance*a.square().sum(-1)).sum(-1))/256)
                return dict(atoms=atoms, latent_mse=(z-reconstruction).square().mean().item(),
                    expected_stochastic_coefficient_mse=torch.cat(noise).mean().item(),
                    entropy_by_depth=entropy.mean((0, 1, 2)).tolist())

        def compare(z, temperature, anchor):
            rows = [profile(z, temperature, seed) for seed in (261014, 261114)]
            return dict(temperature=temperature,
                latent_mse_ratio=float(np.mean([r['latent_mse'] for r in rows]))/anchor['latent_mse'],
                expected_noisy_mse_ratio=float(np.mean([r['expected_stochastic_coefficient_mse'] for r in rows]))/anchor['expected_stochastic_coefficient_mse'],
                changed_atom_fraction=float(np.mean([(r['atoms'] != anchor['atoms']).float().mean().item() for r in rows])),
                changed_site_fraction=float(np.mean([(r['atoms'] != anchor['atoms']).any(-1).float().mean().item() for r in rows])),
                entropy_by_depth=np.mean([r['entropy_by_depth'] for r in rows], axis=0).tolist())

        calibration, held_out = signals[:64], signals[64:]
        anchor = profile(calibration, 0., 0)
        held_anchor = profile(held_out, 0., 0)
        records, selected, confirmation = [], None, None
        # Confirm the existing recipe first. Reduce noise only if it fails.
        candidates = sorted({float(recipe.teacher.atom_temperature), .025, .01, .005}, reverse=True)
        candidates = [value for value in candidates if value <= float(recipe.teacher.atom_temperature)]
        for temperature in candidates:
            row = compare(calibration, temperature, anchor)
            records.append(row)
            print(json.dumps(dict(calibration=row)), flush=True)
            if row['latent_mse_ratio'] > 1.03 or row['expected_noisy_mse_ratio'] > 1.03 or row['changed_atom_fraction'] < .01:
                continue
            confirmation = compare(held_out, temperature, held_anchor)
            print(json.dumps(dict(held_out=confirmation)), flush=True)
            if confirmation['latent_mse_ratio'] <= 1.05 and confirmation['expected_noisy_mse_ratio'] <= 1.05 and confirmation['changed_atom_fraction'] >= .01:
                selected = temperature
                break
        report = dict(passed=selected is not None, selected_temperature=selected,
            configured_temperature=float(recipe.teacher.atom_temperature), coefficient_temperature=float(recipe.teacher.coefficient_temperature),
            encoder_dtype='torch.float32', encoder_autocast=False, transform='upstream imagenet256x256 train Resize/RandomCrop/HorizontalFlip',
            dataset_epoch=0, dataset_seed=261001, checkpoint_sha256=options.sha256,
            site_chunk_size=chunk_size, calibration=records, held_out=confirmation,
            linalg_backend='cusolver',
            calibration_images=64, held_out_images=64, training_indices=indices.tolist(),
            selection='Confirm existing temperature within 3% latent/noisy distortion; otherwise reduce',
            confirmation='Disjoint 64 training images; <=5% extra error; two stochastic seeds',
            representation_diagnostics_only=True, image_quality_metrics_computed=False, time=time.time())
        out = Path('/workspace/Projects/laser/outputs')/recipe.run_id
        write_json(out/'teacher-calibration-fp32-train.json', report)
        print(json.dumps(dict(passed=report['passed'], selected_temperature=selected, held_out=confirmation)), flush=True)
        if selected is None:
            raise RuntimeError('Teacher calibration failed; do not launch production')


if __name__ == '__main__':
    main()
