"""Choose a conservative online support temperature on existing training pixels."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torchvision.datasets import ImageFolder

from src.stochastic_compound import stochastic_omp
from src.training.stochastic_image_pairs import POLICY_VERSION, omp_precision, stochastic_components


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--support', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.support))
    from src.training.rqtransformer import LaserAux, val_image_transform
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(261007)
    device = torch.device('cuda:0')
    scales = [8.203365325927734, 4.265638828277588, 3.0662174224853516, 1.8273425102233887]
    aux = LaserAux(args.tokenizer, 16384, 2048, 3., 6.4,
        coeff_scales=scales, clamp_coeffs=False, sparsity_level=4).to(device).eval()
    dataset = ImageFolder(args.data/'train', transform=val_image_transform())
    if len(dataset) != 1281167:
        raise RuntimeError('Expected the existing complete ImageNet training set')
    indices = np.random.default_rng(261007).choice(len(dataset), 128, replace=False)
    latents = []
    for start in range(0, len(indices), 8):
        images = torch.stack([dataset[int(i)][0] for i in indices[start:start+8]]).to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = aux.quant_conv(aux.encoder(images)).permute(0, 2, 3, 1).float()
        latents.append(z)
    signals = torch.cat(latents)
    with omp_precision(device):
        gram = aux.dictionary.float().T @ aux.dictionary.float()

    def profile(z, temperature, seed):
        with omp_precision(device):
            if temperature == 0:
                # Chunk greedy OMP too, keeping its reconstruction as the anchor.
                rows = [stochastic_omp(chunk, aux.dictionary.float(), depth=4, gram=gram)
                    for chunk in z.reshape(-1, 256).split(128)]
                atoms = torch.cat([row['atoms'] for row in rows]).reshape(*z.shape[:-1], 4)
                c = torch.cat([row['coefficients'] for row in rows]).reshape_as(atoms) / aux.coeff_scales
                entropy = torch.zeros_like(c)
            else:
                out = stochastic_components(z, aux.dictionary, aux.coeff_scales,
                    temperature=temperature, gram=gram, site_chunk_size=128,
                    generator=torch.Generator(device=device).manual_seed(seed))
                atoms, c, entropy = out['atoms'], out['coefficients'], out['support_selection_entropy']
            vectors = aux.dictionary.T[atoms] * aux.coeff_scales[:, None]
            reconstruction = (vectors * c[..., None]).sum(-2)
            mse = (z - reconstruction).square().mean().item()
            # Expected error under the unchanged stochastic coefficient kernel.
            noise = []
            for a, centers, latent in zip(vectors.reshape(-1, 4, 256).split(128),
                    c.reshape(-1, 4).split(128), z.reshape(-1, 256).split(128)):
                q = (-(centers[..., None] - aux.coeff_bins).square() / .01125).softmax(-1)
                mean = (q * aux.coeff_bins).sum(-1)
                variance = (q * (aux.coeff_bins - mean[..., None]).square()).sum(-1)
                expected = (a * mean[..., None]).sum(-2)
                noise.append(((latent-expected).square().sum(-1)
                    + (variance * a.square().sum(-1)).sum(-1))/256)
        return dict(atoms=atoms, latent_mse=mse,
            expected_stochastic_coefficient_mse=torch.cat(noise).mean().item(),
            support_selection_entropy=entropy.mean((0, 1, 2)).tolist())

    def compare(z, temperature, anchor):
        rows = [profile(z, temperature, seed) for seed in (261007, 261107)]
        return dict(temperature=temperature,
            latent_mse_ratio=float(np.mean([r['latent_mse'] for r in rows]))/anchor['latent_mse'],
            expected_noisy_mse_ratio=float(np.mean([r['expected_stochastic_coefficient_mse']
                for r in rows]))/anchor['expected_stochastic_coefficient_mse'],
            changed_atom_fraction=float(np.mean([(r['atoms'] != anchor['atoms']).float().mean().item() for r in rows])),
            changed_site_fraction=float(np.mean([(r['atoms'] != anchor['atoms']).any(-1).float().mean().item() for r in rows])),
            entropy_by_depth=np.mean([r['support_selection_entropy'] for r in rows], axis=0).tolist())

    calibration, held_out = signals[:64], signals[64:]
    anchor = profile(calibration, 0., 0)
    records = []
    for temperature in [.001, .0025, .005, .01, .025, .05, .1, .25, .5]:
        row = compare(calibration, temperature, anchor)
        records.append(row)
        print(json.dumps(row), flush=True)
    eligible = [r for r in records if r['latent_mse_ratio'] <= 1.03
        and r['expected_noisy_mse_ratio'] <= 1.03 and r['changed_atom_fraction'] >= .01]
    if not eligible:
        raise RuntimeError('No calibrated temperature gives nontrivial support variation within the distortion budget')
    selected = eligible[-1]['temperature']
    confirmation = compare(held_out, selected, profile(held_out, 0., 0))
    passed = (confirmation['latent_mse_ratio'] <= 1.05
        and confirmation['expected_noisy_mse_ratio'] <= 1.05
        and confirmation['changed_atom_fraction'] >= .01)
    report = dict(policy=POLICY_VERSION, selected_temperature=selected, site_chunk_size=128,
        coefficient_temperature=.01125, calibration=records, held_out=confirmation, passed=passed,
        selection='largest candidate with <=3% extra reconstruction error and >=1% changed atom positions',
        confirmation='disjoint64 training images; <=5% extra error; two seeds',
        training_indices=indices.tolist(), images=128, image_source=str(args.data/'train'),
        representation_diagnostics_only=True, image_quality_metrics_computed=False, time=time.time())
    (args.output/'calibration.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(selected_temperature=selected, passed=passed, held_out=confirmation)), flush=True)
    if not passed:
        raise RuntimeError('Held-out representation confirmation failed; do not launch training')


if __name__ == '__main__':
    main()
