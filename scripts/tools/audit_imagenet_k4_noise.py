"""Measure coefficient quantization and stochastic-target distortion for K4."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime')]
from src.training.rqtransformer import LaserAux


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(261001)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = torch.load(args.cache, mmap=True, weights_only=True, map_location='cpu')
    scales = torch.tensor(x['meta']['coeff_scales'])
    indices = torch.randperm(len(x['labels']))[:512]
    atoms = x['atoms'][indices].long().cuda()
    coeffs = x['coeffs'][indices].float().cuda()
    aux = LaserAux(args.checkpoint, 16384, 2048, 3.0, coeff_scales=scales.tolist(),
                   clamp_coeffs=False, sparsity_level=4).cuda()
    physical = coeffs * scales.cuda()
    dictionary = aux.dictionary.T
    vectors = dictionary[atoms]
    exact_latent = (vectors * physical[..., None]).sum(-2)
    bins = aux.coeff_bins
    step = float(bins[1] - bins[0])
    ids = ((coeffs - bins[0]) / step).round().long().clamp(0, 2047)
    hard_coeffs = bins[ids]
    hard_physical = hard_coeffs * scales.cuda()
    hard_latent = (vectors * hard_physical[..., None]).sum(-2)

    def decode(values):
        outputs = []
        for chunk in values[:32].split(4):
            z = (vectors[len(outputs)*4:len(outputs)*4+len(chunk)] * chunk[..., None]).sum(-2)
            z = aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous())
            outputs.append(aux.decoder(z).clamp(-1, 1).cpu())
        return torch.cat(outputs)

    clean_images = decode(physical)
    hard_images = decode(hard_physical)
    energy = exact_latent.square().sum(-1).mean()
    candidates = [
        ('reference_temperature_on_k4_ranges', 'normalized', 0.5),
        ('reference_noise_in_bin_units', 'normalized', 0.01125),
        ('historical_k4_physical_temperature', 'physical', 0.03125),
        ('half_historical_noise_std', 'physical', 0.0078125),
        ('quarter_historical_noise_std', 'physical', 0.001953125),
    ]
    rows = []
    for name, space, temperature in candidates:
        aux.soft_target_physical = space == 'physical'
        coeff_step = step * scales.cuda()
        values = physical[..., None] if space == 'physical' else coeffs[..., None]
        centers = bins.view(1, 1, 1, 1, -1) * scales.cuda().view(1,1,1,4,1) if space == 'physical' else bins
        probs = (-(values-centers).square()/temperature).softmax(-1)
        sampled_ids = torch.multinomial(probs.reshape(-1,2048),1).reshape_as(coeffs)
        sampled_physical = bins[sampled_ids] * scales.cuda()
        delta = sampled_physical - physical
        sampled_latent = (vectors * sampled_physical[..., None]).sum(-2)
        images = decode(sampled_physical)
        decoded_mse = ((images-clean_images)/2).square().mean()
        row = dict(name=name, space=space, temperature=temperature,
                   physical_noise_rms_per_depth=delta.square().mean((0,1,2)).sqrt().tolist(),
                   noise_to_coefficient_rms_per_depth=(delta.square().mean((0,1,2))/physical.square().mean((0,1,2))).sqrt().tolist(),
                   noise_rms_in_bins_per_depth=(delta.square().mean((0,1,2)).sqrt()/coeff_step).tolist(),
                   boundary_fraction=float(((sampled_ids==0)|(sampled_ids==2047)).float().mean()),
                   target_entropy_per_depth=(-(probs*probs.clamp_min(1e-30).log()).sum(-1)).mean((0,1,2)).tolist(),
                   latent_perturbation_energy_fraction=float((sampled_latent-exact_latent).square().sum(-1).mean()/energy),
                   decoded_mse_vs_continuous=float(decoded_mse),
                   decoded_psnr_vs_continuous=float(-10*decoded_mse.log10()))
        rows.append(row)
        print(json.dumps(row),flush=True)
        del probs,values,centers
    result = dict(samples=512, decoded_samples=32, seed=261001,
                  stage1_checkpoint_sha256=x['meta']['stage1_checkpoint_sha256'],
                  scales=scales.tolist(), normalized_bin_step=step,
                  physical_bin_step_per_depth=(step*scales).tolist(),
                  physical_coefficient_rms_per_depth=physical.square().mean((0,1,2)).sqrt().tolist(),
                  hard_quantization_latent_energy_fraction=float((hard_latent-exact_latent).square().sum(-1).mean()/energy),
                  hard_quantization_decoded_mse=float(((hard_images-clean_images)/2).square().mean()),
                  candidates=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
