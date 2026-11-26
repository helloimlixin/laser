"""Measure coefficient-noise color distortion with fixed ImageNet OMP atoms.

This is a local tokenizer diagnostic; it does not load or modify the prior,
log W&B metrics, or estimate FID/IS.
"""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=int, default=16)
    parser.add_argument('--draws', type=int, default=4)
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    parser.add_argument('--memory-limit-gib', type=float, default=8.)
    args = parser.parse_args()
    sys.path[:0] = [str(args.base/'source'), str(args.base/'source/runtime')]
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.datasets import ImageFolder
    from src.training.rqtransformer import LaserAux, val_image_transform
    from src.models.rqtransformer.transformers import _top_p_probs, sample_from_logits
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(
        args.memory_limit_gib*2**30/torch.cuda.get_device_properties(device).total_memory,
        device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scales = [8.203365325927734, 4.265638828277588,
              3.0662174224853516, 1.8273425102233887]
    aux = LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
                   coeff_scales=scales, soft_target_physical=False,
                   clamp_coeffs=False, sparsity_level=4).to(device).eval()
    assert aux.dictionary.dtype == torch.float32
    dataset = ImageFolder(args.base/'imagenet'/args.split, transform=val_image_transform())
    assert len(dataset) == {'train': 1281167, 'val': 50000}[args.split]
    indices = np.random.default_rng(261009).choice(len(dataset), args.images,
                                                replace=False).tolist()
    policies = {'nearest_bin': (0., None), 'sigma25.5875bins': (25.5875, None),
                'sigma200bins': (200., None), 'sigma200bins_top_p0.85': (200., .85)}
    rows = []
    pictures = []
    bin_width = 6/2047

    def decode(z):
        # Match the live run: FP32 decoder followed by [-1, 1] clamping.
        raw = aux.decoder(aux.post_quant_conv(z.permute(0, 3, 1, 2).contiguous()))
        assert raw.dtype == torch.float32 and torch.isfinite(raw).all()
        return raw.clamp(-1, 1).add(1).mul(.5), (raw.abs() > 1).float().mean().item()

    with torch.inference_mode():
        gram = aux.dictionary.T @ aux.dictionary
        for image_number, index in enumerate(indices):
            image, label = dataset[index]
            image = image[None].to(device)
            atoms, coeffs = aux.encode_sparse_components(image, dictionary_gram=gram)
            clean_latent = aux.physical_contributions(atoms, coeffs).sum(-2)
            continuous, continuous_clipping = decode(clean_latent)
            original = image.add(1).mul(.5)
            panel = [original[0].cpu()]
            for policy, (sigma_bins, top_p) in policies.items():
                for draw in range(1 if sigma_bins == 0 else args.draws):
                    seed = 261009 + image_number*100 + draw
                    torch.manual_seed(seed)
                    ids, probabilities = aux.compound_coeff_ids(
                        coeffs, stochastic=sigma_bins != 0,
                        temp=2*(sigma_bins*bin_width)**2, hard=sigma_bins == 0)
                    if top_p is not None:
                        probabilities = _top_p_probs(probabilities, top_p)
                        torch.manual_seed(seed)
                        ids = torch.multinomial(probabilities.reshape(-1, 2048), 1).reshape_as(coeffs)
                        if image_number == 0 and draw == 0:
                            torch.manual_seed(seed)
                            logits = -(coeffs[..., None]-aux.coeff_bins).square()/(2*(sigma_bins*bin_width)**2)
                            expected_ids = sample_from_logits(logits.reshape(-1, 2048),
                                temperature=1., top_k=2048, top_p=top_p).reshape_as(ids)
                            assert torch.equal(ids, expected_ids)
                    latent = aux.compound_embeddings(atoms, ids).sum(-2)
                    decoded, clipping = decode(latent)
                    # Check the instrumented path against the actual native decoder.
                    if image_number == 0 and draw == 0:
                        assert torch.equal(decoded, aux.decode_compound(atoms, ids).add(1).mul(.5))
                    delta = decoded-continuous
                    coarse_delta = F.avg_pool2d(delta, 32)
                    coefficient_error = aux.coeff_bins[ids]-coeffs
                    probability_mean = (probabilities*aux.coeff_bins).sum(-1)
                    probability_std = (
                        (probabilities*(aux.coeff_bins-probability_mean[..., None]).square()).sum(-1)
                    ).sqrt()/bin_width
                    row = dict(image_index=index, label=label, policy=policy, draw=draw, top_p=top_p,
                        seed=seed, rgb_mae_255=delta.abs().mean().item()*255,
                        coarse_rgb_mae_255=coarse_delta.abs().mean().item()*255,
                        reconstruction_mae_255=(decoded-original).abs().mean().item()*255,
                        latent_relative_rms=(latent-clean_latent).square().mean().sqrt().item()
                            /clean_latent.square().mean().sqrt().item(),
                        raw_decoder_clipped_fraction=clipping,
                        continuous_decoder_clipped_fraction=continuous_clipping,
                        coefficient_rms_error_bins_by_depth=(coefficient_error.square()
                            .mean((0, 1, 2)).sqrt()/bin_width).tolist(),
                        mean_actual_kernel_std_bins_by_depth=probability_std.mean((0, 1, 2)).tolist(),
                        coefficient_outside_range_fraction=(coeffs.abs() > 3).float().mean().item())
                    rows.append(row)
                    if draw == 0:
                        panel.append(decoded[0].cpu())
            pictures.append(torch.stack(panel))
            print(json.dumps(dict(completed=image_number+1, total=args.images,
                elapsed_seconds=time.time()-started,
                peak_memory_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)

    fields = ['rgb_mae_255', 'coarse_rgb_mae_255', 'reconstruction_mae_255',
              'latent_relative_rms', 'raw_decoder_clipped_fraction',
              'continuous_decoder_clipped_fraction', 'coefficient_outside_range_fraction']
    summaries = {}
    for policy in policies:
        selected = [row for row in rows if row['policy'] == policy]
        per_image = [{field: np.mean([row[field] for row in selected
                                     if row['image_index'] == index])
                      for field in fields} for index in indices]
        summaries[policy] = {field: dict(mean=float(np.mean([row[field] for row in per_image])),
                                        median=float(np.median([row[field] for row in per_image])),
                                        min=float(min(row[field] for row in per_image)),
                                        max=float(max(row[field] for row in per_image)))
                              for field in fields}
    report = dict(seed=261009, images=args.images, draws=args.draws, indices=indices,
        split=f'ImageNet {args.split}: deterministic center crop',
        source=str(args.base/'source/src/training/rqtransformer.py'),
        tokenizer=str(args.base/'inputs/stage1-tokenizer.pt'),
        atom_support='identical clean FP32 OMP atoms for all policies',
        decoder='native frozen ImageNet stage1; FP32, TF32 disabled',
        bins=dict(count=2048, minimum=-3, maximum=3, width=bin_width),
        coefficient_scales=scales, policies=policies, summary=summaries, rows=rows,
        metrics='RGB errors in 8-bit pixel units relative to continuous clean-token reconstruction; '
                'coarse error uses non-overlapping 32x32 pixel averages',
        limitations='Small fixed-atom codec experiment, not a trained-prior ablation or FID/IS evaluation. '
                    'The top-p condition filters the ideal coefficient target kernel, not learned prior logits. '
                    'Targets near range edges have truncated noise; errors include bin quantization.',
        wandb_metrics_logged=False, training_modified=False,
        peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.time()-started)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    titles = ['Original', 'Nearest bin', '25.59-bin sigma', '200-bin sigma', '200 bins, top-p .85']
    for start in range(0, args.images, 4):
        panels = pictures[start:start+4]
        fig, axes = plt.subplots(len(panels), len(titles), figsize=(15, 3*len(panels)), squeeze=False)
        for row, panel in enumerate(panels):
            for col in range(len(titles)):
                axes[row, col].imshow(panel[col].permute(1, 2, 0).numpy(), vmin=0, vmax=1)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(titles[col])
            axes[row, 0].set_ylabel(f'{args.split} index {indices[start+row]}')
        fig.suptitle('Fixed atom codes; native coefficient sampling and FP32 decoding (first draw)')
        fig.tight_layout()
        fig.savefig(args.output/f'comparison_{start//4+1:02d}.png', dpi=110)
        plt.close(fig)
    print(json.dumps(dict(summary=summaries, output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
