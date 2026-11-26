#!/usr/bin/env python3
"""Read-only validation probe of the active finite joint dictionary."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
BASE = Path('/tmp/laser-joint20k-depth-loss-epoch1-20261018/control')
OUT = ROOT / 'outputs/imagenet-laser-joint20k-quality-fix-20261018/dictionary-audit'


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
    sys.path[:0] = [str(BASE / 'source'), str(BASE / 'upstream'), str(BASE), str(BASE / 'runtime')]
    import numpy as np
    import torch
    from src.models.laser_joint_tokenizer import FrozenJointLaserTokenizer
    from src.training.rq_imagenet_runtime import TarImageNet
    from rqvae.utils.config import load_config
    from rqvae.img_datasets.transforms import create_transforms
    torch.set_num_threads(16)
    torch.manual_seed(261018605)
    aux = FrozenJointLaserTokenizer(BASE / 'inputs/stage1-model.pt', BASE / 'inputs/joint-codebook.pt')
    q = aux.quantizer
    config = load_config(BASE / 'inputs/stage2-training-config.yaml')
    dataset = TarImageNet(BASE / 'data', 'val', create_transforms(config.dataset, split='val'))
    indices = np.random.default_rng(261018605).permutation(len(dataset))[:128].tolist()
    total = repeated = opposite = 0
    depth_counts = torch.zeros(5, dtype=torch.float64)
    residual_energies = torch.zeros(6, dtype=torch.float64)
    outside_range = torch.zeros(5, dtype=torch.float64)
    max_increase = 0.
    max_score_error = 0.
    duplicate_codes = []
    by_atom_low, by_atom_high = [], []
    for d in range(5):
        low, high = torch.full((8192,), torch.inf), torch.full((8192,), -torch.inf)
        low.scatter_reduce_(0, q.atom_ids[d, 1:], q.coefficients[d, 1:], reduce='amin')
        high.scatter_reduce_(0, q.atom_ids[d, 1:], q.coefficients[d, 1:], reduce='amax')
        by_atom_low.append(low); by_atom_high.append(high)
    with torch.inference_mode():
        for first in range(0, len(indices), 8):
            images = torch.stack([dataset[index][0] for index in indices[first:first + 8]])
            x = aux.encode(images).reshape(-1, 256)
            value = q.quantize(x, chunk_size=128, return_projections=True)
            codes = value['codes']
            atoms = torch.stack([q.atom_ids[d, codes[:, d]] for d in range(5)], -1)
            coefficients = torch.stack([q.coefficients[d, codes[:, d]] for d in range(5)], -1)
            repeated_here = torch.zeros(len(x), dtype=torch.bool)
            opposite_here = torch.zeros_like(repeated_here)
            for d in range(5):
                for other in range(d):
                    match = (atoms[:, d] == atoms[:, other]) & (codes[:, d] != 0) & (codes[:, other] != 0)
                    repeated_here |= match
                    opposite_here |= match & (coefficients[:, d] * coefficients[:, other] < 0)
            repeated += int(repeated_here.sum()); opposite += int(opposite_here.sum())
            residual = x.clone()
            residual_energies[0] += residual.square().sum().double()
            for d in range(5):
                # Compare the optimized score to direct physical-codeword math
                # at actual residuals, using FP64 as an independent reference.
                if first == 0:
                    r = residual[:16].double()
                    book = q.expanded_codebook(d).double()
                    dense = 2 * r @ book.T - book.square().sum(-1)
                    score = q.scores(residual[:16], d)
                    max_score_error = max(max_score_error, float((score.double() - dense).abs().max()))
                    assert torch.equal(score.argmax(-1), dense.argmax(-1))
                    torch.testing.assert_close(score.double(), dense, rtol=3.e-5, atol=.001)
                before = residual.square().sum(-1)
                residual -= q.embed_depth(codes[:, d], d)
                after = residual.square().sum(-1)
                max_increase = max(max_increase, float((after - before).max()))
                assert (after <= before + 2.e-5 * before.clamp_min(1.)).all()
                residual_energies[d + 1] += after.sum().double()
                depth_counts[d] += (codes[:, d] == 0).sum()
                projection = value['projections'][:, d]
                valid = codes[:, d] != 0
                outside = (projection < by_atom_low[d][atoms[:, d]]) | (projection > by_atom_high[d][atoms[:, d]])
                outside_range[d] += (outside & valid).sum()
            torch.testing.assert_close(x - residual, q.embed(codes).sum(-2), rtol=3.e-5, atol=3.e-6)
            total += len(x)
            write('probe-status.json', dict(images=first + len(images), target_images=128, time=time.time()))
    result = dict(completed=True, validation_images=len(indices), latent_sites=total,
        validation_indices=indices, repeated_atom_site_fraction=repeated / total,
        opposite_sign_repeated_atom_site_fraction=opposite / total,
        null_probability_by_depth=(depth_counts / total).tolist(),
        chosen_atom_projection_outside_available_level_range=(outside_range / total).tolist(),
        residual_squared_norm_per_site=(residual_energies / total).tolist(),
        dense_fp64_score_parity_passed=True, maximum_score_absolute_error=max_score_error,
        greedy_residual_energy_monotone=True, maximum_positive_energy_change=max_increase,
        decoded_latent_matches_sum_of_physical_codewords=True, gpu_used=False,
        limits='128 teacher-forced validation images. This is not a generation-quality or conditional-predictability measurement.',
        time=time.time())
    write('validation-probe.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'validation_indices'}), flush=True)


if __name__ == '__main__':
    main()
