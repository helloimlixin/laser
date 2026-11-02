#!/usr/bin/env python3
"""Count actual atom/bin pairs after coarsening the original coefficient grid."""
import json
from pathlib import Path
import sys
import time

ROOT = Path('/workspace/Projects/laser')
sys.path.insert(0, str(ROOT))
from src import rq_upstream
import numpy as np
import torch
from src.training.laser_hard_ar import sparse_encoding, nearest_coefficient_ids
from src.orthogonal_sparse_codec import dictionary_to_orthogonal_coefficients, ordered_support_basis
from src.training.rq_imagenet_runtime import write_json


@torch.inference_mode()
def main():
    torch.set_num_threads(4); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.13, 0)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path('/tmp/laser-rq-joint-vocabulary-study-20261015')
    out = ROOT/'outputs/imagenet-laser-native-joint-scratch6-4h100-20261015/performance'
    spec = torch.load(path/'dictionary.pt', map_location='cpu', weights_only=True)
    dictionary = spec.cuda(); gram = dictionary.T@dictionary
    latents = np.load(path/'fit-latents.npy', mmap_mode='r')
    scales = torch.tensor([8.203365325927734, 4.265638828277588,
                            3.0662174224853516, 1.8273425102233887], device='cuda')
    sizes = [512, 1024, 2048]
    counts = {b: np.zeros((4, 16384*b), dtype=np.uint32) for b in sizes}
    errors = {b: 0. for b in sizes}
    quantization_error = {b: 0. for b in sizes}
    energy = 0.; source_error = 0.
    for first in range(0, len(latents), 8):
        x = torch.from_numpy(latents[first:first+8].copy()).cuda()
        atoms, coefficients = sparse_encoding(x, dictionary, 4, gram=gram)
        vectors = dictionary.T[atoms]
        gamma, _ = dictionary_to_orthogonal_coefficients(vectors, coefficients)
        basis, _ = ordered_support_basis(vectors)
        original = (basis*gamma[..., None]).sum(-2)
        source_error += (x-original).square().sum().item()
        energy += x.square().sum().item()
        for b in sizes:
            grid = torch.linspace(-3, 3, b, device='cuda')
            ids = nearest_coefficient_ids(gamma/scales, grid)
            quantized = grid[ids]*scales
            reconstructed = (basis*quantized[..., None]).sum(-2)
            errors[b] += (x-reconstructed).square().sum().item()
            quantization_error[b] += (gamma-quantized).square().sum().item()
            for d in range(4):
                pair = (atoms[..., d]*b+ids[..., d]).flatten().cpu().numpy()
                unique, frequency = np.unique(pair, return_counts=True)
                counts[b][d, unique] += frequency.astype(np.uint32)
        if first%256 == 0:
            print(json.dumps(dict(images=first+len(x), total=len(latents), time=time.time())), flush=True)
    results = {}
    for b in sizes:
        union = counts[b].sum(0)
        results[str(b)] = dict(full_cartesian_classes=16384*b,
            occupied_joint_ids_in_8192_images=int(np.count_nonzero(union)),
            occupied_per_depth=np.count_nonzero(counts[b], axis=1).tolist(),
            counts_covered_by_top65536=float(np.sort(union)[-65536:].sum()/union.sum()),
            latent_mse=errors[b]/latents.size,
            coefficient_quantization_mse=quantization_error[b]/(len(latents)*64*4),
            linear_classifier_parameters=16384*b*(1536+1))
    write_json(out/'coefficient-bin-size-study.json', dict(images=len(latents),
        observations=len(latents)*64*4, original_coefficient_bins=2048,
        coordinates='orthogonal gamma with original depth scales',
        original_continuous_latent_mse=source_error/latents.size, results=results,
        observed_vocabulary_is_lower_bound_for_full_training_set=True,
        unseen_pairs_require_a_defined_encoding_if_using_observed_pairs_only=True,
        current_native_joint_vocabulary=65537,
        current_native_joint_levels_per_atom=4,time=time.time()))
    print(json.dumps(results), flush=True)


if __name__ == '__main__': main()
