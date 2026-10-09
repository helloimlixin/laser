"""Replay first trial image batches under cuDNN encoder policies locally."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.base/'source'), str(args.base/'source/runtime')]
    import torch
    from src.training.rqtransformer import LaserAux, image_transform
    from src.training.fresh_images import EpochImageFolder
    from src.training.exact_global_batch import ExactGlobalBatchSampler

    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(8*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    scales = [8.203365325927734, 4.265638828277588, 3.0662174224853516, 1.8273425102233887]
    aux = LaserAux(args.base/'inputs/stage1-tokenizer.pt', 16384, 2048, 3.,
        coeff_scales=scales, soft_target_physical=False, clamp_coeffs=False,
        sparsity_level=4).cuda().eval()
    dataset = EpochImageFolder(args.base/'imagenet/train', transform=image_transform(),
        augmentation_seed=261001)
    def digest(t):
        return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()
    rows = []
    with torch.inference_mode():
        gram = aux.dictionary.T @ aux.dictionary
        for rank in (1, 2):
            sampler = ExactGlobalBatchSampler(dataset, 2048, 4, rank, accumulation=4, seed=261001)
            sampler.set_epoch(8)
            indices = next(iter(sampler))
            images = torch.stack([dataset[i][0] for i in indices]).cuda()
            recorded = {b: json.loads((args.trial/b/f'verification/production/matched-first-batch-rank{rank}.json').read_text())
                        for b in ('low25', 'control200')}
            assert all(digest(images) == v['images_sha256'] for v in recorded.values())
            results = []
            for benchmark, deterministic in ((True, False), (False, True), (False, True)):
                torch.backends.cudnn.benchmark = benchmark
                torch.backends.cudnn.deterministic = deterministic
                atoms, coefficients = [], []
                for chunk in images.split(32):
                    a, c = aux.encode_sparse_components(chunk, dictionary_gram=gram)
                    atoms.append(a.cpu()); coefficients.append(c.cpu())
                a, c = torch.cat(atoms), torch.cat(coefficients)
                results.append((a, c))
                print(json.dumps(dict(rank=rank, benchmark=benchmark, deterministic=deterministic,
                    atom_sha256=digest(a), coefficient_sha256=digest(c))), flush=True)
            a, c = results[0]
            deterministic_a, deterministic_c = results[1]
            repeated_a, repeated_c = results[2]
            row = dict(rank=rank, input_matching_verified=True,
                benchmark_atom_matching={b: digest(a) == v['atoms_sha256'] for b, v in recorded.items()},
                benchmark_coeff_matching={b: digest(c) == v['clean_coefficients_sha256'] for b, v in recorded.items()},
                deterministic_atom_matching={b: digest(deterministic_a) == v['atoms_sha256'] for b, v in recorded.items()},
                deterministic_coeff_matching={b: digest(deterministic_c) == v['clean_coefficients_sha256'] for b, v in recorded.items()},
                changed_atom_fraction_between_policies=(a != deterministic_a).float().mean().item(),
                normalized_coefficient_mae_between_policies=(c-deterministic_c).abs().mean().item(),
                normalized_coefficient_max_error_between_policies=(c-deterministic_c).abs().max().item(),
                deterministic_repeat_atoms_exact=torch.equal(deterministic_a, repeated_a),
                deterministic_repeat_coefficients_exact=torch.equal(deterministic_c, repeated_c))
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(rows=rows, training_modified=False, wandb_metrics_logged=False,
        limitations='Replay on GPU0; cuDNN benchmark choice is workload dependent. '
            'Without stored original token tensors, policy differences do not measure the exact '
            'fraction of differing targets between the two live training branches.')
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
