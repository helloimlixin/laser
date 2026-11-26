#!/usr/bin/env python3
"""Compare the experimental pair teacher with actual upstream RQ on CPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.training.residual_pair_targets import residual_pair_targets


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    quantization_path = args.upstream / 'rqvae/models/rqvae/quantizations.py'
    loss_path = args.upstream / 'rqvae/optimizer/loss.py'
    Quantizer = load_module(quantization_path, 'rq_reference_quantizer').RQBottleneck
    loss_module = load_module(loss_path, 'rq_reference_loss')
    torch.manual_seed(20260926)
    x = torch.randn(8, 2, 2, 3, dtype=torch.float64) * .3
    rq = Quantizer((2, 2, 3), (2, 2, 4), 7, shared_codebook=True).double().eval()
    with torch.no_grad():
        rq.codebooks[0].weight[:-1].mul_(.3)
    book = rq.codebooks[0].weight[:-1].detach()
    torch.manual_seed(73)
    q, codes = rq.get_soft_codes(x, temp=.5, stochastic=True)
    torch.manual_seed(73)
    residual = x.clone()
    independent_q, independent_codes = [], []
    for _ in range(4):
        distances = (residual[..., None, :] - book).square().sum(-1)
        prob = (-distances / .5).softmax(-1)
        selected = torch.multinomial(prob.reshape(-1, 7), 1).reshape(8, 2, 2)
        residual -= book[selected]
        independent_q.append(prob)
        independent_codes.append(selected)
    independent_q = torch.stack(independent_q, -2)
    assert torch.equal(codes, torch.stack(independent_codes, -1))
    torch.testing.assert_close(q, independent_q, rtol=1e-12, atol=1e-12)
    logits = torch.randn_like(q, requires_grad=True)
    upstream_loss = loss_module.soft_target_cross_entropy(logits, q)
    reference_loss = -(q * logits.log_softmax(-1)).sum(-1).mean()
    g1, = torch.autograd.grad(upstream_loss, logits, retain_graph=True)
    g2, = torch.autograd.grad(reference_loss, logits)
    # Released log_prob_from_logits adds 1e-7 to its denominator.
    assert abs(float(upstream_loss.detach() - reference_loss.detach())) < 1e-7
    assert float((g1 - g2).abs().max()) < 1e-8

    dictionary = torch.randn(3, 5, dtype=torch.float64) * .3
    grids = torch.tensor([[-1., -.2, .4, 1.], [-.8, -.1, .3, .7],
                          [-.6, -.2, .2, .6]], dtype=torch.float64)
    pair = residual_pair_targets(x, dictionary, grids, temperature=.5,
                                 generator=torch.Generator().manual_seed(31),
                                 site_chunk_size=11, atom_chunk_size=2)
    expanded = Quantizer((2, 2, 3), (2, 2, 3), 20, shared_codebook=False).double().eval()
    with torch.no_grad():
        for d, grid in enumerate(grids):
            expanded.codebooks[d].weight[:-1].copy_(
                (dictionary.T[:, None, :] * grid[None, :, None]).reshape(20, 3))
    residual = x.clone()
    pair_error = 0.
    for d in range(3):
        distances = expanded.codebooks[d].compute_distances(residual)
        joint = (-distances / .5).softmax(-1).reshape(*x.shape[:-1], 5, 4)
        marginal = joint.sum(-1)
        index = pair.atoms[..., d, None, None].expand(*x.shape[:-1], 1, 4)
        conditional = joint.gather(-2, index).squeeze(-2)
        conditional /= conditional.sum(-1, keepdim=True)
        torch.testing.assert_close(pair.atom_probabilities[..., d, :], marginal,
                                   rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(pair.coefficient_probabilities[..., d, :], conditional,
                                   rtol=1e-12, atol=1e-12)
        pair_error = max(pair_error, float((pair.atom_probabilities[..., d, :] - marginal).abs().max()))
        selected = pair.atoms[..., d] * 4 + pair.coefficient_ids[..., d]
        residual -= expanded.codebooks[d].embed(selected)
    torch.testing.assert_close(pair.reconstruction, x - residual)
    result = dict(passed=True, device='cpu', dtype='float64', temperature=.5,
                  rq_sampled_codes_identical=True,
                  rq_soft_target_max_error=float((q - independent_q).abs().max()),
                  rq_loss_abs_error=float((upstream_loss - reference_loss).detach().abs()),
                  rq_loss_gradient_max_error=float((g1 - g2).abs().max()),
                  pair_targets_match_expanded_upstream_dictionary=True,
                  pair_atom_target_max_error=pair_error,
                  production_integration=False,
                  limitations=['Toy dictionaries; no real-image reconstruction or GPU throughput validation.',
                               'Pair vocabulary and residual coder differ from final-refit OMP.',
                               'Factorized draws have the same probability law, not the same RNG sequence as one expanded draw.'],
                  sources={str(p): sha(p) for p in [quantization_path, loss_path,
                      ROOT / 'src/training/residual_pair_targets.py', Path(__file__)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
