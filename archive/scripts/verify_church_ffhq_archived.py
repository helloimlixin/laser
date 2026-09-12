#!/usr/bin/env python3
"""Exercise geometry gradients and every cached position of the actual Church model."""
import argparse
import codecs
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.church_ffhq_archived import ChurchAux, make_prior, objective, targets, full_training_cache
from archive.scripts.train_church_ffhq_archived import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.serialization.add_safe_globals([codecs.encode])
    raw = torch.load(ROOT/'outputs/church-ffhq-recipe-20260911/continuous-cache.pt', weights_only=False)
    data, scales = full_training_cache(raw)
    aux = ChurchAux(ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt',
        16384, 2048, 3., coeff_scales=scales, sparsity_level=4).cuda().eval().requires_grad_(False)
    model = make_prior('looped').cuda()
    a = data['train']['atoms'][:32].cuda().long()
    c = data['train']['coefficients'][:32].cuda()
    model.train()
    loss, metrics = objective(model, aux, a, c, .05)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert torch.isfinite(norm)
    assert model.head_transformer.loop_gates.grad.abs().sum() > 0
    assert all(v.grad is None for v in aux.parameters())
    result = {'geometry_gradient_norm': float(norm), 'geometry_training': metrics,
              'gate_gradients': model.head_transformer.loop_gates.grad.tolist()}
    model.zero_grad(set_to_none=True)
    del loss, a, c
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    model.head_transformer.loop_gates.data.copy_(torch.tensor([.12, -.08], device='cuda'))
    a = data['validation']['atoms'][:2].cuda().long()
    c = data['validation']['coefficients'][:2].cuda()
    packed, _ = targets(aux, a, c, stochastic=False)
    maximum = {'atom': 0., 'coefficient': 0.}
    with torch.no_grad():
        # TF32 uses different GEMM shapes in parallel teacher forcing and
        # single-position decoding. Measure that rounding separately from the
        # strict FP32 test of cache semantics.
        production_teacher = model(packed, model_aux=aux, amp=False)
        model.init_cache()
        hidden = model.cached_head_output(packed, aux, None, (0, 0, 0), amp=False)
        result['tf32_first_position_atom_max_error'] = float((
            model.classifier(hidden) - production_teacher['atom_logits'][:, 0, 0, 0]).abs().max())
        del production_teacher
        torch.backends.cuda.matmul.allow_tf32 = False
        teacher = model(packed, model_aux=aux, amp=False)
        model.init_cache()
        for h in range(8):
            for w in range(8):
                for d in range(4):
                    hidden = model.cached_head_output(packed, aux, None, (h, w, d), amp=False)
                    atom = model.classifier(hidden)
                    coefficient = model.coefficient_logits(hidden, aux.dictionary.t()[a[:, h, w, d]], d)
                    for key, value, target in [('atom', atom, teacher['atom_logits'][:, h, w, d]),
                            ('coefficient', coefficient, teacher['coeff_logits'][:, h, w, d])]:
                        maximum[key] = max(maximum[key], float((value-target).abs().max()))
                        torch.testing.assert_close(value, target, atol=1e-4, rtol=1e-4)
            print(json.dumps({'cache_rows_checked': h+1, 'max_errors': maximum}), flush=True)
    result.update(cache_images=2, cache_sites_per_image=64, active_loop_fp32_cache_max_errors=maximum)
    del teacher, packed
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    args.fid_stats = ROOT/'outputs/lsun-church-bar-20260910/assets/lsun_256_church.npz'
    args.generation_batch = 32
    result['sampler_smoke'] = generate(model, aux, args, args.output/'smoke', 64, 3141,
        lambda row: print(json.dumps(row), flush=True))
    result['quality_result'] = False
    result['passed'] = True
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
