#!/usr/bin/env python3
"""Verify source restoration, gated cache causality, and the established sampler."""
import codecs
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_official_rqtransformer_laser_stage2 import LaserAux
from scripts.train_church_bar_coefficients import make_model
from scripts.tools.build_sign_probe_cache import sha256_file
from src.church_epoch50_loop import epoch50_prior, cached_splits


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    torch.serialization.add_safe_globals([codecs.encode])
    torch.backends.cuda.matmul.allow_tf32 = True
    base = ROOT/'outputs/church-epoch50-loop-20260911'
    checkpoint = ROOT/'outputs/lsun-church-bar-20260910/assets/epoch50-source.pt'
    stage1 = ROOT/'outputs/lsun-church-sign-pattern-20260910/assets/best_rfid_slot1_model.pt'
    raw = torch.load(ROOT/'outputs/lsun-church-bar-20260910/church-cache.pt', map_location='cpu', weights_only=False)
    packed = cached_splits(raw)['validation'][:2].cuda()
    aux = LaserAux(stage1, 16384, 2048, 20., coeff_scales=raw['meta']['coeff_scales'], sparsity_level=4).cuda().eval()
    original, _ = epoch50_prior(checkpoint, 'control')
    looped, _ = epoch50_prior(checkpoint, 'looped')
    original.cuda().eval(); looped.cuda().eval()
    for key, value in original.state_dict().items():
        assert torch.equal(value, looped.state_dict()[key]), key
    a, b = [m(packed, model_aux=aux, amp=False) for m in (original, looped)]
    assert all(torch.equal(a[key], b[key]) for key in a)
    del a, b
    atoms, _ = looped.unpack(packed)
    looped.head_transformer.loop_gates.copy_(torch.tensor([.1, -.05], device='cuda'))
    # Isolate cache correctness in full FP32, then report source-sampler error.
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = looped(packed, model_aux=aux, amp=False)
    looped.init_cache()
    max_errors = {'atom': 0., 'coefficient': 0.}
    for h in range(8):
        for w in range(8):
            for d in range(4):
                hidden = looped.cached_head_output(packed, aux, None, (h, w, d), amp=False)
                logits = looped.classifier(hidden)
                if d: logits.scatter_(1, atoms[:, h, w, :d], -float('inf'))
                coeff = looped.coefficient_logits(hidden, aux.dictionary.t()[atoms[:, h, w, d]], d)
                for key, actual, expected in [
                    ('atom', logits, teacher['atom_logits'][:, h, w, d]),
                    ('coefficient', coeff, teacher['coeff_logits'][:, h, w, d])]:
                    assert torch.equal(torch.isfinite(actual), torch.isfinite(expected))
                    mask = torch.isfinite(actual)
                    error = float((actual[mask]-expected[mask]).abs().max())
                    max_errors[key] = max(max_errors[key], error)
    assert max(max_errors.values()) < 2e-4, max_errors
    del teacher, original
    looped.head_transformer.loop_gates.zero_()
    looped.init_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    reference, _ = make_model(checkpoint, 'categorical')
    reference.cuda().eval()
    torch.manual_seed(17701)
    legacy_atoms, legacy_ids = reference.generate(aux, 128)
    torch.manual_seed(17701)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        new_atoms, new_ids = looped.sample_compound(128, aux, atom_top_k=250,
            atom_top_p=1., coeff_top_k=0, coeff_top_p=None, amp=False)
    assert torch.equal(legacy_atoms, new_atoms) and torch.equal(legacy_ids, new_ids)
    archived = torch.load(ROOT/'outputs/lsun-church-neural-repair-20260910/independent-50000/generated-codes.pt', map_location='cpu', weights_only=True)
    archive_matches = torch.equal(new_atoms.cpu(), archived['atoms'][:128].long()) and torch.equal(new_ids.cpu(), archived['coefficient_ids'][:128].long())
    result = {'passed': True, 'all_source_tensors_equal': True, 'closed_gate_teacher_logits_bitwise_equal': True,
        'active_gate_fp32_cache_max_errors': max_errors, 'cache_images': 2, 'cache_sites_per_image': 64,
        'generated_images_checked': 128, 'matches_legacy_sampler_bitwise': True,
        'matches_archived_fid50k_first128_codes': archive_matches,
        'source_sha256': sha256_file(checkpoint), 'stage1_sha256': sha256_file(stage1)}
    (base/'source-restoration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
