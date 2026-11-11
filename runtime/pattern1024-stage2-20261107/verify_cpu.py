"""Check native compatibility, causality, cache parity and the five-event NLL."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/tmp/laser-pattern1024-stage2-20261107')
BASE = ROOT/'frozen'
sys.path[:0] = [str(BASE/'source'), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE/'upstream')

import torch
from omegaconf import OmegaConf
from src import rq_upstream
from rqvae.models.rqtransformer.transformers import RQTransformer
from rqvae.models.rqtransformer.configs import RQTransformerConfig
from runtime.pattern_model import build_model, pack, unpack, verify_codes
from runtime.pattern_recipe import PatternRecipe


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2611071024)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2,2,4], embed_dim=12,
        input_embed_dim=8, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=11, vocab_size_cond=1000,
        block_size_cond=1, body=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.))))
    dictionary = torch.nn.functional.normalize(torch.randn(8,11), dim=0)
    patterns = torch.randn(1024,4)
    aux = SimpleNamespace(dictionary=dictionary, coefficient_patterns=patterns)
    model = build_model(arch, atoms=11, patterns=1024).eval()
    atoms = torch.stack([torch.randperm(11)[:4] for _ in range(8)]).reshape(2,2,2,4)
    ids = torch.randint(1024,(2,2,2))
    codes = pack(atoms, ids, 1024)
    a, p = unpack(codes, 1024)
    assert torch.equal(a,atoms) and torch.equal(p,ids)
    verify_codes(codes, atoms=11)
    labels = torch.tensor([933,849])
    with torch.no_grad():
        reference = model(codes, model_aux=aux, cond=labels)
        checks = 0
        for h in range(2):
            for w in range(2):
                alternate = ids.clone()
                alternate[:,h,w] = (alternate[:,h,w]+17)%1024
                changed = model(pack(atoms,alternate,1024), model_aux=aux, cond=labels)
                torch.testing.assert_close(changed['atom_logits'][:,h,w], reference['atom_logits'][:,h,w], rtol=0, atol=0)
                torch.testing.assert_close(changed['pattern_logits'][:,h,w], reference['pattern_logits'][:,h,w], rtol=0, atol=0)
                future_atoms = atoms.clone()
                future_atoms[:,h,w,2:] = future_atoms[:,h,w,2:].flip(-1)
                future = model(pack(future_atoms,ids,1024), model_aux=aux, cond=labels)
                torch.testing.assert_close(future['atom_logits'][:,h,w,:3], reference['atom_logits'][:,h,w,:3], rtol=0, atol=0)
                assert not torch.equal(future['pattern_logits'][:,h,w], reference['pattern_logits'][:,h,w])
                checks += 1
        model.init_cache()
        cached_checks = 0
        for h in range(2):
            for w in range(2):
                for depth in range(4):
                    actual = model.cached_forward(codes, model_aux=aux, cond=labels, sample_loc=(h,w,depth))
                    torch.testing.assert_close(actual, reference['atom_logits'][:,h,w,depth], rtol=2e-5, atol=2e-5)
                    cached_checks += 1
                actual = model.cached_pattern_logits(atoms[:,h,w],aux,amp=False)
                torch.testing.assert_close(actual,reference['pattern_logits'][:,h,w],rtol=2e-5,atol=2e-5)
                cached_checks += 1
        generated = model.sample_patterns(2,aux,labels,amp=False)
        verify_codes(generated,atoms=11)
        # Hooks with their defaults must reproduce unpatched native RQ exactly.
        source = Path('/tmp/laser-compound128-20261107/frozen/upstream/rqvae/models/rqtransformer/transformers.py')
        name = 'rqvae.models.rqtransformer.original_compatibility_reference'
        spec = importlib.util.spec_from_file_location(name,source)
        original = importlib.util.module_from_spec(spec)
        sys.modules[name] = original
        spec.loader.exec_module(original)
        config = RQTransformerConfig.create(arch)
        before, after = original.RQTransformer(config).eval(), RQTransformer(config).eval()
        after.load_state_dict(before.state_dict(), strict=True)
        native_aux = SimpleNamespace(get_code_emb_with_depth=lambda tokens:(dictionary.T[tokens],None))
        torch.testing.assert_close(before(atoms,model_aux=native_aux,cond=labels),
            after(atoms,model_aux=native_aux,cond=labels),rtol=0,atol=0)
    # Verify actual model backward, including masked atoms and the conditional head.
    model.train()
    output = model(codes,model_aux=aux,cond=labels)
    atom_ce = torch.nn.functional.cross_entropy(output['atom_logits'].reshape(-1,11),atoms.reshape(-1))
    pattern_ce = torch.nn.functional.cross_entropy(output['pattern_logits'].reshape(-1,1024),ids.reshape(-1))
    loss = (4*atom_ce+pattern_ce)/5
    loss.backward()
    assert torch.isfinite(loss)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    profile = PatternRecipe(BASE)
    full = OmegaConf.load(BASE/'inputs/stage2-training-config.yaml').arch
    from rqvae.utils.config import augment_arch_defaults
    with torch.device('meta'):
        production = profile.model(augment_arch_defaults(full))
    report = dict(passed=True, packing_round_trip=True, causal_sites_checked=checks,
        target_pattern_does_not_leak=True, future_atoms_do_not_leak=True,
        pattern_head_uses_ordered_full_support=True, cached_logits_checked=cached_checks,
        sampled_codes_valid=True, native_default_hooks_bitwise_equal=True,
        finite_backward_all_parameters=True, patterns=1024, native_depth_steps=4,
        parameter_count=sum(v.numel() for v in production.parameters()),
        parameter_tensors=len(list(production.parameters())))
    (ROOT/'cpu-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
