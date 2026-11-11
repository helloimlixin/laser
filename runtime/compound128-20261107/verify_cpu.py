"""Verify 128-bin training/cache likelihood parity and deterministic targets."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/tmp/laser-compound128-20261107')
BASE = ROOT / 'frozen'
sys.path[:0] = [str(BASE / 'source'), str(BASE)]
os.environ['RQ_UPSTREAM_ROOT'] = str(BASE / 'upstream')
import torch
from omegaconf import OmegaConf
from src.models.laser_native_compound_adapter import build_native_compound_model
from src.training.laser_hard_ar import nearest_coefficient_ids
from runtime.compound_recipe import Compound128Recipe


def main():
    torch.set_num_threads(2)
    torch.manual_seed(261107)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=8, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=11, vocab_size_cond=1000,
        block_size_cond=1, body=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=2, block=dict(n_head=3, resid_pdrop=0.))))
    model = build_native_compound_model(arch, num_atoms=11, coefficient_bins=128, depth=4).eval()
    model.apply(model._init_weights)
    aux = SimpleNamespace(dictionary=torch.nn.functional.normalize(torch.randn(8, 11), dim=0),
        coeff_bins=torch.linspace(-3, 3, 128), coeff_scales=torch.tensor([4.58794, 2.47499, 1.61067, 1.01760]))
    tokens = torch.empty(2, 2, 2, 8, dtype=torch.long)
    tokens[..., 0::2] = torch.stack([torch.randperm(11)[:4] for _ in range(8)]).reshape(2, 2, 2, 4)
    tokens[..., 1::2] = torch.randint(128, (2, 2, 2, 4)) + 11
    labels = torch.tensor([933, 849])
    with torch.no_grad():
        full = model(tokens, model_aux=aux, cond=labels)
        model.init_cache()
        checked = 0
        for h in range(2):
            for w in range(2):
                for event in range(8):
                    actual = model.cached_forward(tokens, model_aux=aux, cond=labels, sample_loc=(h, w, event))
                    expected = full['atom_logits' if event % 2 == 0 else 'coeff_logits'][:, h, w, event//2]
                    selected = actual[:, :11] if event % 2 == 0 else actual[:, 11:]
                    torch.testing.assert_close(selected, expected, rtol=2e-5, atol=2e-5)
                    checked += 1
        model.init_cache()
    # Nearest-bin coding agrees with exhaustive distances, including clipping.
    values = torch.cat([torch.randn(1000)*3, torch.tensor([-100., 100., 0.])])
    ids = nearest_coefficient_ids(values, aux.coeff_bins)
    assert torch.equal(ids, (values[:, None]-aux.coeff_bins).abs().argmin(-1))
    # The production loss has finite gradients even with duplicate-support masking.
    profile = Compound128Recipe(BASE)
    production_atoms = torch.randn(2, 1, 1, 4, 16384, requires_grad=True)
    production_coeff = torch.randn(2, 1, 1, 4, 128, requires_grad=True)
    targets = torch.empty(2, 1, 1, 8, dtype=torch.long)
    targets[..., 0::2] = torch.arange(4)
    targets[..., 1::2] = torch.randint(128, (2, 1, 1, 4))+16384
    masked = production_atoms.clone()
    for depth in range(1, 4):
        masked[..., depth, :depth] = -torch.inf
    loss = profile.loss(None, dict(atom_logits=masked, coeff_logits=production_coeff), targets)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(production_atoms.grad).all() and torch.isfinite(production_coeff.grad).all()
    report = dict(passed=True, coefficient_bins=128, cached_training_logits_checked=checked,
        nearest_bin_parity=True, endpoint_clipping_verified=True, finite_masked_loss_and_gradients=True)
    (ROOT / 'cpu-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
