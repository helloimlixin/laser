from types import SimpleNamespace, MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from src import rq_upstream
from src.models.laser_fine_joint_tokenizer import FrozenFineJointTokenizer
from src.models.laser_signed_joint_adapter import (SignedJointLinear,
    split_coefficient_bins, merge_coefficient_bins, install_signed_joint_head,
    exclude_previous_signed_atoms, signed_joint_cross_entropy)


@pytest.mark.parametrize('bins', [2, 6, 512, 1024])
def test_sign_magnitude_mapping_preserves_every_signed_coefficient_and_packed_id(bins):
    coefficients = torch.arange(bins).expand(3, bins)
    atoms = torch.tensor([0, 1, 16383])[:, None].expand_as(coefficients)
    signs, magnitude = split_coefficient_bins(coefficients, bins)
    restored = merge_coefficient_bins(signs, magnitude, bins)
    torch.testing.assert_close(restored, coefficients, rtol=0, atol=0)
    torch.testing.assert_close(atoms*bins+restored, atoms*bins+coefficients, rtol=0, atol=0)
    grid = torch.linspace(-3, 3, bins)
    unsigned = grid[bins//2:]
    torch.testing.assert_close(unsigned[magnitude]*(2*signs-1), grid[coefficients], rtol=1e-6, atol=3e-7)


def test_soft_joint_likelihood_and_gradients_equal_explicit_atom_coefficient_distribution():
    torch.manual_seed(951)
    head = SignedJointLinear(7, 5, 6, 4).double()
    hidden = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    atoms = torch.tensor([[0, 2, 4], [1, 4, 3]])
    probabilities = torch.randn(2, 3, 6, dtype=torch.float64).softmax(-1)
    targets = dict(atoms=atoms, coefficient_probabilities=probabilities, hard_codes=atoms*6+2)
    features = head(hidden)
    actual, diagnostics = signed_joint_cross_entropy(head, features, targets)
    main = exclude_previous_signed_atoms(features[..., :10], atoms).log_softmax(-1)
    signed_ids = torch.arange(10).expand(2, 3, 10)
    magnitude = head.magnitude_logits(features.unsqueeze(-2), signed_ids).log_softmax(-1)
    joint = (main[..., None]+magnitude).reshape(2, 3, 5, 2, 3)
    joint = torch.cat((joint[..., 0, :].flip(-1), joint[..., 1, :]), -1)
    torch.testing.assert_close(joint.exp().sum((-1, -2)), torch.ones(2, 3, dtype=torch.float64))
    selected = joint.gather(-2, atoms[..., None, None].expand(2, 3, 1, 6)).squeeze(-2)
    expected = -(probabilities*selected).sum(-1).mean()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-7, atol=3e-7)
    torch.testing.assert_close(diagnostics['atom_ce']+diagnostics['sign_ce']+diagnostics['magnitude_ce'], actual.detach())
    torch.testing.assert_close(diagnostics['joint_ce_per_scalar_event'], actual.detach()/2)
    ga = torch.autograd.grad(actual, tuple(head.parameters())+(hidden,), retain_graph=True)
    gb = torch.autograd.grad(expected, tuple(head.parameters())+(hidden,))
    for a, b in zip(ga, gb): torch.testing.assert_close(a, b, rtol=3e-6, atol=2e-7)


def tiny_case():
    torch.manual_seed(953)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=9,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(arch)).eval()
    install_signed_joint_head(model, 9, 6, 6)
    aux = SimpleNamespace(dictionary=F.normalize(torch.randn(7, 9), dim=0),
        coeff_vocab_size=6, coeff_bins=torch.linspace(-3, 3, 6),
        coeff_scales=torch.tensor([3., 2., 1., .5]))
    aux.get_code_emb_with_depth = MethodType(FrozenFineJointTokenizer.get_code_emb_with_depth, aux)
    atoms = torch.tensor([0, 3, 5, 8]).expand(2, 2, 2, 4)
    codes = atoms*6+torch.randint(6, atoms.shape)
    return model, aux, codes, torch.tensor([0, 999])


@torch.no_grad()
def test_native_cache_matches_full_forward_and_future_pairs_are_causal():
    model, aux, codes, labels = tiny_case()
    assert type(model) is rq_upstream.RQTransformer
    expected = model(codes, aux, labels)
    prefix = torch.zeros_like(codes); model.init_cache()
    for h in range(2):
        for w in range(2):
            for d in range(4):
                actual = model.cached_forward(prefix[:, :h+1], aux, labels, sample_loc=(h, w, d))
                torch.testing.assert_close(actual, expected[:, h, w, d], rtol=1e-5, atol=2e-6)
                prefix[:, h, w, d] = codes[:, h, w, d]
    changed = codes.clone(); changed[:, 0, 0, 2:] = torch.tensor([6*6+5, 7*6])
    torch.testing.assert_close(model(changed, aux, labels)[:, 0, 0, :3], expected[:, 0, 0, :3], rtol=0, atol=0)
    p = (codes[:, 0, 0, 0]%6)
    changed = codes.clone(); changed[:, 0, 0, 0] = 5-p
    altered = model(changed, aux, labels)
    torch.testing.assert_close(altered[:, 0, 0, 0], expected[:, 0, 0, 0], rtol=0, atol=0)
    assert not torch.allclose(altered[:, 0, 1], expected[:, 0, 1])


def test_support_exclusion_masks_both_signs_and_no_future_atoms():
    atoms = torch.tensor([[3, 1, 4, 0]])
    logits = exclude_previous_signed_atoms(torch.zeros(1, 4, 12), atoms)
    for depth in range(4):
        assert torch.isneginf(logits[0, depth]).sum() == 2*depth
        for atom in atoms[0, depth:]: assert torch.isfinite(logits[0, depth, 2*atom:2*atom+2]).all()


def test_native_backward_updates_every_parameter_including_both_soft_sign_branches():
    model, aux, codes, labels = tiny_case(); model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    probabilities = torch.randn(*codes.shape, 6).softmax(-1)
    loss, _ = signed_joint_cross_entropy(model.classifier.linear, model(codes, aux, labels),
        dict(atoms=codes//6, hard_codes=codes, coefficient_probabilities=probabilities))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert len(optimizer.state) == len(list(model.parameters()))


def test_main_classifier_can_learn_atom_dependent_sign_without_magnitude_head():
    torch.manual_seed(954)
    head = SignedJointLinear(8, 4, 6, 8)
    prototype = torch.eye(8)[:4]
    signed_classes = torch.arange(8).repeat(8)
    atoms = signed_classes//2; signs = signed_classes%2
    hidden = prototype[atoms]*(2*signs[:, None]-1)+.02*torch.randn(64, 8)
    optimizer = torch.optim.AdamW(head.signed_atom.parameters(), lr=.08, weight_decay=0.)
    for _ in range(60):
        optimizer.zero_grad()
        F.cross_entropy(head.signed_atom(hidden), signed_classes).backward(); optimizer.step()
    logits = head.signed_atom(hidden).reshape(64, 4, 2)
    selected = logits[torch.arange(64), atoms].softmax(-1)
    assert selected.gather(-1, signs[:, None]).mean() > .98
    assert (logits.reshape(64, 8).argmax(-1)==signed_classes).float().mean() > .98


def test_chain_rule_sampler_has_the_same_atom_sign_law_as_joint_classifier():
    torch.manual_seed(955)
    logits = torch.randn(3, 12, dtype=torch.float64)/.85
    joint = logits.softmax(-1).reshape(3, 6, 2)
    shaped = logits.reshape(3, 6, 2)
    marginal = shaped.logsumexp(-1).softmax(-1)
    conditional = shaped.softmax(-1)
    torch.testing.assert_close(marginal[..., None]*conditional, joint)
