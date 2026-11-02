from types import MethodType, SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from src import rq_upstream
from src.models.laser_fine_joint_adapter import (ConditionalJointLinear,
    install_conditional_joint_head, exclude_previous_atoms, joint_cross_entropy)
from src.models.laser_fine_joint_tokenizer import FrozenFineJointTokenizer


def test_joint_nll_and_gradients_equal_explicit_cartesian_distribution():
    torch.manual_seed(762)
    head = ConditionalJointLinear(nn.Linear(7, 5), 3, 4).double()
    hidden = torch.randn(2, 2, 7, dtype=torch.float64, requires_grad=True)
    features = head(hidden)
    atoms = torch.tensor([[0, 2], [4, 1]])
    p = torch.rand(2, 2, 3, dtype=torch.float64).softmax(-1)
    actual = joint_cross_entropy(head, features, dict(atoms=atoms, coefficient_probabilities=p))
    atom_logp = exclude_previous_atoms(features[..., :5], atoms).log_softmax(-1)
    all_atoms = torch.arange(5).expand(2, 2, 5)
    coefficient_logp = head.coefficient_logits(features.unsqueeze(-2).expand(2, 2, 5, 9),
                                               all_atoms).log_softmax(-1)
    joint_logp = atom_logp[..., None] + coefficient_logp
    torch.testing.assert_close(joint_logp.exp().sum((-1, -2)), torch.ones(2, 2, dtype=torch.float64))
    selected = joint_logp.gather(-2, atoms[..., None, None].expand(2, 2, 1, 3)).squeeze(-2)
    expected = -(p * selected).sum(-1).mean()
    # Production computes CE in FP32, including under mixed precision.
    torch.testing.assert_close(actual.double(), expected, rtol=2e-7, atol=2e-7)
    ga = torch.autograd.grad(actual, tuple(head.parameters())+(hidden,), retain_graph=True)
    gb = torch.autograd.grad(expected, tuple(head.parameters())+(hidden,))
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-7)


def tiny_case():
    torch.manual_seed(251)
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4],
        embed_dim=12, input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=9,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(arch)).eval()
    install_conditional_joint_head(model, 5, 6)
    aux = SimpleNamespace(dictionary=F.normalize(torch.randn(7, 9), dim=0),
        coeff_vocab_size=5, coeff_bins=torch.linspace(-3, 3, 5),
        coeff_scales=torch.tensor([3., 2., 1., .5]))
    aux.get_code_emb_with_depth = MethodType(FrozenFineJointTokenizer.get_code_emb_with_depth, aux)
    atoms = torch.tensor([0, 3, 5, 8]).expand(2, 2, 2, 4)
    codes = atoms * 5 + torch.randint(5, atoms.shape)
    return model, aux, codes, torch.tensor([0, 999])


@torch.no_grad()
def test_packed_ids_use_native_cache_and_preserve_full_prefix_causality():
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
    changed = codes.clone(); changed[:, 0, 0, 2:] = torch.tensor([6*5+4, 7*5])
    altered = model(changed, aux, labels)
    torch.testing.assert_close(altered[:, 0, 0, :3], expected[:, 0, 0, :3], rtol=0, atol=0)
    assert not torch.allclose(altered[:, 0, 0, 3], expected[:, 0, 0, 3])
    changed = codes.clone(); changed[:, 0, 0, 0] = 4-codes[:, 0, 0, 0]%5
    altered = model(changed, aux, labels)
    torch.testing.assert_close(altered[:, 0, 0, 0], expected[:, 0, 0, 0], rtol=0, atol=0)
    assert not torch.allclose(altered[:, 0, 1], expected[:, 0, 1])


def test_all_conditional_head_parameters_receive_joint_loss_gradients():
    model, aux, codes, labels = tiny_case()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    p = F.one_hot(codes % 5, 5).float()
    loss = joint_cross_entropy(model.classifier.linear, model(codes, aux, labels),
                              dict(atoms=codes//5, coefficient_probabilities=p))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert len(optimizer.state) == len(list(model.parameters()))


def test_exclusion_uses_only_past_support():
    atoms = torch.tensor([[3, 1, 4, 0]])
    logits = exclude_previous_atoms(torch.zeros(1, 4, 6), atoms)
    for d in range(4):
        assert torch.isneginf(logits[0, d]).sum() == d
        assert torch.isfinite(logits[0, d, atoms[0, d:]]).all()
