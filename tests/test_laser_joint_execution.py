import copy

import pytest
import torch
from torch import nn

from src.models.laser_joint_execution import AlignedJointLinear, align_joint_classifier


@pytest.mark.parametrize('vocabulary', [11, 64, 65])
@pytest.mark.parametrize('bias', [True, False])
def test_alignment_preserves_native_logits_loss_gradients_and_optimizer_references(vocabulary, bias):
    torch.manual_seed(615)
    native = nn.Linear(12, vocabulary, bias=bias).double()
    reference = copy.deepcopy(native)
    optimizer = torch.optim.AdamW(native.parameters(), lr=.0005)
    aligned = AlignedJointLinear(native)
    assert list(optimizer.param_groups[0]['params'])[0] is aligned.weight is native.weight
    assert aligned.state_dict().keys() == native.state_dict().keys()
    inputs = torch.randn(7, 12, dtype=torch.float64, requires_grad=True)
    other = inputs.detach().clone().requires_grad_()
    labels = torch.randint(vocabulary, (7,))
    a, b = aligned(inputs), reference(other)
    assert a.shape == (7, vocabulary)
    torch.testing.assert_close(a, b, rtol=1.e-14, atol=1.e-14)
    torch.nn.functional.cross_entropy(a, labels).backward()
    torch.nn.functional.cross_entropy(b, labels).backward()
    torch.testing.assert_close(inputs.grad, other.grad, rtol=1.e-13, atol=1.e-14)
    for p, q in zip(aligned.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1.e-13, atol=1.e-14)
    # The original upstream classifier can load this checkpoint strictly.
    native.load_state_dict(aligned.state_dict(), strict=True)
    optimizer.step()
    assert int(optimizer.state[aligned.weight]['step']) == 1


def test_native_joint_transformer_retains_state_keys_and_cached_logits_after_alignment():
    from src import rq_upstream
    from omegaconf import OmegaConf
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    torch.manual_seed(6115)
    q = JointAtomCoefficientRQ(torch.randn(7, 3), torch.tensor([[0, 0, 1, 1, 2]]*4),
                               torch.tensor([[0., -1., 1., -2., 2.]]*4))
    arch = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=5,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(arch)).eval()
    original = copy.deepcopy(model)
    parameters = list(model.parameters())
    align_joint_classifier(model)
    assert type(model) is rq_upstream.RQTransformer
    assert all(p is q for p, q in zip(model.parameters(), parameters))
    assert model.state_dict().keys() == original.state_dict().keys()
    tokens = torch.randint(5, (2, 2, 2, 4)); labels = torch.tensor([0, 999])
    with torch.no_grad():
        expected = original(tokens, q, labels)
        torch.testing.assert_close(model(tokens, q, labels), expected, rtol=1.e-5, atol=1.e-6)
        model.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    actual = model.cached_forward(tokens[:, :h+1], q, labels, sample_loc=(h, w, d))
                    torch.testing.assert_close(actual, expected[:, h, w, d], rtol=1.e-5, atol=1.e-6)
