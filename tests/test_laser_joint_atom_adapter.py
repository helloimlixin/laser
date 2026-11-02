import copy

import pytest
import torch

from src.models.laser_joint_atom_adapter import AtomSharedJointLinear, add_atom_shared_joint_classifier
from src.models.laser_joint_execution import AlignedJointLinear


@pytest.mark.parametrize('aligned', [False, True])
def test_joint_logits_and_base_gradients_unchanged_at_initialization(aligned):
    torch.manual_seed(438)
    original = torch.nn.Linear(9, 13)
    if aligned:
        original = AlignedJointLinear(original, multiple=8)
    reference = copy.deepcopy(original)
    adapter = AtomSharedJointLinear(original, 3, 4)
    x = torch.randn(2, 2, 4, 9)
    actual = adapter(x)
    expected = reference(x)
    assert torch.equal(actual, expected)
    upstream = torch.randn_like(actual)
    (actual*upstream).sum().backward()
    (expected*upstream).sum().backward()
    torch.testing.assert_close(adapter.base.weight.grad, reference.weight.grad)
    torch.testing.assert_close(adapter.base.bias.grad, reference.bias.grad)
    grouped = torch.cat((upstream[..., 1:].reshape(2, 2, 4, 3, 4).sum(-1),
                         upstream[..., :1]), -1)
    torch.testing.assert_close(adapter.atom_delta.weight.grad, grouped.flatten(0, -2).T@x.flatten(0, -2))
    torch.testing.assert_close(adapter.atom_delta.bias.grad, grouped.sum((0, 1, 2)))


def test_one_joint_distribution_and_positionwise_execution_match():
    torch.manual_seed(439)
    head = AtomSharedJointLinear(torch.nn.Linear(9, 13), 3, 4)
    torch.nn.init.normal_(head.atom_delta.weight, std=.1)
    torch.nn.init.normal_(head.atom_delta.bias, std=.1)
    x = torch.randn(2, 3, 4, 9)
    full = head(x)
    for h in range(3):
        for d in range(4):
            torch.testing.assert_close(head(x[:, h, d]), full[:, h, d])
    assert full.shape[-1] == 13
    torch.testing.assert_close(full.softmax(-1).sum(-1), torch.ones(2, 3, 4))
    assert torch.isfinite(head.atom_delta.weight).all()


def test_native_full_and_cached_sampler_paths_with_learned_atom_branch():
    from omegaconf import OmegaConf
    from src import rq_upstream
    from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
    torch.manual_seed(440)
    dictionary = torch.nn.functional.normalize(torch.randn(7, 3), dim=0)
    atoms = torch.cat((torch.zeros(1, dtype=torch.long), torch.arange(3).repeat_interleave(4))).expand(4, -1)
    coefficients = torch.cat((torch.zeros(4, 1), torch.randn(4, 12)), -1)
    q = JointAtomCoefficientRQ(dictionary, atoms, coefficients)
    config = OmegaConf.create(dict(type='rq-transformer', block_size=[2, 2, 4], embed_dim=12,
        input_embed_dim=7, shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=13,
        vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.))))
    model = rq_upstream.RQTransformer(rq_upstream.RQTransformerConfig.create(config)).eval()
    add_atom_shared_joint_classifier(model, q)
    torch.nn.init.normal_(model.classifier.linear.atom_delta.weight, std=.1)
    codes = torch.randint(13, (2, 2, 2, 4))
    labels = torch.tensor([0, 999])
    with torch.no_grad():
        expected = model(codes, q, labels)
        prefix = torch.zeros_like(codes)
        model.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    actual = model.cached_forward(prefix[:, :h+1], q, labels, sample_loc=(h, w, d))
                    torch.testing.assert_close(actual, expected[:, h, w, d], rtol=1.e-5, atol=1.e-6)
                    prefix[:, h, w, d] = codes[:, h, w, d]
