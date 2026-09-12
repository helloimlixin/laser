import copy

import pytest
import torch
from torch import nn

from src.learned_sparse_site_codec import LearnedSparseSiteCodec, SparseSiteProjector


def setup_codec():
    torch.manual_seed(13)
    dictionary = torch.eye(8)
    bins = torch.linspace(-3, 3, 127)
    projector = SparseSiteProjector(dictionary, bins, torch.ones(4))
    source = torch.randn(16, 8)
    initial = projector(source)['latents']
    return LearnedSparseSiteCodec(initial, width=16, layers=1), projector


def test_projector_emits_four_distinct_atoms_signed_bins_and_their_actual_sum():
    codec, projector = setup_codec()
    z = torch.tensor([[2., -1.5, .8, -.6, .1, -.05, .02, .01]])
    result = projector(z)
    assert result['atoms'].shape == (1, 4)
    assert torch.unique(result['atoms']).numel() == 4
    coefficients = projector.bins[result['coefficient_ids']] * projector.scales
    expected = (projector.dictionary[result['atoms']] * coefficients[..., None]).sum(-2)
    assert torch.equal(result['latents'], expected)
    assert (coefficients < 0).any() and (coefficients > 0).any()


def test_hard_forward_equals_id_only_decode_and_export_without_continuous_bypass():
    codec, projector = setup_codec()
    source = torch.randn(2, 8, 3, 3)
    output, ids, _, _ = codec(source, projector)
    decoded = codec.decode_ids(ids, projector)['latents'].permute(0, 3, 1, 2)
    assert torch.equal(output, decoded)
    exported = codec.export_codebook(projector, batch_size=5)
    assert torch.equal(decoded, exported['latents'][ids].permute(0, 3, 1, 2))
    repeated = codec.decode_ids(torch.tensor([[[2, 2], [2, 2]]]), projector)['latents']
    assert torch.equal(repeated, repeated[:, :1, :1].expand_as(repeated))
    with pytest.raises(ValueError):
        codec.decode_ids(torch.tensor([16]), projector)


def test_image_gradient_reaches_encoder_and_codewords_through_frozen_decoder():
    codec, projector = setup_codec()
    decoder = nn.Conv2d(8, 3, 1).requires_grad_(False)
    source = torch.randn(2, 8, 3, 3)
    before = decoder.weight.detach().clone()
    out, _, _, _ = codec(source, projector)
    (decoder(out) - torch.ones(2, 3, 3, 3)).square().mean().backward()
    assert codec.codewords.grad is not None and codec.codewords.grad.abs().sum() > 0
    assert codec.encoder[-1].weight.grad.abs().sum() > 0
    assert decoder.weight.grad is None and torch.equal(before, decoder.weight)
    assert not projector.state_dict()  # Static checkpoint assets never enter the optimizer/checkpoint.


def test_optimizer_and_codec_resume_reproduce_the_next_update():
    codec, projector = setup_codec()
    optimizer = torch.optim.AdamW(codec.parameters(), lr=1e-4)
    source = torch.randn(2, 8, 3, 3)
    def step(model, opt):
        opt.zero_grad()
        value, _, penalties, _ = model(source, projector)
        (value.square().mean() + sum(penalties.values())).backward()
        opt.step()
    step(codec, optimizer)
    state, optstate = copy.deepcopy(codec.state_dict()), copy.deepcopy(optimizer.state_dict())
    step(codec, optimizer)
    resumed, _ = setup_codec()
    resumed.load_state_dict(state)
    other = torch.optim.AdamW(resumed.parameters(), lr=1e-4)
    other.load_state_dict(optstate)
    step(resumed, other)
    for a, b in zip(codec.parameters(), resumed.parameters()):
        assert torch.equal(a, b)
