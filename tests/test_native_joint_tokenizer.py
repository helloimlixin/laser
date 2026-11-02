"""Joint product IDs must keep the recovered beam and native teacher contracts."""
import pytest
import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ


def test_nonzero_product_vocabulary_preserves_id_zero_and_beam_anchor():
    dictionary = torch.tensor([[1., .6], [0., .8]])
    atoms = torch.tensor([[0, 0, 1, 1]]).expand(2, -1)
    coefficients = torch.tensor([[-2., 2., -1., 1.]]).expand(2, -1)
    book = JointAtomCoefficientRQ(dictionary, atoms, coefficients,
        zero_code_id=None, encoding_beam_width=4)
    assert torch.equal(book.embed_depth(torch.tensor(0), 0), torch.tensor([-2., 0.]))
    signals = torch.randn(2, 3, 2, generator=torch.Generator().manual_seed(7))
    greedy = book._quantize_greedy(signals)
    beam = book.quantize(signals)
    torch.testing.assert_close(beam['quantized'], book.embed(beam['codes']).sum(-2), rtol=0, atol=0)
    assert ((signals-beam['quantized']).square().sum(-1)
            <= (signals-greedy['quantized']).square().sum(-1)+1e-6).all()
    assert beam['codes'].min() >= 0 and beam['codes'].max() < 4


def test_optional_zero_policy_rejects_inconsistent_codewords():
    with pytest.raises(ValueError, match='nonzero'):
        JointAtomCoefficientRQ(torch.eye(2), torch.tensor([[0, 1]]),
            torch.tensor([[0., 1.]]), zero_code_id=None)
