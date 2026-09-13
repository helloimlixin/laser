import torch
from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ
from src.scaled_atom_rq import ScaledAtomRQ


def test_adaptive_search_matches_explicit_shared_rq_book():
    torch.manual_seed(922)
    dictionary=torch.randn(5,9)
    levels=torch.tensor([-2.,-.5,.5,2.])[None,:]*torch.rand(9,1).add(.2)
    model=AdaptiveScaledAtomRQ(dictionary,levels)
    x=torch.randn(2,2,2,5)
    output=model.quantize(x)
    residual=x.clone()
    book=model.expanded_codebook()
    for d in range(4):
        code=(residual[...,None,:]-book).square().sum(-1).argmin(-1)
        assert torch.equal(code,output['codes'][...,d])
        residual-=book[code]
    torch.testing.assert_close(model.embed(output['codes']).sum(-2),output['quantized'])
    torch.testing.assert_close(x-residual,output['quantized'])


def test_shared_initial_levels_match_existing_quantizer():
    torch.manual_seed(375)
    dictionary=torch.randn(7,12)
    levels=torch.tensor([-2.,-.5,.5,2.])
    shared=ScaledAtomRQ(dictionary,levels)
    adaptive=AdaptiveScaledAtomRQ(dictionary,levels.expand(12,-1))
    x=torch.randn(8,7)
    a=shared.quantize(x)
    b=adaptive.quantize(x)
    assert torch.equal(a['codes'],b['codes'])
    torch.testing.assert_close(a['quantized'],b['quantized'])
