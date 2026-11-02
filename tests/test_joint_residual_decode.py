import pytest
import torch

from src.training.joint_residual_decode import decode_residuals


class Book:
    depth = 3

    def embed(self, codes):
        # Each depth contributes to a different physical coordinate.
        return torch.nn.functional.one_hot(torch.arange(3),3).float()*codes[...,None]


class Tokenizer:
    quantizer = Book()

    def decode(self, latent):
        return latent

    def decode_code(self,codes):
        return self.decode(self.quantizer.embed(codes).sum(-2))


def test_weights_apply_to_residual_depth_without_changing_codes():
    codes=torch.tensor([[[[2,4,6],[3,5,7]]]])
    snapshot=codes.clone()
    actual=decode_residuals(Tokenizer(),codes,[1,.5,0])
    torch.testing.assert_close(actual,torch.tensor([[[[2.,2.,0.],[3.,2.5,0.]]]]))
    assert torch.equal(codes,snapshot)


def test_identity_decoding_is_exact():
    codes=torch.randint(10,(2,3,4,3))
    assert torch.equal(decode_residuals(Tokenizer(),codes,[1,1,1]),Tokenizer().decode_code(codes))


@pytest.mark.parametrize('weights',[[1,1],[1,-1,1],[1,float('nan'),1]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        decode_residuals(Tokenizer(),torch.zeros(1,1,1,3,dtype=torch.long),weights)
