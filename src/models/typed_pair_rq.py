"""Two stored vocabularies for one implicit atom/coefficient pair codebook."""
import torch
from torch import nn

class TypedPairCodebook(nn.Module):
    """Pair (a,b) contributes c[b]D[a], with no Cartesian embedding table.

    Alternating IDs are only serialization. The RQ prior consumes one composed
    embedding per completed pair, and advances its depth state once per pair.
    Coefficient bins are shared physical values at every residual depth.
    """
    def __init__(self, dictionary, coefficient_bins, depth=4):
        super().__init__()
        dictionary = torch.as_tensor(dictionary).detach().float().clone()
        bins = torch.as_tensor(coefficient_bins).detach().float().clone()
        if dictionary.ndim != 2 or bins.ndim != 1 or len(bins)<2 or depth<1:
            raise ValueError('expected [channels, atoms], [bins], and positive depth')
        if not torch.isfinite(dictionary).all() or not torch.isfinite(bins).all():
            raise ValueError('codebook must be finite')
        if (dictionary.square().sum(0)<=0).any() or not (bins.diff()>0).all():
            raise ValueError('atoms must be nonzero and bins strictly increasing')
        self.register_buffer('dictionary', dictionary)
        self.register_buffer('coeff_bins', bins)
        self.register_buffer('coeff_scales', torch.ones(depth))
        self.num_atoms, self.coeff_vocab_size = dictionary.shape[1], len(bins)
        self.depth = int(depth)
        self.sparsity_level = self.depth
        self.vocab_size = self.num_atoms+self.coeff_vocab_size
        self.code_shape = torch.Size([8, 8, 2*self.depth])

    def fields(self, tokens):
        if tokens.dtype != torch.long or tokens.shape[-1] != 2*self.depth:
            raise ValueError('expected int64 alternating atom/coefficient tokens')
        atoms, coefficients = tokens[..., 0::2], tokens[..., 1::2]-self.num_atoms
        if ((atoms<0).any() or (atoms>=self.num_atoms).any() or
                (coefficients<0).any() or (coefficients>=self.coeff_vocab_size).any()):
            raise ValueError('token outside its disjoint half-open field range')
        return atoms, coefficients

    def pack(self, atoms, coefficients):
        if atoms.shape != coefficients.shape or atoms.dtype != torch.long or coefficients.dtype != torch.long:
            raise ValueError('pair fields must be equal-shaped int64 tensors')
        tokens = torch.stack((atoms, coefficients+self.num_atoms), -1).flatten(-2)
        self.fields(tokens)
        return tokens

    def empty_tokens(self, batch, height=8, width=8):
        tokens = torch.zeros(batch, height, width, 2*self.depth,
                             dtype=torch.long, device=self.dictionary.device)
        tokens[..., 1::2] = self.num_atoms+int(self.coeff_bins.abs().argmin())
        return tokens

    def compound_embeddings(self, atoms, coefficient_ids):
        return self.dictionary.T[atoms]*self.coeff_bins[coefficient_ids].unsqueeze(-1)

    def reconstruct(self, tokens):
        atoms, bins = self.fields(tokens)
        return self.compound_embeddings(atoms, bins).sum(-2)
