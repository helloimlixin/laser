"""Training adapter for atom-specific coefficient levels in a frozen RQ book."""
from pathlib import Path

import torch
from torch import nn

from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ
from src.scaled_atom_rq import FrozenSparseBackbone
from src.scaled_atom_training import FrozenScaledTokenizer
from src.training.stochastic_targets import compact_soft_codes


class TrainingAdaptiveScaledAtomRQ(AdaptiveScaledAtomRQ):
    def get_code_emb_with_depth(self,codes):
        return self.embed(codes),None

    @torch.no_grad()
    def forward(self,z):
        result=self.quantize(z)
        prefixes=self.embed(result['codes']).cumsum(-2)
        commitment=(z[...,None,:]-prefixes).square().mean()
        return result['quantized'],commitment,result['codes']

    @torch.no_grad()
    def get_soft_codes(self,x,temp=.5,stochastic=True,chunk_size=128):
        return compact_soft_codes(self,x,temp=temp,stochastic=stochastic,chunk_size=chunk_size)


class DepthAdaptiveScaledAtomRQ(nn.Module):
    """The residual depth selects a frozen atom-specific coefficient table.

    IDs retain one zero and atom-major coefficient bins at each depth. The
    transformer always supplies the complete depth axis to its embedding hook.
    """
    def __init__(self,dictionary,levels):
        super().__init__()
        if levels.ndim!=3:
            raise ValueError('Expected [depth, atoms, levels_per_atom]')
        self.codebooks=nn.ModuleList([TrainingAdaptiveScaledAtomRQ(dictionary,book,depth=1) for book in levels])
        self.depth=len(self.codebooks)
        self.levels_per_atom=levels.shape[-1]
        self.vocab_size=self.codebooks[0].vocab_size

    @property
    def levels(self):
        return torch.stack([q.levels for q in self.codebooks])

    @property
    def dictionary(self):
        return self.codebooks[0].dictionary

    def embed(self,codes):
        if codes.shape[-1]!=self.depth:
            raise ValueError('Depth-specific embedding requires the complete depth axis')
        return torch.stack([q.embed(codes[...,d]) for d,q in enumerate(self.codebooks)],-2)

    def get_code_emb_with_depth(self,codes):
        return self.embed(codes),None

    @torch.no_grad()
    def quantize(self,x,return_projections=False):
        residual=x.clone()
        reconstructed=torch.zeros_like(x)
        codes,projections=[],[]
        for q in self.codebooks:
            result=q.quantize(residual,return_projections=return_projections)
            codes.append(result['codes'][...,0])
            residual-=result['quantized']
            reconstructed+=result['quantized']
            if return_projections:
                projections.append(result['projections'][...,0])
        codes=torch.stack(codes,-1)
        result=dict(codes=codes,quantized=reconstructed)
        if return_projections:
            result['projections']=torch.stack(projections,-1)
        return result

    @torch.no_grad()
    def forward(self,z):
        result=self.quantize(z)
        prefixes=self.embed(result['codes']).cumsum(-2)
        return result['quantized'],(z[...,None,:]-prefixes).square().mean(),result['codes']

    @torch.no_grad()
    def get_soft_codes(self,x,temp=.5,stochastic=True,chunk_size=128):
        return compact_soft_codes(self,x,temp=temp,stochastic=stochastic,chunk_size=chunk_size)


def adaptive_quantizer(dictionary,levels,depth=4):
    if levels.ndim==3:
        if levels.shape[0]!=depth:
            raise ValueError('Coefficient tables do not match residual depth')
        return DepthAdaptiveScaledAtomRQ(dictionary,levels)
    return TrainingAdaptiveScaledAtomRQ(dictionary,levels,depth=depth)


class FrozenCompactTokenizer(FrozenScaledTokenizer):
    def __init__(self,checkpoint,codebook):
        nn.Module.__init__(self)
        self.backbone=FrozenSparseBackbone(Path(checkpoint))
        spec=torch.load(codebook,map_location='cpu',weights_only=True)
        if spec['kind'] not in ('adaptive_scaled_atom_rq','depth_adaptive_scaled_atom_rq'):
            raise ValueError('Expected a calibrated atom-specific codebook')
        torch.testing.assert_close(spec['dictionary'],self.backbone.dictionary,rtol=0,atol=0)
        self.quantizer=adaptive_quantizer(spec['dictionary'],spec['levels'],depth=spec['depth'])
        self.code_shape=torch.Size(spec['code_shape'])
        self.requires_grad_(False).eval()
