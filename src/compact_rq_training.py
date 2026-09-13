"""Training adapter for atom-specific coefficient levels in a frozen RQ book."""
from pathlib import Path

import torch
from torch import nn

from src.adaptive_scaled_atom_rq import AdaptiveScaledAtomRQ
from src.scaled_atom_rq import FrozenSparseBackbone
from src.scaled_atom_training import FrozenScaledTokenizer


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
        if temp<=0 or chunk_size<1:
            raise ValueError('Positive temperature and chunk size required')
        shape=x.shape[:-1]
        residual=x.reshape(-1,x.shape[-1]).float().clone()
        targets=residual.new_empty(len(residual),self.depth,self.vocab_size)
        codes=torch.empty(len(residual),self.depth,device=x.device,dtype=torch.long)
        penalty=self.norms[:,None]*self.levels.square()
        for d in range(self.depth):
            for start in range(0,len(residual),chunk_size):
                r=residual[start:start+chunk_size]
                correlations=r@self.dictionary
                score=(2*correlations[...,None]*self.levels[None,:,:]-penalty).flatten(1)
                score=torch.cat([score.new_zeros(len(r),1),score],1)
                probability=(score/temp).softmax(-1)
                chosen=(torch.multinomial(probability,1).squeeze(-1) if stochastic else score.argmax(-1))
                targets[start:start+len(r),d]=probability
                codes[start:start+len(r),d]=chosen
                r.sub_(self.embed(chosen))
        return targets.reshape(*shape,self.depth,self.vocab_size),codes.reshape(*shape,self.depth)


class FrozenCompactTokenizer(FrozenScaledTokenizer):
    def __init__(self,checkpoint,codebook):
        nn.Module.__init__(self)
        self.backbone=FrozenSparseBackbone(Path(checkpoint))
        spec=torch.load(codebook,map_location='cpu',weights_only=True)
        if spec['kind']!='adaptive_scaled_atom_rq':
            raise ValueError('Expected a calibrated atom-specific codebook')
        torch.testing.assert_close(spec['dictionary'],self.backbone.dictionary,rtol=0,atol=0)
        self.quantizer=TrainingAdaptiveScaledAtomRQ(spec['dictionary'],spec['levels'],depth=spec['depth'])
        self.code_shape=torch.Size(spec['code_shape'])
        self.requires_grad_(False).eval()
