"""Native compound128 coding: one greedy OMP path and hard categorical labels.

There is no beam/candidate bank, sequence posterior, or soft target sampler.
Each site selects two distinct atoms by ordinary residual correlation, jointly
refits their coefficients, and quantizes them to a fixed 128-bin physical grid.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .multiscale_laser_var import MultiScaleLaser
from .compound_multiscale_var import CompoundLaserVAR


class NativeCompound128Quantizer(nn.Module):
    def __init__(self,source,chunk_size=2048,ridge=1e-5,max_coherence=.999):
        super().__init__();self.source=source.source.requires_grad_(False).eval()
        dictionary=source.physical_tables()[0].detach().float().clone()
        self.register_buffer('dictionary',dictionary);self.register_buffer('gram',dictionary.T@dictionary)
        self.Cvae,self.vocab_size=dictionary.shape;self.sparsity=2;self.v_patch_nums=source.v_patch_nums
        if source.sparsity!=2:raise ValueError('This native codec requires two-atom supports')
        self.coefficient_bins=128;self.chunk_size=chunk_size;self.omp_ridge=float(ridge);self.omp_max_coherence=float(max_coherence)
        self.register_buffer('coefficient_max',torch.ones(len(self.v_patch_nums),device=dictionary.device))
        self.register_buffer('companding',torch.tensor(10.,device=dictionary.device))

    coefficient_values=MultiScaleLaser.coefficient_values
    coefficient_ids=MultiScaleLaser.coefficient_ids
    def normalized_dictionary(self):return self.dictionary
    def contribution(self,vectors,scale):return self.source.contribution(vectors,scale)
    def next_input(self,accumulated,scale):return self.source.next_input(accumulated,scale)

    @torch.no_grad()
    def omp_codes(self,signals):
        shape=signals.shape[:-1];flat=signals.reshape(-1,self.Cvae).float();all_atoms=[];all_coefficients=[]
        diagonal=1.+self.omp_ridge
        for x in flat.split(self.chunk_size):
            correlation=x@self.dictionary
            first=correlation.abs().argmax(-1);u=correlation.gather(-1,first[:,None]).squeeze(-1)
            residual=correlation-(u/diagonal)[:,None]*self.gram[first]
            available=torch.ones_like(residual,dtype=torch.bool).scatter_(1,first[:,None],False)
            allowed=available&(self.gram[first].abs()<=self.omp_max_coherence+1e-6)
            allowed=torch.where(allowed.any(-1,keepdim=True),allowed,available)
            second=residual.abs().masked_fill(~allowed,-1.).argmax(-1)
            v=correlation.gather(-1,second[:,None]).squeeze(-1);rho=self.gram[first,second]
            determinant=(diagonal*diagonal-rho.square()).clamp_min(1e-10)
            coefficients=torch.stack(((diagonal*u-rho*v)/determinant,(diagonal*v-rho*u)/determinant),-1)
            all_atoms.append(torch.stack((first,second),-1));all_coefficients.append(coefficients)
        return torch.cat(all_atoms).reshape(*shape,2),torch.cat(all_coefficients).reshape(*shape,2)

    def embed(self,atoms,coefficients,scale):
        if coefficients.dtype!=torch.long:raise ValueError('Expected coefficient-bin IDs')
        return (F.embedding(atoms,self.dictionary.T)*self.coefficient_values(coefficients,scale)[...,None]).sum(-2)

    def pack_pairs(self,atoms,coefficients):
        if atoms.dtype!=torch.long or coefficients.dtype!=torch.long or atoms.shape!=coefficients.shape:raise ValueError('Invalid integer pair fields')
        if (atoms<0).any() or (atoms>=self.vocab_size).any() or (coefficients<0).any() or (coefficients>=128).any():raise ValueError('Pair field out of range')
        return atoms*128+coefficients

    def unpack_pairs(self,codes):
        if codes.dtype!=torch.long or (codes<0).any() or (codes>=self.vocab_size*128).any():raise ValueError('Invalid packed pair IDs')
        return codes//128,codes%128

    @torch.no_grad()
    def _decompose(self,latent,quantized):
        with torch.autocast(device_type=latent.device.type,enabled=False):
            z=latent.float()
            if z.shape[1:]!=(self.Cvae,self.v_patch_nums[-1],self.v_patch_nums[-1]) or not z.isfinite().all():raise ValueError('Invalid latent')
            accumulated=torch.zeros_like(z);atoms=[];ids=[];values=[];inputs=[];clipping=[]
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated,scale))
                signals=F.interpolate(z-accumulated,(pn,pn),mode='area').flatten(2).transpose(1,2)
                a,c=self.omp_codes(signals);bins=self.coefficient_ids(c,scale)
                clipping.append((c.abs()>self.coefficient_max[scale]).float().mean())
                vectors=self.embed(a,bins,scale) if quantized else (F.embedding(a,self.dictionary.T)*c[...,None]).sum(-2)
                accumulated=accumulated+self.contribution(vectors,scale)
                atoms.append(a);ids.append(bins);values.append(c)
            atoms=torch.cat(atoms,1);ids=torch.cat(ids,1)
            return dict(atoms=atoms,coefficients=ids,physical_coefficients=torch.cat(values,1),codes=self.pack_pairs(atoms,ids),
                        inputs=torch.cat(inputs,1),latent=accumulated,clip_fraction=torch.stack(clipping).mean())

    def decompose(self,latent):return self._decompose(latent,True)
    def decompose_raw(self,latent):return self._decompose(latent,False)

    @torch.no_grad()
    def from_codes(self,atoms,coefficients=None):
        if coefficients is None:atoms,coefficients=self.unpack_pairs(atoms)
        self.pack_pairs(atoms,coefficients)
        if atoms.shape[1:]!=(sum(p*p for p in self.v_patch_nums),2) or (atoms[...,0]==atoms[...,1]).any():raise ValueError('Invalid support layout')
        with torch.autocast(device_type=atoms.device.type,enabled=False):
            accumulated=self.dictionary.new_zeros(len(atoms),self.Cvae,self.v_patch_nums[-1],self.v_patch_nums[-1]);inputs=[];offset=0
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated,scale))
                end=offset+pn*pn
                accumulated=accumulated+self.contribution(self.embed(atoms[:,offset:end],coefficients[:,offset:end],scale),scale);offset=end
        return accumulated,torch.cat(inputs,1)

    def forward(self,latent,ret_usages=False):
        result=self.decompose(latent);self.last_atom_ids=result['atoms'];self.last_clip_fraction=result['clip_fraction']
        return latent+(result['latent']-latent).detach(),None,latent.new_zeros(())


class NativeCompound128VAR(CompoundLaserVAR):
    def forward(self,labels,inputs,atoms,coefficients):
        loss,parts=super().forward(labels,inputs,atoms,coefficients)
        self.last_losses=dict(atom_nll=parts[0],coefficient_nll=parts[1])
        return loss
