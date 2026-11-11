"""A beam-aligned compound prior with 128-bin physical coefficient quantization.

A logical pair remains packable as atom_id * 128 + coefficient_bin_id. The
compound decoder factorizes its distribution into atom and coefficient events
using a shared, strictly causal local transformer. Beam candidates are scored
AFTER coefficient quantization, and the same quantized pairs form all inputs.
"""
import torch
from torch.nn import functional as F
from .continuous_beam_var import ContinuousBeamQuantizer
from .multiscale_laser_var import MultiScaleLaser,VAR
from .compound_multiscale_var import CompoundLaserVAR
from .beam_joint_var import joint_prefix_conditionals


class CompoundBeam128Quantizer(ContinuousBeamQuantizer):
    def __init__(self,source,*,beam_width=16,chunk_size=256,sequence_temperature=.01,max_relative_error=.01):
        super().__init__(source,beam_width=beam_width,chunk_size=chunk_size,
                         sequence_temperature=sequence_temperature,max_relative_error=max_relative_error)
        self.coefficient_bins=128
        self.register_buffer('coefficient_max',torch.ones(len(self.v_patch_nums),device=self.dictionary.device))
        self.register_buffer('companding',torch.tensor(10.,device=self.dictionary.device))

    coefficient_values=MultiScaleLaser.coefficient_values
    coefficient_ids=MultiScaleLaser.coefficient_ids

    def embed(self,atoms,coefficients,scale=None):
        # The parent support search uses physical continuous OLS coefficients.
        # Every codec/prior call supplies a scale and integer coefficient IDs.
        if scale is None:
            if not coefficients.is_floating_point():raise ValueError('Continuous support search requires physical coefficients')
            physical=coefficients
        else:
            if coefficients.dtype!=torch.long:raise ValueError('The frozen compound codec requires coefficient bin IDs')
            physical=self.coefficient_values(coefficients,scale)
        return (F.embedding(atoms,self.dictionary.T)*physical[...,None]).sum(-2)

    def pack_pairs(self,atoms,coefficients):
        if atoms.dtype!=torch.long or coefficients.dtype!=torch.long or atoms.shape!=coefficients.shape:raise ValueError('Integer pair fields must have matching shapes')
        if (atoms<0).any() or (atoms>=self.vocab_size).any() or (coefficients<0).any() or (coefficients>=128).any():raise ValueError('Pair IDs out of range')
        return atoms*128+coefficients

    def unpack_pairs(self,codes):
        if codes.dtype!=torch.long or (codes<0).any() or (codes>=self.vocab_size*128).any():raise ValueError('Invalid packed compound IDs')
        return codes//128,codes%128

    def scalar_events(self,atoms,coefficients):
        return torch.stack((atoms,self.vocab_size+coefficients),-1).flatten(-2)

    @torch.no_grad()
    def quantized_pool(self,signals,scale):
        pool=super().beam_pool(signals)
        atoms=pool['atoms'];physical=pool['coefficients'];ids=self.coefficient_ids(physical,scale)
        reconstructed=self.embed(atoms,ids,scale)
        costs=(signals.float()[...,None,:]-reconstructed).square().sum(-1)
        valid=torch.isfinite(pool['costs']);costs=costs.masked_fill(~valid,torch.inf)
        canonical=costs.argmin(-1);best=costs.gather(-1,canonical[...,None]);delta=(costs-best).clamp_min(0)
        allowed=valid&(delta<=self.max_relative_error*best.clamp_min(1e-8)+1e-6)
        if self.sequence_temperature==0:
            weights=torch.zeros_like(costs).scatter_(-1,canonical[...,None],1.)
        else:
            weights=(-delta/(self.sequence_temperature*best.clamp_min(1e-6))).masked_fill(~allowed,-torch.inf).softmax(-1)
        return dict(atoms=atoms,coefficients=ids,costs=costs,canonical_index=canonical,weights=weights,
                    clipped=(physical.abs()>self.coefficient_max[scale]))

    @torch.no_grad()
    def decompose(self,latent,*,teacher=False):
        with torch.autocast(device_type=latent.device.type,enabled=False):
            z=latent.float()
            if z.shape[1:]!=(self.Cvae,self.v_patch_nums[-1],self.v_patch_nums[-1]) or not z.isfinite().all():raise ValueError('Invalid latent')
            accumulated=torch.zeros_like(z);atoms=[];coefficients=[];inputs=[];targets=[];clipping=[];assignment=[]
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated,scale))
                signals=F.interpolate(z-accumulated,(pn,pn),mode='area').flatten(2).transpose(1,2)
                pool=self.quantized_pool(signals,scale);weights=pool['weights'];shape=weights.shape[:-1]
                selected=(torch.multinomial(weights.reshape(-1,weights.shape[-1]),1).reshape(shape) if teacher else pool['canonical_index'])
                index=selected[...,None,None].expand(*shape,1,self.sparsity)
                a=pool['atoms'].gather(-2,index).squeeze(-2);c=pool['coefficients'].gather(-2,index).squeeze(-2)
                if teacher:
                    candidate_events=self.scalar_events(pool['atoms'],pool['coefficients'])
                    sampled_events=self.scalar_events(a,c)
                    targets.append(joint_prefix_conditionals(candidate_events,weights,sampled_events))
                clipping.append(pool['clipped'].gather(-2,index).float().mean())
                assignment.append(pool['costs'].gather(-1,selected[...,None]).mean())
                accumulated=accumulated+self.contribution(self.embed(a,c,scale),scale)
                atoms.append(a);coefficients.append(c)
            atoms=torch.cat(atoms,1);coefficients=torch.cat(coefficients,1)
            result=dict(atoms=atoms,coefficients=coefficients,codes=self.pack_pairs(atoms,coefficients),inputs=torch.cat(inputs,1),latent=accumulated,
                        clip_fraction=torch.stack(clipping).mean(),assignment_mse=torch.stack(assignment).mean())
            if teacher:result['targets']={k:torch.cat([t[k] for t in targets],1) for k in targets[0]}
            return result

    @torch.no_grad()
    def from_codes(self,atoms,coefficients=None):
        if coefficients is None:atoms,coefficients=self.unpack_pairs(atoms)
        self.pack_pairs(atoms,coefficients)
        if atoms.shape[1:]!=(sum(p*p for p in self.v_patch_nums),2) or (atoms[...,0]==atoms[...,1]).any():raise ValueError('Invalid compound support layout')
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


class BeamAlignedCompound128VAR(CompoundLaserVAR):
    def forward(self,labels,inputs,atoms,coefficients,targets=None):
        tokens=self.pack_tokens(atoms,coefficients);features=VAR.forward(self,labels,inputs);logits=self.token_logits(features,tokens)
        if targets is None:
            atom_nll=F.cross_entropy(logits[:,:,::2,:self.num_atoms].reshape(-1,self.num_atoms),atoms.reshape(-1))
            coefficient_nll=F.cross_entropy(logits[:,:,1::2,self.num_atoms:].reshape(-1,128),coefficients.reshape(-1))
        else:
            weights=targets['joint_weights'];logp=logits.log_softmax(-1).gather(-1,targets['joint_ids'])
            loss=-(weights*torch.where(weights>0,logp,torch.zeros_like(logp))).sum(-1)
            atom_nll=loss[...,::2].mean();coefficient_nll=loss[...,1::2].mean()
        self.last_losses=dict(atom_nll=atom_nll.detach(),coefficient_nll=coefficient_nll.detach())
        return atom_nll+coefficient_nll
