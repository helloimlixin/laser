"""A compact scaled-atom RQ book with coefficient levels specific to each atom."""
import torch
from torch import nn


class AdaptiveScaledAtomRQ(nn.Module):
    def __init__(self,dictionary,levels,depth=4):
        super().__init__()
        dictionary=torch.as_tensor(dictionary).detach().clone()
        levels=torch.as_tensor(levels,device=dictionary.device,dtype=dictionary.dtype).detach().clone()
        if levels.ndim!=2 or levels.shape[0]!=dictionary.shape[1]:
            raise ValueError('Levels must have shape [atoms, levels_per_atom]')
        if not torch.isfinite(levels).all() or not torch.isfinite(dictionary).all():
            raise ValueError('Finite codebook required')
        if (levels==0).any() or not (levels[:,1:]>levels[:,:-1]).all():
            raise ValueError('Each atom needs sorted nonzero levels')
        self.register_buffer('dictionary',dictionary)
        self.register_buffer('levels',levels)
        self.register_buffer('norms',dictionary.square().sum(0))
        if (self.norms<=0).any() or depth<1:
            raise ValueError('Nonzero atoms and positive depth required')
        self.depth=depth
        self.levels_per_atom=levels.shape[1]
        self.vocab_size=1+levels.numel()

    def embed(self,codes):
        if (codes<0).any() or (codes>=self.vocab_size).any():
            raise ValueError('Invalid code')
        packed=(codes.long()-1).clamp_min(0)
        atoms=packed//self.levels_per_atom
        bins=packed%self.levels_per_atom
        coefficient=self.levels[atoms,bins]*(codes!=0)
        return self.dictionary.T[atoms]*coefficient[...,None]

    def expanded_codebook(self):
        entries=(self.dictionary.T[:,None]*self.levels[...,None]).flatten(0,1)
        return torch.cat([entries.new_zeros(1,entries.shape[1]),entries],0)

    @torch.no_grad()
    def quantize(self,x,return_projections=False):
        residual=x.reshape(-1,x.shape[-1]).clone()
        reconstructed=torch.zeros_like(residual)
        codes,projections=[],[]
        rows=torch.arange(len(residual),device=x.device)
        for _ in range(self.depth):
            correlations=residual@self.dictionary
            best_score=residual.new_zeros(len(residual))
            chosen=torch.zeros(len(residual),device=x.device,dtype=torch.long)
            for b in range(self.levels_per_atom):
                q=self.levels[:,b]
                scores=2*correlations*q-self.norms*q.square()
                score,atom=scores.max(-1)
                candidate=1+atom*self.levels_per_atom+b
                take=(score>best_score)|((score==best_score)&(candidate<chosen))
                best_score=torch.where(take,score,best_score)
                chosen=torch.where(take,candidate,chosen)
            if return_projections:
                atom=(chosen-1).clamp_min(0)//self.levels_per_atom
                ideal=correlations[rows,atom]/self.norms[atom]
                projections.append(ideal)
            contribution=self.embed(chosen)
            reconstructed+=contribution
            residual-=contribution
            codes.append(chosen)
        result=dict(codes=torch.stack(codes,-1).reshape(*x.shape[:-1],self.depth),
                    quantized=reconstructed.reshape_as(x))
        if return_projections:
            result['projections']=torch.stack(projections,-1).reshape(*x.shape[:-1],self.depth)
        return result


@torch.no_grad()
def fit_atom_levels(quantizer,latents,passes=4,batch_size=16,prior_weight=4.,callback=None):
    """Constrained Lloyd updates from current RQ residuals; no backbone updates.

    A pass uses one fixed codebook for all depths. Updates occur between passes.
    A small prior stabilizes coefficient estimates for rarely selected entries.
    """
    device=quantizer.dictionary.device
    prior=quantizer.levels.clone()
    trace=[]
    for iteration in range(passes):
        counts=torch.zeros(quantizer.vocab_size,device=device,dtype=torch.float64)
        sums=torch.zeros_like(counts)
        mse=0.
        for start in range(0,len(latents),batch_size):
            z=torch.as_tensor(latents[start:start+batch_size].copy(),device=device)
            result=quantizer.quantize(z,return_projections=True)
            codes=result['codes'].flatten()
            targets=result['projections'].flatten().double()
            counts+=torch.bincount(codes,minlength=quantizer.vocab_size)
            sums+=torch.bincount(codes,weights=targets,minlength=quantizer.vocab_size)
            mse+=(z-result['quantized']).square().mean().item()*len(z)
        updated=(sums[1:].reshape_as(prior)+prior_weight*prior)/(counts[1:].reshape_as(prior)+prior_weight)
        updated=updated.float().sort(-1).values
        assert (updated!=0).all() and (updated[:,1:]>updated[:,:-1]).all()
        quantizer.levels.copy_(updated)
        row=dict(iteration=iteration+1,latent_mse_before_update=mse/len(latents),
                 used_tokens=int((counts>0).sum()),zero_fraction=(counts[0]/counts.sum()).item())
        trace.append(row)
        if callback:
            callback(row)
    return trace
