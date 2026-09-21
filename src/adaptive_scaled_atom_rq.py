"""A compact scaled-atom RQ book with coefficient levels specific to each atom."""
import math
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
    """Fit shared levels to the final reconstruction, including repeated tokens.

    Solve the fixed-code normal equations with a quadratic prior. Reassign the
    greedy residual codes and backtrack until both reconstruction SSE and the
    regularized objective do not increase. Rejected proposals restore the book.
    """
    from src.compact_rq_fitting import coefficient_normal_equations, solve_regularized_levels
    if passes < 0 or batch_size < 1 or len(latents) == 0:
        raise ValueError('Nonempty latents, nonnegative passes, and positive batch size required')
    if not math.isfinite(prior_weight) or prior_weight <= 0:
        raise ValueError('A finite positive prior_weight is required')
    device=quantizer.dictionary.device
    prior=quantizer.levels.clone()
    trace=[]

    def batches():
        for start in range(0,len(latents),batch_size):
            values = latents[start:start+batch_size]
            if not torch.is_tensor(values):
                values = values.copy()
            z = torch.as_tensor(values, device=device, dtype=quantizer.dictionary.dtype)
            if not torch.isfinite(z).all():
                raise ValueError('Finite fitting latents required')
            yield z

    def reconstruction_sse():
        return sum((z.double()-quantizer.quantize(z)['quantized'].double()).square().sum().item()
                   for z in batches())

    for iteration in range(passes):
        counts=torch.zeros(quantizer.vocab_size,device=device,dtype=torch.float64)
        rhs=torch.zeros(quantizer.vocab_size-1,device=device,dtype=torch.float64)
        indices,entries=[],[]
        sse=0.
        elements=0
        current=quantizer.levels.clone()
        for z in batches():
            result=quantizer.quantize(z)
            codes=result['codes'].flatten()
            counts+=torch.bincount(codes,minlength=quantizer.vocab_size)
            normal,local_rhs=coefficient_normal_equations(
                quantizer.dictionary,result['codes'],z,quantizer.levels_per_atom)
            normal=normal.coalesce()
            indices.append(normal.indices())
            entries.append(normal.values())
            rhs+=local_rhs
            sse+=(z.double()-result['quantized'].double()).square().sum().item()
            elements+=z.numel()
        normal=torch.sparse_coo_tensor(torch.cat(indices,1),torch.cat(entries),
            (len(rhs),len(rhs)),device=device).coalesce()
        proposal=solve_regularized_levels(normal,rhs,current,prior,prior_weight)
        baseline=sse+prior_weight*(current.double()-prior.double()).square().sum().item()
        accepted=False
        after_sse,after_objective=sse,baseline
        relaxation=0.
        try:
            for backtrack in range(9):
                fraction=2.**(-backtrack)
                candidate=current.lerp(proposal,fraction)
                if not (torch.isfinite(candidate).all() and (candidate!=0).all()
                        and (candidate[:,1:]>candidate[:,:-1]).all()):
                    continue
                quantizer.levels.copy_(candidate)
                trial_sse=reconstruction_sse()
                trial_objective=trial_sse+prior_weight*(candidate.double()-prior.double()).square().sum().item()
                if trial_sse <= sse and trial_objective <= baseline:
                    accepted=True
                    relaxation=fraction
                    after_sse,after_objective=trial_sse,trial_objective
                    break
        finally:
            if not accepted:
                quantizer.levels.copy_(current)
        row=dict(iteration=iteration+1,latent_mse_before_update=sse/elements,
                 latent_mse_after_update=after_sse/elements,accepted=accepted,relaxation=relaxation,
                 regularized_objective_before_update=baseline,
                 regularized_objective_after_update=after_objective,
                 used_tokens=int((counts>0).sum()),zero_fraction=(counts[0]/counts.sum()).item())
        trace.append(row)
        if callback:
            callback(row)
    return trace
