"""Offline whole-pair vector fitting at a fixed autoregressive vocabulary size."""
import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training.laser_compact_joint_vocabulary import reconstruct_fixed_codes


@torch.no_grad()
def untie_pair_directions(quantizer):
    """Keep every codeword exactly; release the collinearity between pair IDs."""
    if not torch.equal(quantizer.atom_ids, quantizer.atom_ids[:1].expand_as(quantizer.atom_ids)):
        raise ValueError('Pair IDs must have a common atom layout across depths')
    count=quantizer.vocab_size-1
    dictionary=quantizer.dictionary[:,quantizer.atom_ids[0,1:]]
    atoms=torch.cat((torch.zeros(1,device=dictionary.device,dtype=torch.long),
        torch.arange(count,device=dictionary.device))).expand(quantizer.depth,-1)
    return JointAtomCoefficientRQ(dictionary,atoms,quantizer.coefficients).to(dictionary.device)


@torch.no_grad()
def sparse_dictionary_solve(signals,atoms,coefficients,prior,*,ridge_fraction=.01,iterations=16,tolerance=1.e-5):
    """Solve the exact sparse-design normal equations with preconditioned CG.

    All cross-depth terms, including repeated pair IDs, are included. No dense
    20k-by-20k Gram matrix or image-specific coefficient fitting is needed.
    """
    if (signals.ndim!=2 or atoms.shape!=coefficients.shape or atoms.ndim!=2
            or atoms.shape[0]!=len(signals) or prior.shape[0]!=signals.shape[1]
            or ridge_fraction<=0 or iterations<1):
        raise ValueError('Invalid sparse regression dimensions or regularization')
    x=signals.float();c=coefficients.float();depth=atoms.shape[1];count=prior.shape[1]
    diagonal=x.new_zeros(count);rhs=x.new_zeros(count,x.shape[1])
    for d in range(depth):
        diagonal.index_add_(0,atoms[:,d],c[:,d].square())
        rhs.index_add_(0,atoms[:,d],x*c[:,d,None])
    ridge=ridge_fraction*diagonal.clamp_min(1.)
    rhs.add_(prior.T*ridge[:,None])

    def multiply(values):
        prediction=torch.zeros_like(x)
        for d in range(depth):prediction.add_(values[atoms[:,d]]*c[:,d,None])
        result=values*ridge[:,None]
        for d in range(depth):result.index_add_(0,atoms[:,d],prediction*c[:,d,None])
        return result

    solution=prior.T.clone()
    residual=rhs-multiply(solution)
    preconditioner=diagonal+ridge
    preconditioned=residual/preconditioner[:,None]
    direction=preconditioned.clone()
    rz=(residual*preconditioned).sum(dtype=torch.float64)
    initial=float(residual.square().sum(dtype=torch.float64).sqrt())
    trace=[]
    if initial==0.:return solution.T,dict(iterations=0,initial_residual_norm=0.,relative_residual_norm=0.)
    for iteration in range(iterations):
        product=multiply(direction)
        denominator=(direction*product).sum(dtype=torch.float64)
        if denominator<=0:raise ArithmeticError('Sparse normal operator is not positive definite')
        alpha=(rz/denominator).float()
        solution.add_(direction*alpha)
        residual.sub_(product*alpha)
        ratio=float(residual.square().sum(dtype=torch.float64).sqrt())/initial
        trace.append(ratio)
        if ratio<tolerance:break
        preconditioned=residual/preconditioner[:,None]
        updated_rz=(residual*preconditioned).sum(dtype=torch.float64)
        direction=preconditioned+direction*(updated_rz/rz).float()
        rz=updated_rz
    assert torch.isfinite(solution).all()
    return solution.T,dict(iterations=len(trace),initial_residual_norm=initial,
        relative_residual_norm=trace[-1],residual_trace=trace)


@torch.no_grad()
def refit_pair_prototypes(quantizer,signals,codes,*,cg_iterations=16,ridge_fraction=.01,prior_weight=4.):
    x=signals.reshape(-1,signals.shape[-1]).float();codes=codes.reshape(-1,quantizer.depth)
    count=quantizer.vocab_size-1
    if quantizer.dictionary.shape[1]!=count or not torch.equal(quantizer.atom_ids[:,1:],
            torch.arange(count,device=codes.device).expand(quantizer.depth,-1)):
        raise ValueError('Expected one independent direction per non-null pair ID')
    atoms=torch.stack([quantizer.atom_ids[d,codes[:,d]] for d in range(quantizer.depth)],-1)
    coefficients=torch.stack([quantizer.coefficients[d,codes[:,d]] for d in range(quantizer.depth)],-1)
    before=float((x-reconstruct_fixed_codes(quantizer,codes)).square().mean())
    updated,solver=sparse_dictionary_solve(x,atoms,coefficients,quantizer.dictionary,
        ridge_fraction=ridge_fraction,iterations=cg_iterations)
    norm=updated.norm(dim=0)
    assert (norm>1.e-8).all()
    quantizer.dictionary.copy_(updated/norm)
    quantizer.coefficients[:,1:].mul_(norm[None,:])
    quantizer.norms.copy_(quantizer.dictionary.square().sum(0))
    error=x-reconstruct_fixed_codes(quantizer,codes)
    after_dictionary=float(error.square().mean())
    for d in range(quantizer.depth):
        chosen=codes[:,d];vectors=quantizer.dictionary.T[atoms[:,d]]
        old=quantizer.coefficients[d].clone()
        optimum=old[chosen]+(error*vectors).sum(-1)/quantizer.norms[atoms[:,d]]
        frequency=torch.bincount(chosen,minlength=quantizer.vocab_size)
        total=torch.bincount(chosen,weights=optimum,minlength=quantizer.vocab_size)
        proposal=(total+prior_weight*old)/(frequency+prior_weight)
        proposal=torch.where(old<0,proposal.clamp(max=-1.e-6),proposal.clamp(min=1.e-6))
        proposal[0]=0.
        error.sub_(vectors*(proposal[chosen]-old[chosen])[:,None])
        quantizer.coefficients[d].copy_(proposal)
    after=float(error.square().mean())
    assert after_dictionary<=before+2.e-6*max(1.,before),(before,after_dictionary)
    assert after<=after_dictionary+2.e-6*max(1.,after_dictionary),(after_dictionary,after)
    return dict(fixed_assignment_mse_before=before,fixed_assignment_mse_after_dictionary=after_dictionary,
        fixed_assignment_mse_after=after,solver=solver)
