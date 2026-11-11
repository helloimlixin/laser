"""Faster beam operations with the original FP32 arithmetic and tie behavior."""
import atexit
import json
import os
from pathlib import Path
import torch
import triton
import triton.language as tl

STATS=dict(hierarchical_calls=0,native_tie_fallbacks=0)


@triton.jit
def _cost_kernel(Corr,Ids,Values,Norms,ResidualNorm,Output,
                 A:tl.constexpr,V:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);column=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    atom=tl.load(Ids+column,column<V,0).to(tl.int32)
    value=tl.load(Values+column,column<V,0)
    correlation=tl.load(Corr+row*A+atom,column<V,0)
    norm=tl.load(Norms+atom,column<V,0)
    residual=tl.load(ResidualNorm+row)
    # Disable contraction at launch: native PyTorch rounds these separate
    # multiplies and subtractions individually, so an FMA would change codes.
    score=((2.0*correlation)*value)-(norm*(value*value))
    tl.store(Output+row*V+column,residual-score,column<V)


def candidate_cost(book,residual,depth):
    n,branches,channels=residual.shape
    correlations=residual.reshape(-1,channels)@book.dictionary
    norm=residual.square().sum(-1,keepdim=True).reshape(-1)
    result=torch.empty((n*branches,book.vocab_size),device=residual.device,dtype=torch.float32)
    _cost_kernel[(n*branches,triton.cdiv(book.vocab_size,256))](
        correlations,book.atom_ids[depth],book.coefficients[depth],book.norms,norm,result,
        A=book.dictionary.shape[1],V=book.vocab_size,BLOCK=256,enable_fp_fusion=False)
    return result.reshape(n,branches,book.vocab_size)


def select(cost,width):
    if cost.shape[1]==1 or os.environ.get('LASER_FAST_BEAM_TOPK','1')!='1':
        return cost.flatten(1).topk(width,dim=-1,largest=False).indices
    STATS['hierarchical_calls']+=1
    k=min(width+1,cost.shape[-1])
    local_values,local_tokens=cost.topk(k,dim=-1,largest=False)
    values,positions=local_values.flatten(1).topk(width+1,dim=-1,largest=False)
    # A selected tie (including the cutoff) must use native topk on the same
    # complete tensor, preserving its exact selection and ordering policy.
    if bool((values[:,1:]==values[:,:-1]).any()):
        STATS['native_tie_fallbacks']+=1
        return cost.flatten(1).topk(width,dim=-1,largest=False).indices
    positions=positions[:,:width]
    parent=positions//k
    token=local_tokens.flatten(1).gather(1,positions)
    return parent*cost.shape[-1]+token


@torch.no_grad()
def beam_pool(book,latent,beam_width=16,chunk_size=256):
    flat=latent.reshape(-1,latent.shape[-1]).float()
    greedy=book._quantize_greedy(latent,chunk_size=chunk_size)
    anchor=greedy['codes'].reshape(-1,book.depth)
    anchor_values=greedy['quantized'].reshape_as(flat)
    all_codes,all_costs,all_canonical,all_valid=[],[],[],[]
    for first in range(0,len(flat),chunk_size):
        x=flat[first:first+chunk_size];n=len(x)
        residual=x[:,None].clone()
        history=torch.empty(n,1,0,device=x.device,dtype=torch.long)
        for depth in range(book.depth):
            cost=candidate_cost(book,residual,depth)
            width=min(beam_width,cost.shape[1]*cost.shape[2])
            chosen=select(cost,width)
            parent,token=chosen//book.vocab_size,chosen%book.vocab_size
            residual=residual.gather(1,parent[...,None].expand(-1,-1,x.shape[-1]))
            residual=residual-book.embed_depth(token,depth)
            history=torch.cat([history.gather(1,parent[...,None].expand(-1,-1,depth)),token[...,None]],-1)
        best=residual.square().sum(-1).argmin(-1)
        row=torch.arange(n,device=x.device)
        candidate_values=book.embed(history).sum(-2)
        costs=(x[:,None]-candidate_values).square().sum(-1)
        anchor_cost=(x-anchor_values[first:first+n]).square().sum(-1)
        canonical=torch.where(costs[row,best]<anchor_cost,best,torch.full_like(best,width))
        anchor_codes=anchor[first:first+n]
        valid=~(history==anchor_codes[:,None]).all(-1)
        valid=torch.cat([valid,torch.ones(n,1,dtype=torch.bool,device=x.device)],-1)
        all_codes.append(torch.cat([history,anchor_codes[:,None]],1))
        all_costs.append(torch.cat([costs,anchor_cost[:,None]],1))
        all_canonical.append(canonical);all_valid.append(valid)
    return dict(candidates=torch.cat(all_codes),costs=torch.cat(all_costs),
        canonical_index=torch.cat(all_canonical),valid=torch.cat(all_valid),latent_shape=tuple(latent.shape[:-1]))


def install():
    from runtime import beam_teacher
    beam_teacher.beam_pool=beam_pool
    def save_stats():
        folder=os.environ.get('LASER_PRIVATE_SMOKE_DIR')
        if folder:
            (Path(folder)/f"exact-beam-stats-rank{os.environ['RANK']}.json").write_text(json.dumps(STATS,indent=2)+'\n')
    atexit.register(save_stats)
