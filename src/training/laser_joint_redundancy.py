"""Read-only geometric redundancy measurements for a shared pair-ID layout."""
import torch


@torch.no_grad()
def nearest_cross_atom_pairs(dictionary, atom_ids, coefficients, *, top_atoms=16, chunk_size=512):
    if (dictionary.ndim != 2 or atom_ids.ndim != 2 or coefficients.shape != atom_ids.shape
            or not torch.equal(atom_ids, atom_ids[:1].expand_as(atom_ids))
            or (atom_ids[:, 1:].diff(dim=-1) < 0).any()
            or (coefficients[:, 0] != 0).any() or (coefficients[:, 1:] == 0).any()
            or dictionary.shape[1] < 2 or top_atoms < 1 or chunk_size < 1):
        raise ValueError('Expected a shared contiguous pair layout, null ID and nonzero pair vectors')
    d=dictionary.float();a=atom_ids[0,1:];c=coefficients.float()
    count=d.shape[1];k=min(top_atoms,count-1)
    frequency=torch.bincount(a,minlength=count)
    starts=frequency.cumsum(0)-frequency
    lookup=torch.full((count,int(frequency.max())), -1,device=d.device,dtype=torch.long)
    lookup[a,torch.arange(len(a),device=d.device)-starts[a]]=torch.arange(1,len(a)+1,device=d.device)
    norms=d.square().sum(0)
    if (norms<=0).any():raise ValueError('Zero dictionary directions')
    gram=d.T@d
    cosine=gram/(norms[:,None]*norms[None,:]).sqrt()
    similarity=cosine.abs();similarity.fill_diagonal_(-1.)
    ranked,neighbors=similarity.topk(min(k+1,count-1),dim=-1)
    omitted_cosine=ranked[:,k] if k<count-1 else torch.zeros(count,device=d.device)
    omitted_bound=(1.-omitted_cosine.square()).clamp_min(0).sqrt()
    neighbors=neighbors[:,:k]
    nearest=torch.full((len(a)+1,),-1,device=d.device,dtype=torch.long)
    relative=torch.full((len(a)+1,),torch.inf,device=d.device)
    for first in range(0,len(a),chunk_size):
        ids=torch.arange(first+1,min(first+chunk_size,len(a))+1,device=d.device)
        source=a[first:first+len(ids)]
        other=neighbors[source]
        candidate=lookup[other]
        valid=candidate>=1;safe=candidate.clamp_min(0)
        own=c[:,ids,None,None]
        values=c[:,safe]
        distances=(own.square()*norms[source][None,:,None,None]
            +values.square()*norms[other][None,:,:,None]
            -2.*own*values*gram[source[:,None],other][None,:,:,None]).sum(0).clamp_min(0)
        distances.masked_fill_(~valid,torch.inf)
        minimum,index=distances.flatten(1).min(-1)
        nearest[ids]=candidate.flatten(1).gather(1,index[:,None]).squeeze(1)
        relative[ids]=(minimum/(c[:,ids].square().sum(0)*norms[source])).sqrt()
    return dict(nearest_code=nearest.cpu(),relative_vector_rmse=relative.cpu(),
        omitted_direction_lower_bound=omitted_bound[a].cpu(),
        maximum_cross_atom_cosine=ranked[:,0].cpu(),candidate_directions=k)
