"""Beam-selected sparse supports, continuous coefficients, and an aligned VAR prior.

The finite teacher is a posterior over complete (support, full-refit coefficient)
trajectories. Prefix conditioning includes BOTH fields. Density targets retain
physical FP32 coefficients; no coefficient bins or fixed scalar tables are used.
"""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .multiscale_laser_var import VAR, sample_with_top_k_top_p_
from .continuous_coefficient_mixture import ContinuousCoefficientMixture


@torch.no_grad()
def continuous_conditionals(atoms, coefficients, weights, selected):
    """Sparse q(atom_d|pairs_<d), q(coeff_d|pairs_<d, observed atom_d)."""
    if atoms.shape != coefficients.shape or weights.shape != atoms.shape[:-1]:
        raise ValueError('Expected [..., beams, depth] trajectories and [..., beams] weights')
    row = torch.arange(selected.numel(), device=atoms.device)
    shape, beams, depth = atoms.shape[:-2], atoms.shape[-2], atoms.shape[-1]
    a, c, w = atoms.reshape(-1,beams,depth), coefficients.reshape(-1,beams,depth), weights.reshape(-1,beams)
    chosen = selected.reshape(-1)
    observed_a, observed_c = a[row,chosen], c[row,chosen]
    posterior=w.clone();atom_weights=[];coefficient_weights=[]
    for d in range(depth):
        normalizer=posterior.sum(-1,keepdim=True)
        if (normalizer<=0).any():raise ValueError('Selected pair prefix is absent from the teacher')
        before=posterior/normalizer;atom_weights.append(before)
        same_atom=a[...,d]==observed_a[:,d,None]
        after=before*same_atom
        after=after/after.sum(-1,keepdim=True)
        coefficient_weights.append(after)
        # Full-refit earlier coefficients differ between trajectories with the
        # same atom prefix. Conditioning only on atom IDs would be incorrect.
        posterior=after*(c[...,d]==observed_c[:,d,None])
    return dict(atom_ids=atoms.transpose(-1,-2),
                coefficient_values=coefficients.transpose(-1,-2),
                atom_weights=torch.stack(atom_weights,-2).reshape(*shape,depth,beams),
                coefficient_weights=torch.stack(coefficient_weights,-2).reshape(*shape,depth,beams))


class ContinuousBeamQuantizer(nn.Module):
    """Two-atom beam16 OLS encoder with a shared frozen physical dictionary.

    Every first atom is scored by exact one-atom projection. The 16 retained
    prefixes expand over every available second atom using a full two-variable
    least-squares solve. The terminal bank contains the best distinct ordered
    trajectories, including the greedy OLS and original-support OLS anchors.
    Earlier scales are fixed; coefficients are jointly refitted within a site.
    """
    def __init__(self, source, *, beam_width=16, chunk_size=256,
                 sequence_temperature=.01, max_relative_error=.01):
        super().__init__()
        if source.sparsity!=2:raise ValueError('This implementation requires two atoms per site')
        self.source=source.source.requires_grad_(False).eval()
        dictionary=source.physical_tables()[0].detach().clone()
        self.register_buffer('dictionary',dictionary)
        self.register_buffer('gram',dictionary.T@dictionary)
        self.register_buffer('coefficient_scales',torch.ones(len(source.v_patch_nums),device=dictionary.device))
        self.Cvae,self.vocab_size=dictionary.shape
        self.sparsity=2;self.v_patch_nums=source.v_patch_nums
        self.beam_width=int(beam_width);self.chunk_size=int(chunk_size)
        self.sequence_temperature=float(sequence_temperature);self.max_relative_error=float(max_relative_error)
        if self.beam_width<1 or self.sequence_temperature<0 or self.max_relative_error<0:raise ValueError('Invalid beam policy')

    def normalized_dictionary(self):return self.dictionary
    def contribution(self,vectors,scale):return self.source.contribution(vectors,scale)
    def next_input(self,accumulated,scale):return self.source.next_input(accumulated,scale)
    def embed(self,atoms,coefficients):return (F.embedding(atoms,self.dictionary.T)*coefficients[...,None]).sum(-2)

    def _refit(self,correlations,atoms):
        first,second=atoms.unbind(-1)
        n1=self.gram.diag()[first];n2=self.gram.diag()[second];rho=self.gram[first,second]
        u=correlations.gather(-1,first);v=correlations.gather(-1,second)
        determinant=n1*n2-rho.square()
        coefficients=torch.stack(((n2*u-rho*v)/determinant.clamp_min(1e-7),
                                   (n1*v-rho*u)/determinant.clamp_min(1e-7)),-1)
        return coefficients,determinant>1e-5

    @torch.no_grad()
    def beam_pool(self,signals,anchor_atoms=None):
        shape=signals.shape[:-1];flat=signals.reshape(-1,self.Cvae).float()
        norm=self.gram.diag();out_a=[];out_c=[];out_cost=[]
        for start in range(0,len(flat),self.chunk_size):
            x=flat[start:start+self.chunk_size];correlations=x@self.dictionary
            width=min(self.beam_width,self.vocab_size-1)
            first=(correlations.square()/norm).topk(width,-1).indices
            u=correlations.gather(-1,first)[...,None]
            rho=self.gram[first];n1=norm[first][...,None];n2=norm[None,None]
            determinant=n1*n2-rho.square()
            v=correlations[:,None,:]
            a=(n2*u-rho*v)/determinant.clamp_min(1e-7)
            b=(n1*v-rho*u)/determinant.clamp_min(1e-7)
            cost=x.square().sum(-1)[:,None,None]-(a*u+b*v)
            available=(determinant>1e-5)&(first[...,None]!=torch.arange(self.vocab_size,device=x.device))
            cost=cost.masked_fill(~available,torch.inf)
            chosen=cost.flatten(1).topk(width,-1,largest=False).indices
            parent,second=chosen//self.vocab_size,chosen%self.vocab_size
            atoms=torch.stack((first.gather(-1,parent),second),-1)
            coefficients=torch.stack((a.flatten(1).gather(-1,chosen),b.flatten(1).gather(-1,chosen)),-1)
            # The first beam prefix is the greedy OLS prefix, so its best
            # completion is already in the search space. Keep it explicitly
            # for audits and to avoid losing it on terminal top-k ties.
            greedy_second=cost[:,0].argmin(-1)
            greedy=torch.stack((first[:,0],greedy_second),-1)[:,None]
            anchors=[greedy]
            if anchor_atoms is not None:anchors.append(anchor_atoms.reshape(-1,2)[start:start+len(x),None])
            anchors=torch.cat(anchors,1)
            anchor_c,valid=self._refit(correlations,anchors)
            if not valid.all():raise ValueError('Ill-conditioned original support')
            atoms=torch.cat((atoms,anchors),1);coefficients=torch.cat((coefficients,anchor_c),1)
            reconstruction=self.embed(atoms,coefficients)
            costs=(x[:,None]-reconstruction).square().sum(-1)
            # Remove exact ordered-support duplicates. Distinct orders are
            # distinct AR trajectories; both decode to the same physical sum.
            identity=atoms[...,0]*self.vocab_size+atoms[...,1]
            duplicate=(identity[:,:,None]==identity[:,None,:]).tril(-1).any(-1)
            costs=costs.masked_fill(duplicate,torch.inf)
            chosen=costs.topk(min(width,costs.shape[-1]),-1,largest=False).indices
            out_a.append(atoms.gather(1,chosen[...,None].expand(-1,-1,2)))
            out_c.append(coefficients.gather(1,chosen[...,None].expand(-1,-1,2)))
            out_cost.append(costs.gather(1,chosen))
        atoms=torch.cat(out_a).reshape(*shape,-1,2);coefficients=torch.cat(out_c).reshape_as(atoms)
        costs=torch.cat(out_cost).reshape(*shape,-1)
        best=costs.amin(-1,keepdim=True)
        valid=torch.isfinite(costs)&(costs<=best+self.max_relative_error*best.clamp_min(1e-8)+1e-6)
        if self.sequence_temperature==0:
            weights=torch.zeros_like(costs).scatter_(-1,costs.argmin(-1,keepdim=True),1.)
        else:
            weights=(-(costs-best)/(self.sequence_temperature*best.clamp_min(1e-6))).masked_fill(~valid,-torch.inf).softmax(-1)
        return dict(atoms=atoms,coefficients=coefficients,costs=costs,weights=weights)

    @torch.no_grad()
    def decompose(self,latent,*,teacher=False,anchor=None):
        with torch.autocast(device_type=latent.device.type,enabled=False):
            z=latent.float()
            if z.shape[1:]!=(self.Cvae,self.v_patch_nums[-1],self.v_patch_nums[-1]):raise ValueError('Invalid latent shape')
            accumulated=torch.zeros_like(z);atoms=[];coefficients=[];inputs=[];target_parts=[];costs=[]
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated,scale))
                residual=F.interpolate(z-accumulated,(pn,pn),mode='area').flatten(2).transpose(1,2)
                pool=self.beam_pool(residual,anchor_atoms=None if anchor is None else anchor[scale])
                shape=pool['weights'].shape[:-1];weights=pool['weights'].reshape(-1,pool['weights'].shape[-1])
                selected=(torch.multinomial(weights,1).squeeze(-1) if teacher else pool['costs'].reshape_as(weights).argmin(-1))
                row=torch.arange(len(selected),device=z.device)
                a=pool['atoms'].reshape(len(selected),-1,2)[row,selected].reshape(*shape,2)
                c=pool['coefficients'].reshape(len(selected),-1,2)[row,selected].reshape(*shape,2)
                if teacher:target_parts.append(continuous_conditionals(pool['atoms'],pool['coefficients'],pool['weights'],selected.reshape(shape)))
                accumulated=accumulated+self.contribution(self.embed(a,c),scale)
                atoms.append(a);coefficients.append(c);costs.append(pool['costs'].amin(-1).mean())
            result=dict(atoms=torch.cat(atoms,1),coefficients=torch.cat(coefficients,1),inputs=torch.cat(inputs,1),latent=accumulated,assignment_mse=torch.stack(costs).mean())
            if teacher:result['targets']={k:torch.cat([t[k] for t in target_parts],1) for k in target_parts[0]}
            return result

    @torch.no_grad()
    def from_codes(self,atoms,coefficients):
        if atoms.dtype!=torch.long or atoms.shape!=coefficients.shape or not coefficients.is_floating_point():raise ValueError('Expected atom IDs and continuous coefficients')
        if atoms.shape[1:]!=(sum(p*p for p in self.v_patch_nums),2) or (atoms<0).any() or (atoms>=self.vocab_size).any():raise ValueError('Invalid sparse layout')
        if not coefficients.isfinite().all() or (atoms[...,0]==atoms[...,1]).any():raise ValueError('Invalid coefficient/support values')
        with torch.autocast(device_type=atoms.device.type,enabled=False):
            accumulated=self.dictionary.new_zeros(len(atoms),self.Cvae,self.v_patch_nums[-1],self.v_patch_nums[-1]);inputs=[];offset=0
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated,scale))
                end=offset+pn*pn
                accumulated=accumulated+self.contribution(self.embed(atoms[:,offset:end],coefficients[:,offset:end].float()),scale);offset=end
        return accumulated,torch.cat(inputs,1)

    def forward(self,latent,ret_usages=False):
        result=self.decompose(latent);self.last_atom_ids=result['atoms'];self.last_clip_fraction=latent.new_zeros(())
        return latent+(result['latent']-latent).detach(),None,latent.new_zeros(())


class ContinuousBeamVAR(VAR):
    """VAR-d16 with categorical atom IDs and a conditional continuous density."""
    def __init__(self,quantizer,depth=16,width=None,heads=None,num_classes=1000):
        q=quantizer;proxy=SimpleNamespace(Cvae=q.Cvae,vocab_size=q.vocab_size,quantize=q)
        super().__init__(proxy,num_classes=num_classes,depth=depth,embed_dim=width or depth*64,num_heads=heads or depth,
                         patch_nums=q.v_patch_nums,attn_l2_norm=True,drop_path_rate=.1*depth/24,norm_eps=1e-6,
                         flash_if_available=False,fused_if_available=False)
        self._q=(q,);self.sparsity=q.sparsity
        self.depth_context=nn.Sequential(nn.Linear(q.Cvae,self.C),nn.SiLU(),nn.Linear(self.C,self.C))
        self.depth_embedding=nn.Embedding(q.sparsity,self.C);self.atom_context=nn.Linear(q.Cvae,self.C,bias=False)
        self.coefficient_head=ContinuousCoefficientMixture(self.C)
        self.init_weights(init_adaln=.5,init_adaln_gamma=1e-3,init_head=.02,init_std=-1.)
        self.coefficient_head.reset_parameters()

    def get_logits(self,hidden,condition):return self.head_nm(hidden.float(),condition).float()
    def depth_features(self,features,prefix,depth):return features+self.depth_context(prefix)+self.depth_embedding.weight[depth]
    def coefficient_parameters(self,h,atom_vectors):return self.coefficient_head(F.silu(h+self.atom_context(atom_vectors)))

    def token_parameters(self,features,atoms,coefficients):
        q=self._q[0];vectors=F.embedding(atoms,q.dictionary.T)
        prefix=features.new_zeros(*features.shape[:2],q.Cvae);logits=[];parameters=[]
        for d in range(self.sparsity):
            h=self.depth_features(features,prefix,d);score=self.head(h).float()
            if d:score=score.scatter(-1,atoms[...,:d],-torch.inf)
            logits.append(score);parameters.append(self.coefficient_parameters(h,vectors[...,d,:]))
            prefix=prefix+vectors[...,d,:]*coefficients[...,d,None]
        return torch.stack(logits,-2),tuple(torch.stack([p[i] for p in parameters],-2) for i in range(3))

    def forward(self,labels,inputs,atoms,coefficients,targets=None):
        features=super().forward(labels,inputs);logits,parameters=self.token_parameters(features,atoms,coefficients)
        if targets is None:
            target_atoms=atoms[...,None];target_values=coefficients[...,None]
            atom_weights=coefficient_weights=torch.ones_like(target_values)
        else:
            target_atoms=targets['atom_ids'];target_values=targets['coefficient_values']
            atom_weights=targets['atom_weights'];coefficient_weights=targets['coefficient_weights']
        log_atoms=logits.log_softmax(-1).gather(-1,target_atoms)
        safe=torch.where(atom_weights>0,log_atoms,torch.zeros_like(log_atoms))
        atom_nll=-(atom_weights*safe).sum(-1).mean()
        scales=torch.cat([self._q[0].coefficient_scales[s].expand(hi-lo) for s,(lo,hi) in enumerate(self.begin_ends)])
        log_coefficients=ContinuousCoefficientMixture.log_prob(parameters,target_values,scales[None,:,None,None])
        coefficient_nll=-(coefficient_weights*log_coefficients).sum(-1).mean()
        self.last_losses=dict(atom_nll=atom_nll.detach(),coefficient_nll=coefficient_nll.detach())
        return atom_nll+coefficient_nll

    @torch.no_grad()
    def sample(self,labels,cfg=1.5,top_k=900,top_p=.96,seed=0,return_codes=False):
        q=self._q[0];batch=len(labels);rng=torch.Generator(device=labels.device).manual_seed(seed)
        condition=self.class_emb(torch.cat((labels,torch.full_like(labels,self.num_classes))))
        positions=self.lvl_embed(self.lvl_1L)+self.pos_1LC
        x=condition[:,None]+self.pos_start+positions[:,:self.first_l]
        accumulated=q.dictionary.new_zeros(batch,q.Cvae,self.patch_nums[-1],self.patch_nums[-1]);offset=0;all_atoms=[];all_coefficients=[]
        for block in self.blocks:block.attn.kv_caching(True)
        try:
            for scale,pn in enumerate(self.patch_nums):
                for block in self.blocks:x=block(x=x,cond_BD=self.shared_ada_lin(condition),attn_bias=None)
                features=self.get_logits(x,condition);prefix=accumulated.new_zeros(batch,pn*pn,q.Cvae)
                atoms=torch.empty(batch,pn*pn,0,device=labels.device,dtype=torch.long);values=[]
                guidance=cfg*scale/self.num_stages_minus_1
                for d in range(self.sparsity):
                    h=self.depth_features(features,prefix.repeat(2,1,1),d)
                    score=self.head(h).float();score=(1+guidance)*score[:batch]-guidance*score[batch:]
                    if d:score=score.scatter(-1,atoms,-torch.inf)
                    selected=sample_with_top_k_top_p_(score.clone(),rng=rng,top_k=min(top_k,self.V),top_p=top_p)[:,:,0]
                    vectors=F.embedding(selected,q.dictionary.T)
                    # CFG modifies categorical atom logits. The coefficient is
                    # drawn from the normalized class-conditional density.
                    parameters=self.coefficient_parameters(h[:batch],vectors)
                    coefficient=ContinuousCoefficientMixture.sample(parameters,q.coefficient_scales[scale],rng)
                    atoms=torch.cat((atoms,selected[...,None]),-1);values.append(coefficient)
                    prefix=prefix+vectors*coefficient[...,None]
                with torch.autocast(device_type=labels.device.type,enabled=False):accumulated=accumulated+q.contribution(prefix.float(),scale)
                all_atoms.append(atoms);all_coefficients.append(torch.stack(values,-1));offset+=pn*pn
                if scale<self.num_stages_minus_1:
                    x=(self.word_embed(q.next_input(accumulated,scale+1))+positions[:,offset:offset+self.patch_nums[scale+1]**2]).repeat(2,1,1)
            return (accumulated,torch.cat(all_atoms,1),torch.cat(all_coefficients,1)) if return_codes else accumulated
        finally:
            for block in self.blocks:block.attn.kv_caching(False)
