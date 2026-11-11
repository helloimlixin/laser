"""One categorical integer per atom/coefficient pair, with an aligned beam teacher.

The selected stage1 dictionary and per-atom scalar levels stay frozen. A pair
is represented by atom_id * levels + level_id. Beam search and the prior use
these same physical codewords; decoding never receives a separate coefficient.
"""
import torch
from torch.nn import functional as F
from .joint_multiscale_var import JointMultiScaleQuantizer,JointLaserVAR
from .multiscale_laser_var import VAR


@torch.no_grad()
def joint_prefix_conditionals(candidates,weights,codes):
    """Sparse exact q(joint_id_d | sampled joint_ids_<d)."""
    if candidates.shape[:-1]!=weights.shape or candidates.shape[:-2]+candidates.shape[-1:]!=codes.shape:
        raise ValueError('Expected [..., beams, depth] candidates and [..., depth] observed IDs')
    matching=torch.ones_like(weights,dtype=torch.bool);conditional=[]
    for depth in range(candidates.shape[-1]):
        mass=weights*matching;normalizer=mass.sum(-1,keepdim=True)
        if (normalizer<=0).any():raise ValueError('Observed joint prefix is absent from the beam posterior')
        conditional.append(mass/normalizer)
        matching &= candidates[...,depth]==codes[...,depth,None]
    return dict(joint_ids=candidates.transpose(-1,-2),joint_weights=torch.stack(conditional,-2))


class BeamJointQuantizer(JointMultiScaleQuantizer):
    """Beam16 over complete joint-ID trajectories, with a greedy anchor."""
    def __init__(self,source,*,beam_width=16,chunk_size=256,sequence_temperature=.01,max_relative_error=.01):
        dictionary,coefficients=source.physical_tables() if hasattr(source,'physical_tables') else (source.dictionary,source.coefficients)
        super().__init__(source.source,dictionary,coefficients)
        self.beam_width=int(beam_width);self.chunk_size=int(chunk_size)
        self.sequence_temperature=float(sequence_temperature);self.max_relative_error=float(max_relative_error)
        if self.beam_width<1 or self.chunk_size<1 or self.sequence_temperature<0 or self.max_relative_error<0:raise ValueError('Invalid beam policy')

    def pairs_to_ids(self,atoms,coefficients,scale):
        """Quantize physical coefficients to the learned levels for each atom."""
        if atoms.dtype!=torch.long or atoms.shape!=coefficients.shape or (atoms<0).any() or (atoms>=self.num_directions).any():
            raise ValueError('Invalid atom/coefficient pairs')
        if not coefficients.isfinite().all():raise ValueError('Nonfinite physical coefficients')
        level=(coefficients[...,None]-self.coefficients[scale,atoms]).square().argmin(-1)
        return atoms*self.levels+level

    def ids_to_pairs(self,codes,scale):
        if codes.dtype!=torch.long or (codes<0).any() or (codes>=self.vocab_size).any():raise ValueError('Invalid joint IDs')
        atoms=codes.div(self.levels,rounding_mode='floor');levels=codes.remainder(self.levels)
        return atoms,self.coefficients[scale,atoms,levels]

    @torch.no_grad()
    def beam_pool(self,signals,scale):
        shape=signals.shape[:-1];flat=signals.reshape(-1,self.Cvae).float()
        greedy,_=super().quantize_scale(signals,scale,chunk_size=self.chunk_size)
        greedy=greedy.reshape(-1,self.sparsity)
        outputs={k:[] for k in ['candidates','costs','valid','canonical_index','weights']}
        norms=self.dictionary.square().sum(0);table=self.coefficients[scale]
        for start in range(0,len(flat),self.chunk_size):
            x=flat[start:start+self.chunk_size];n=len(x)
            residual=x[:,None].clone();history=torch.empty(n,1,0,device=x.device,dtype=torch.long)
            for depth in range(self.sparsity):
                branches=residual.shape[1]
                correlations=(residual.reshape(-1,self.Cvae)@self.dictionary).reshape(n,branches,self.num_directions)
                score=(2*correlations[...,None]*table-norms[:,None]*table.square()).flatten(-2)
                score=self.mask_previous(score,history)
                cost=residual.square().sum(-1,keepdim=True)-score
                width=min(self.beam_width,cost.shape[1]*cost.shape[2])
                selected=cost.flatten(1).topk(width,-1,largest=False).indices
                parent,token=selected//self.vocab_size,selected%self.vocab_size
                residual=residual.gather(1,parent[...,None].expand(-1,-1,self.Cvae))-self.embed_joint(token,scale)
                history=torch.cat((history.gather(1,parent[...,None].expand(-1,-1,depth)),token[...,None]),-1)
            anchor=greedy[start:start+n,None]
            candidates=torch.cat((history,anchor),1)
            values=self.embed_joint(candidates,scale).sum(-2)
            costs=(x[:,None]-values).square().sum(-1)
            valid=torch.cat((~(history==anchor).all(-1),torch.ones(n,1,device=x.device,dtype=torch.bool)),-1)
            row=torch.arange(n,device=x.device);best=costs[:,:width].argmin(-1)
            canonical=torch.where(costs[row,best]<costs[:,-1],best,torch.full_like(best,width))
            best_cost=costs.gather(1,canonical[:,None]);delta=(costs-best_cost).clamp_min(0)
            allowed=valid&(delta<=self.max_relative_error*best_cost.clamp_min(1e-8)+1e-6)
            allowed.scatter_(1,canonical[:,None],True)
            if self.sequence_temperature==0:
                weights=torch.zeros_like(costs).scatter_(1,canonical[:,None],1.)
            else:
                weights=(-delta/(self.sequence_temperature*best_cost.clamp_min(1e-6))).masked_fill(~allowed,-torch.inf).softmax(-1)
            for key,value in [('candidates',candidates),('costs',costs),('valid',valid),('canonical_index',canonical),('weights',weights)]:outputs[key].append(value)
        return {key:torch.cat(values).reshape(*shape,*values[0].shape[1:]) for key,values in outputs.items()}

    @torch.no_grad()
    def decompose(self,latent,*,teacher=False):
        with torch.autocast(device_type=latent.device.type,enabled=False):
            z=latent.float()
            if z.shape[1:]!=(self.Cvae,self.v_patch_nums[-1],self.v_patch_nums[-1]) or not z.isfinite().all():raise ValueError('Invalid encoder latent')
            accumulated=torch.zeros_like(z);codes=[];inputs=[];targets=[];beam_cost=[];anchor_cost=[]
            for scale,pn in enumerate(self.v_patch_nums):
                if scale:inputs.append(self.next_input(accumulated,scale))
                signals=F.interpolate(z-accumulated,(pn,pn),mode='area').flatten(2).transpose(1,2)
                pool=self.beam_pool(signals,scale);weights=pool['weights'];leading=weights.shape[:-1]
                chosen=(torch.multinomial(weights.reshape(-1,weights.shape[-1]),1).reshape(leading) if teacher else pool['canonical_index'])
                ids=pool['candidates'].gather(-2,chosen[...,None,None].expand(*leading,1,self.sparsity)).squeeze(-2)
                if teacher:targets.append(joint_prefix_conditionals(pool['candidates'],weights,ids))
                accumulated=accumulated+self.contribution(self.embed_joint(ids,scale).sum(-2),scale)
                codes.append(ids);beam_cost.append(pool['costs'].gather(-1,pool['canonical_index'][...,None]).mean());anchor_cost.append(pool['costs'][...,-1].mean())
            result=dict(codes=torch.cat(codes,1),inputs=torch.cat(inputs,1),latent=accumulated,
                        beam_assignment_mse=torch.stack(beam_cost).mean(),greedy_anchor_mse=torch.stack(anchor_cost).mean())
            if teacher:result['targets']={key:torch.cat([t[key] for t in targets],1) for key in targets[0]}
            return result

    def forward(self,latent,ret_usages=False):
        result=self.decompose(latent);self.last_atom_ids=result['codes']//self.levels;self.last_clip_fraction=latent.new_zeros(())
        return latent+(result['latent']-latent).detach(),None,latent.new_zeros(())


class BeamAlignedJointVAR(JointLaserVAR):
    """The same VAR body and shared joint classifier, trained on beam conditionals."""
    def forward(self,labels,inputs,codes,targets=None):
        features=VAR.forward(self,labels,inputs);logits=self.token_logits(features,codes)
        if targets is None:return F.cross_entropy(logits.flatten(0,2),codes.flatten())
        weights=targets['joint_weights'];logp=logits.log_softmax(-1).gather(-1,targets['joint_ids'])
        safe=torch.where(weights>0,logp,torch.zeros_like(logp))
        return -(weights*safe).sum(-1).mean()
