"""Fused execution of the existing native compound128 loss, with unchanged parameters."""
import types
import torch
from torch.nn import functional as F
from src.models.multiscale_laser_var import VAR


def typed_compound_core(model,labels,inputs,atoms,coefficients):
    features=VAR.forward(model,labels,inputs)
    batch,sites,depth=atoms.shape
    tokens=torch.stack((atoms,model.num_atoms+coefficients),-1).flatten(-2)
    previous=model.token_embedding(tokens[...,:-1]).reshape(batch*sites,model.tokens_per_site-1,-1)
    x=torch.cat((model.token_start.expand(batch*sites,-1,-1),previous),1)
    x=x+model.token_position.weight+model.spatial_condition(features).reshape(batch*sites,1,-1)
    x=model.token_transformer(x,mask=model.token_causal_mask,is_causal=True)
    hidden=model.token_output(x).reshape(batch,sites,model.tokens_per_site,model.C)
    # The original full classifier discards the other token type at each
    # event. Evaluate only the rows that participate in each conditional CE.
    atom_logits=F.linear(hidden[:,:,::2],model.head.weight[:model.num_atoms],model.head.bias[:model.num_atoms]).float()
    coefficient_logits=F.linear(hidden[:,:,1::2],model.head.weight[model.num_atoms:],model.head.bias[model.num_atoms:]).float()
    first=atom_logits[:,:,0]
    second=atom_logits[:,:,1].scatter(-1,atoms[:,:,:1],-torch.inf)
    atom_logits=torch.stack((first,second),2)
    atom_nll=F.cross_entropy(atom_logits.reshape(-1,model.num_atoms),atoms.reshape(-1))
    coefficient_nll=F.cross_entropy(coefficient_logits.reshape(-1,model.coefficient_bins),coefficients.reshape(-1))
    return atom_nll+coefficient_nll,atom_nll.detach(),coefficient_nll.detach()


def install_typed_head(model,compile_forward=False):
    assert model.sparsity==2 and model.tokens_per_site==4
    expected=torch.ones_like(model.token_causal_mask,dtype=torch.bool).triu(1)
    assert torch.equal(expected,model.token_causal_mask)
    original=model.forward
    core=typed_compound_core
    if compile_forward:
        core=torch.compile(core,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def forward(self,labels,inputs,atoms,coefficients):
        if not self.training:return original(labels,inputs,atoms,coefficients)
        loss,atom_nll,coefficient_nll=core(self,labels,inputs,atoms,coefficients)
        self.last_losses=dict(atom_nll=atom_nll,coefficient_nll=coefficient_nll)
        return loss
    model.forward=types.MethodType(forward,model)
    return original
