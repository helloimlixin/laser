"""Retain the flat joint head and add a zero-initialized signed-atom score."""
import torch
from torch import nn
from torch.nn import functional as F


class GeometricJointLinear(nn.Module):
    def __init__(self,original,dictionary,token_atoms,token_signs):
        super().__init__()
        self.weight=original.weight;self.bias=original.bias
        self.in_features,self.out_features=original.in_features,original.out_features
        self.multiple=getattr(original,'multiple',1)
        self.padding=(-self.out_features)%self.multiple
        dictionary=F.normalize(dictionary.detach().to(self.weight.device).float(),dim=0)
        atoms=token_atoms.detach().to(self.weight.device).long().clone()
        signs=token_signs.detach().to(self.weight.device).float().clone()
        if atoms.shape!=signs.shape or len(atoms)!=self.out_features:
            raise ValueError('One signed atom is required for each joint token')
        if signs[0]!=0 or (signs[1:].abs()!=1).any():
            raise ValueError('The null token must be the only zero sign')
        self.register_buffer('geometry_dictionary',dictionary,persistent=False)
        self.register_buffer('token_atoms',atoms,persistent=False)
        self.register_buffer('token_signs',signs,persistent=False)
        self.geometry_weight=nn.Parameter(self.weight.new_zeros(dictionary.shape[0],self.in_features))

    def forward(self,hidden):
        if self.padding:
            weight=F.pad(self.weight,(0,0,0,self.padding))
            bias=None if self.bias is None else F.pad(self.bias,(0,self.padding))
            logits=F.linear(hidden,weight,bias)[...,:self.out_features]
        else:logits=F.linear(hidden,self.weight,self.bias)
        query=F.linear(hidden,self.geometry_weight)
        correlations=query@self.geometry_dictionary.to(query.dtype)
        delta=correlations.index_select(-1,self.token_atoms)*self.token_signs.to(correlations.dtype)
        return logits+delta


def add_geometric_joint_classifier(model,quantizer,*,shuffle_seed=None):
    atoms=quantizer.atom_ids
    signs=quantizer.coefficients.sign()
    if not torch.equal(atoms,atoms[:1].expand_as(atoms)) or not torch.equal(signs,signs[:1].expand_as(signs)):
        raise ValueError('Atom and sign identity must agree at every depth')
    dictionary=quantizer.dictionary
    if shuffle_seed is not None:
        order=torch.randperm(dictionary.shape[1],generator=torch.Generator().manual_seed(shuffle_seed))
        dictionary=dictionary[:,order.to(dictionary.device)]
    model.classifier.linear=GeometricJointLinear(model.classifier.linear,dictionary,atoms[0],signs[0])
