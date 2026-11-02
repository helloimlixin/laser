"""Predict complete finer pairs from the full image of completed coarse pairs."""
from collections import OrderedDict
from types import SimpleNamespace

import torch
from torch import nn

from src.models.structured_pair_energy import StructuredPairEnergyRQ


class _RefinementHeads(nn.Module):
    def __init__(self, width, atoms, bins, fine_levels):
        super().__init__()
        self.config=SimpleNamespace(embed_dim=width)
        self.mask_seen_atoms_training=False
        self.classifier=nn.Sequential(OrderedDict([
            ('layer_norm',nn.LayerNorm(width)),('linear',nn.Linear(width,atoms)),
        ]))
        self.coeff_classifier=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,bins))
                                            for _ in range(fine_levels)])

    def classify_coefficients(self, hidden, depth_index=None):
        if depth_index is not None:
            return self.coeff_classifier[depth_index](hidden)
        return torch.stack([head(hidden[...,d,:]) for d,head in enumerate(self.coeff_classifier)],-2)


class CoarsePrefixPairRefiner(nn.Module):
    """A normalized conditional prior for levels 1..K-1, with level 0 supplied.

    Within a level, sites are conditionally independent given complete coarser
    maps. Attention is bidirectional in space and never crosses the batched
    refinement-level dimension. Current/future pairs cannot enter a prediction.
    """
    def __init__(self, channels, atoms, bins, *, height=8, width=8, depth=4,
                 hidden=512, layers=6, heads=8, classes=1000, dropout=.1,
                 conditional_coefficients=False):
        super().__init__()
        if depth<2:
            raise ValueError('refinement needs at least two levels')
        self.height,self.width,self.depth=height,width,depth
        self.num_atoms=atoms.shape[1]
        self.conditional_coefficients=bool(conditional_coefficients)
        self.coefficient_embedding=nn.Embedding(len(bins),channels)
        self.pair_adapter=nn.Sequential(nn.LayerNorm(3*channels),nn.Linear(3*channels,channels),
                                        nn.SiLU(),nn.Linear(channels,channels))
        self.input_projection=nn.Linear(channels,hidden)
        self.position=nn.Parameter(torch.randn(1,height*width,hidden)*.02)
        self.level=nn.Parameter(torch.randn(1,depth-1,1,hidden)*.02)
        self.condition=nn.Embedding(classes,hidden)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,heads,4*hidden,
            dropout=dropout,activation='gelu',batch_first=True,norm_first=True),layers,
            enable_nested_tensor=False)
        self.readout=StructuredPairEnergyRQ(_RefinementHeads(hidden,self.num_atoms,len(bins),depth-1),atoms,bins)
        if self.conditional_coefficients:
            self.coefficient_atom_projection=nn.Linear(channels,hidden,bias=False)
            self.coefficient_refiner=nn.Sequential(nn.LayerNorm(3*hidden),nn.Linear(3*hidden,hidden),
                nn.SiLU(),nn.Linear(hidden,hidden))
        self.apply(self._initialize)
        nn.init.zeros_(self.readout.query.weight)

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,std=.02)
            if getattr(module,'bias',None) is not None: nn.init.zeros_(module.bias)

    def forward(self, tokens, codebook, cond, *, local_only=False):
        atoms,bins=codebook.fields(tokens)
        physical=codebook.compound_embeddings(atoms,bins)
        features=torch.cat((codebook.dictionary.T[atoms],self.coefficient_embedding(bins),physical),-1)
        pairs=physical+self.pair_adapter(features)
        # Shift before summing to avoid even numerical dependence on the
        # current target. The first prefix is the completed level-zero map.
        prefix=pairs[...,:-1,:].cumsum(-2)
        batch,height,width,levels,_=prefix.shape
        hidden=self.input_projection(prefix).permute(0,3,1,2,4).reshape(batch,levels,height*width,-1)
        hidden=hidden+self.position[:,None]+self.level+self.condition(cond)[:,None,None]
        mask=(~torch.eye(height*width,device=hidden.device,dtype=torch.bool)) if local_only else None
        hidden=self.encoder(hidden.flatten(0,1),mask=mask).reshape(batch,levels,height,width,-1).permute(0,2,3,1,4)
        if self.conditional_coefficients:
            vectors=codebook.dictionary.T[atoms[...,1:]]
            return dict(atom_logits=self.readout.backbone.classifier(hidden),
                        coeff_logits=self.conditional_coefficient_logits(hidden,vectors),hidden=hidden)
        return self.readout.classify(hidden,atoms[...,1:])

    def conditional_coefficient_logits(self, hidden, vectors, depth_index=None):
        atom=self.coefficient_atom_projection(vectors)
        refined=hidden+self.coefficient_refiner(torch.cat((hidden,atom,hidden*atom),-1))
        return self.readout.backbone.classify_coefficients(refined,depth_index=depth_index)

    @torch.no_grad()
    def sample_finer(self, tokens, codebook, cond, *, top_k=4096, temperature=1., generator=None, amp=True):
        """Keep supplied coarse pairs; commit each complete finer map once."""
        from src.joint_pair_topk import sample_joint_pair_topk
        if self.training: raise ValueError('sampling requires evaluation mode')
        if top_k==0 and temperature!=1.:
            raise ValueError('unfiltered joint draws use temperature 1; use joint top-k for tempering')
        tokens=tokens.clone()
        for depth in range(1,self.depth):
            with torch.autocast(device_type=tokens.device.type,dtype=torch.bfloat16,enabled=amp):
                output=self(tokens,codebook,cond)
            atom_logits=output['atom_logits'][...,depth-1,:].flatten(0,-2)
            if self.conditional_coefficients:
                hidden=output['hidden'][...,depth-1,:].flatten(0,-2)
                def scorer(rows,candidates):
                    vectors=codebook.dictionary.T[candidates]
                    with torch.autocast(device_type=tokens.device.type,dtype=torch.bfloat16,enabled=amp):
                        return self.conditional_coefficient_logits(
                            hidden[rows,None].expand(-1,candidates.shape[1],-1),vectors,depth_index=depth-1)
                if top_k==0:
                    a=torch.multinomial(atom_logits.float().softmax(-1),1,generator=generator).squeeze(-1)
                    coefficients=scorer(torch.arange(len(a),device=a.device),a[:,None])[:,0]
                    b=torch.multinomial(coefficients.float().softmax(-1),1,generator=generator).squeeze(-1)
                else:
                    a,b,_=sample_joint_pair_topk(atom_logits,scorer,top_k=top_k,
                                                temperature=temperature,generator=generator)
                tokens[...,2*depth]=a.reshape(tokens.shape[:-1])
                tokens[...,2*depth+1]=b.reshape(tokens.shape[:-1])+self.num_atoms
                continue
            bias=output['coefficient_bias'][...,depth-1,:].flatten(0,-2)
            coupling=output['coupling'][...,depth-1,:].flatten(0,-2)
            def scorer(rows,candidates):
                return bias[rows,None]+coupling[rows[:,None],candidates][...,None]*self.readout.bins
            if top_k==0:
                a=torch.multinomial(atom_logits.float().softmax(-1),1,generator=generator).squeeze(-1)
                coefficients=scorer(torch.arange(len(a),device=a.device),a[:,None])[:,0]
                b=torch.multinomial(coefficients.float().softmax(-1),1,generator=generator).squeeze(-1)
            else:
                a,b,_=sample_joint_pair_topk(atom_logits,scorer,top_k=top_k,
                                            temperature=temperature,generator=generator)
            tokens[...,2*depth]=a.reshape(tokens.shape[:-1])
            tokens[...,2*depth+1]=b.reshape(tokens.shape[:-1])+self.num_atoms
        return tokens
