"""A fresh generative hierarchy: AR coarse pairs, then complete finer maps."""
from dataclasses import replace

import torch
from torch import nn

from src.models.coarse_prefix_pair_refiner import CoarsePrefixPairRefiner
from src.models.offset_pair_prior import OffsetPairRQTransformer
from src.models.structured_pair_energy import StructuredPairEnergyRQ, structured_pair_cross_entropy
from src.models.typed_pair_rq import TypedPairCodebook


class HiddenPairRQ(OffsetPairRQTransformer):
    """The native RQ state is classified only by the structured joint head."""
    def classify_head_outputs(self, hidden):
        return {'shared_hidden':hidden}


def select_pair_targets(targets, first, last=None):
    depth=targets.candidate_atoms.shape[-2]
    last=depth if last is None else last
    return replace(targets,tokens=targets.tokens[...,2*first:2*last],
        candidate_atoms=targets.candidate_atoms[...,first:last,:],
        candidate_bins=targets.candidate_bins[...,first:last,:],
        pair_weights=targets.pair_weights[...,first:last,:],
        coefficient_weights_given_atom=targets.coefficient_weights_given_atom[...,first:last,:],
        residual_energies=targets.residual_energies[...,first:last])


def joint_hierarchy_loss(outputs, targets, bins):
    coarse,fine=outputs
    coarse_loss,a0,c0=structured_pair_cross_entropy(coarse,select_pair_targets(targets,0,1),bins)
    fine_loss,af,cf=structured_pair_cross_entropy(fine,select_pair_targets(targets,1),bins)
    depth=targets.candidate_atoms.shape[-2]
    loss=(coarse_loss+(depth-1)*fine_loss)/depth
    return loss,torch.cat((a0,af)),torch.cat((c0,cf))


class CoarseFinePairPrior(nn.Module):
    def __init__(self, coarse, refiner):
        super().__init__()
        self.coarse,self.refiner=coarse,refiner

    def forward(self, tokens, codebook, cond, coarse_codebook, amp=False):
        coarse=self.coarse(tokens[...,:2],coarse_codebook,cond,amp=amp)
        fine=self.refiner(tokens,codebook,cond)
        return coarse,fine

    @torch.no_grad()
    def sample(self, batch, codebook, coarse_codebook, cond, *, generator=None, amp=True):
        if self.training:raise ValueError('sampling requires evaluation mode')
        first=self.coarse.sample_joint_sparse(batch,coarse_codebook,cond,top_k=0,
            temperature=1.,top_p=1.,generator=generator,amp=amp)
        tokens=codebook.empty_tokens(batch,*self.coarse.backbone.block_size[:2]);tokens[...,:2]=first
        return self.refiner.sample_finer(tokens,codebook,cond,top_k=0,
            temperature=1.,generator=generator,amp=amp)


def build_scratch_pair_prior(codebook, *, model_preset='imagenet-1400m'):
    from src.training.rqtransformer import build_model
    backbone=build_model(codebook.vocab_size,codebook.num_atoms,
        coeff_vocab_size=codebook.coeff_vocab_size,sparsity_level=1,
        physical_pair_context=True,efficient_pair_backbone=True,
        compound_mask_seen_atoms_training=False,compound_depth_specific_coeff_heads=True,
        coefficient_history_layers=1,coefficient_history_width=64,coefficient_history_heads=4,
        coefficient_history_dropout=0.,model_preset=model_preset)
    backbone.__class__=HiddenPairRQ
    # The structured word head replaces these categorical coefficient paths.
    # Remove their parameters rather than silently keeping an unused optimizer.
    for name in ('coeff_fusion','coeff_atom_proj','coefficient_history_decoder'):
        delattr(backbone,name)
    coarse=StructuredPairEnergyRQ(backbone,codebook.dictionary,codebook.coeff_bins)
    refiner=CoarsePrefixPairRefiner(codebook.dictionary.shape[0],codebook.dictionary,codebook.coeff_bins,
                                    depth=codebook.depth)
    return CoarseFinePairPrior(coarse,refiner)
