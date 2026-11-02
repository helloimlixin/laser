"""Joint atom/sign classification with an exact conditional magnitude likelihood."""
import torch
from torch import nn


def split_coefficient_bins(bins, coefficient_bins):
    if coefficient_bins <= 0 or coefficient_bins % 2:
        raise ValueError('A symmetric coefficient vocabulary requires an even bin count')
    half = coefficient_bins//2
    signs = (bins >= half).long()
    magnitudes = torch.where(signs.bool(), bins-half, half-1-bins)
    return signs, magnitudes


def merge_coefficient_bins(signs, magnitudes, coefficient_bins):
    if coefficient_bins <= 0 or coefficient_bins % 2:
        raise ValueError('A symmetric coefficient vocabulary requires an even bin count')
    half = coefficient_bins//2
    return torch.where(signs.bool(), half+magnitudes, half-1-magnitudes)


class SignedJointLinear(nn.Module):
    def __init__(self, hidden_width, num_atoms, coefficient_bins, context_width=256):
        super().__init__()
        if coefficient_bins <= 0 or coefficient_bins % 2:
            raise ValueError('Signed coefficients require an even number of bins')
        self.num_atoms = int(num_atoms)
        self.signed_atom_classes = 2*self.num_atoms
        self.coefficient_bins = int(coefficient_bins)
        self.magnitude_bins = self.coefficient_bins//2
        self.signed_atom = nn.Linear(hidden_width, self.signed_atom_classes)
        self.context = nn.Linear(hidden_width, context_width)
        self.signed_atom_context = nn.Embedding(self.signed_atom_classes, context_width)
        self.magnitude = nn.Sequential(nn.LayerNorm(context_width),
            nn.Linear(context_width, context_width), nn.GELU(),
            nn.Linear(context_width, self.magnitude_bins))

    def forward(self, hidden):
        return torch.cat((self.signed_atom(hidden), self.context(hidden)), -1)

    def magnitude_logits(self, features, signed_atoms):
        with torch.autocast(device_type=features.device.type, dtype=torch.float16,
                            enabled=features.is_cuda):
            context = features[..., self.signed_atom_classes:] + self.signed_atom_context(signed_atoms)
            return self.magnitude(context)


def install_signed_joint_head(model, num_atoms, coefficient_bins, context_width=256):
    if not model.config.shared_cls_emb:
        raise ValueError('The joint head requires the shared native classifier')
    head = SignedJointLinear(model.classifier.linear.in_features, num_atoms,
                             coefficient_bins, context_width)
    head.apply(model._init_weights)
    model.classifier.linear = head
    model.classifier.logit_mask = nn.Identity()
    return model


def exclude_previous_signed_atoms(logits, atoms):
    masked = logits.clone()
    for depth in range(1, atoms.shape[-1]):
        previous = atoms[..., :depth]
        masked[..., depth, :].scatter_(-1, 2*previous, -torch.inf)
        masked[..., depth, :].scatter_(-1, 2*previous+1, -torch.inf)
    return masked


def signed_joint_cross_entropy(head, features, targets):
    atoms = targets['atoms']
    coefficient_p = targets['coefficient_probabilities']
    half = head.magnitude_bins
    # Negative original bins run from largest to smallest magnitude.
    p = torch.stack((coefficient_p[..., :half].flip(-1), coefficient_p[..., half:]), -2)
    sign_p = p.sum(-1)
    signed_ids = 2*atoms[..., None] + torch.arange(2, device=atoms.device)
    logits = exclude_previous_signed_atoms(features[..., :head.signed_atom_classes], atoms)
    logp = logits.float().log_softmax(-1).gather(-1, signed_ids)
    signed_ce = -(sign_p*logp).sum(-1).mean()
    # Both signs receive their exact soft-label probability, including rare
    # posterior mass across zero. No hard-sign approximation or extra loss.
    magnitude_logp = head.magnitude_logits(features.unsqueeze(-2), signed_ids).float().log_softmax(-1)
    magnitude_ce = -(p*magnitude_logp).sum((-1, -2)).mean()
    loss = signed_ce + magnitude_ce
    with torch.no_grad():
        atom_logp = logp.logsumexp(-1)
        atom_ce = -atom_logp.mean()
        sign_ce = -(sign_p*(logp-atom_logp[..., None])).sum(-1).mean()
        hard_signs, _ = split_coefficient_bins(targets['hard_codes']%head.coefficient_bins,
                                             head.coefficient_bins)
        conditional_sign = logp.softmax(-1).gather(-1, hard_signs[..., None]).squeeze(-1)
        diagnostics = dict(atom_ce=atom_ce, sign_ce=sign_ce,
            signed_atom_ce=signed_ce.detach(), magnitude_ce=magnitude_ce.detach(),
            joint_ce_per_scalar_event=loss.detach()/2,
            correct_sign_probability=conditional_sign.mean())
        for depth in range(atoms.shape[-1]):
            diagnostics[f'correct_sign_probability_depth{depth+1}'] = conditional_sign[..., depth].mean()
    return loss, diagnostics
