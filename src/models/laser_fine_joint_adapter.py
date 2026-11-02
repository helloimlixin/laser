"""A normalized joint categorical head without a Cartesian output matrix."""
import torch
from torch import nn
from torch.nn import functional as F


class ConditionalJointLinear(nn.Module):
    def __init__(self, atom_linear, coefficient_bins, context_width=256):
        super().__init__()
        self.atom = atom_linear
        self.num_atoms = atom_linear.out_features
        self.coefficient_bins = int(coefficient_bins)
        self.context_width = int(context_width)
        self.context = nn.Linear(atom_linear.in_features, context_width)
        self.atom_context = nn.Embedding(self.num_atoms, context_width)
        self.coefficient = nn.Sequential(nn.LayerNorm(context_width),
            nn.Linear(context_width, context_width), nn.GELU(),
            nn.Linear(context_width, self.coefficient_bins))

    def forward(self, hidden):
        return torch.cat((self.atom(hidden), self.context(hidden)), -1)

    def coefficient_logits(self, features, atoms):
        with torch.autocast(device_type=features.device.type, dtype=torch.float16,
                            enabled=features.is_cuda):
            context = features[..., self.num_atoms:] + self.atom_context(atoms)
            return self.coefficient(context)


def install_conditional_joint_head(model, coefficient_bins, context_width=256):
    if not model.config.shared_cls_emb:
        raise ValueError('The joint head requires the shared native classifier')
    head = ConditionalJointLinear(model.classifier.linear, coefficient_bins, context_width)
    for module in (head.context, head.atom_context, head.coefficient):
        module.apply(model._init_weights)
    model.classifier.linear = head
    model.classifier.logit_mask = nn.Identity()
    return model


def exclude_previous_atoms(logits, atoms):
    masked = logits.clone()
    for depth in range(1, atoms.shape[-1]):
        masked[..., depth, :].scatter_(-1, atoms[..., :depth], -torch.inf)
    return masked


def joint_cross_entropy(head, features, targets):
    atoms = targets['atoms']
    atom_logits = exclude_previous_atoms(features[..., :head.num_atoms], atoms)
    atom_ce = F.cross_entropy(atom_logits.float().reshape(-1, head.num_atoms), atoms.reshape(-1))
    coefficient_logits = head.coefficient_logits(features, atoms).float()
    coefficient_ce = -(targets['coefficient_probabilities'] *
                       coefficient_logits.log_softmax(-1)).sum(-1).mean()
    # One joint event: -log p(a,b) = -log p(a) - log p(b|a).
    return atom_ce + coefficient_ce
