"""Pool evidence about an atom without splitting a joint categorical event."""
import torch
from torch import nn


class AtomSharedJointLinear(nn.Module):
    """Add one trainable atom score to every coefficient token for that atom.

    The existing joint classifier retains full capacity. A zero-initialized
    branch pools gradients across coefficient levels and preserves all initial
    joint logits. Code zero has its own score. Loss and sampling still see one
    distribution over the original joint integer vocabulary.
    """
    def __init__(self, original, num_atoms, coefficient_levels):
        super().__init__()
        if min(num_atoms, coefficient_levels) < 1:
            raise ValueError('Positive atom and coefficient counts required')
        if original.out_features != 1+num_atoms*coefficient_levels:
            raise ValueError('Expected zero followed by equal coefficient groups')
        self.base = original
        self.num_atoms, self.coefficient_levels = num_atoms, coefficient_levels
        self.in_features, self.out_features = original.in_features, original.out_features
        self.atom_delta = nn.Linear(self.in_features, num_atoms+1,
            device=original.weight.device, dtype=original.weight.dtype)
        nn.init.zeros_(self.atom_delta.weight)
        nn.init.zeros_(self.atom_delta.bias)

    def forward(self, hidden):
        logits = self.base(hidden)
        atoms = self.atom_delta(hidden)
        # Broadcasting avoids another full joint-vocabulary activation.
        logits[..., 1:].view(*logits.shape[:-1], self.num_atoms, self.coefficient_levels).add_(atoms[..., :self.num_atoms, None])
        logits[..., 0].add_(atoms[..., self.num_atoms])
        return logits


def add_atom_shared_joint_classifier(model, quantizer):
    atoms = quantizer.dictionary.shape[1]
    levels, remainder = divmod(quantizer.vocab_size-1, atoms)
    expected = torch.cat((torch.zeros(1, dtype=torch.long, device=quantizer.atom_ids.device),
        torch.arange(atoms, device=quantizer.atom_ids.device).repeat_interleave(levels)))
    if remainder or not torch.equal(quantizer.atom_ids, expected.expand_as(quantizer.atom_ids)):
        raise ValueError('Atom IDs must have the same uniform grouping at every depth')
    model.classifier.linear = AtomSharedJointLinear(model.classifier.linear, atoms, levels)
