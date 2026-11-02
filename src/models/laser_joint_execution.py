"""Execution-only alignment for a finite joint vocabulary.

The native classifier's parameters, valid logits and softmax domain are kept.
Temporary zero rows align the matrix multiplication; their outputs are sliced
away before the upstream logit mask, loss or sampler sees them.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AlignedJointLinear(nn.Module):
    def __init__(self, original, multiple=64):
        super().__init__()
        if not isinstance(original, nn.Linear) or multiple < 1:
            raise ValueError('Expected a linear classifier and positive alignment')
        # Reuse the same Parameter objects, including existing optimizer refs.
        self.weight = original.weight
        self.bias = original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.multiple = int(multiple)
        self.padding = (-self.out_features) % self.multiple

    def forward(self, inputs):
        if not self.padding:
            return F.linear(inputs, self.weight, self.bias)
        weight = F.pad(self.weight, (0, 0, 0, self.padding))
        bias = None if self.bias is None else F.pad(self.bias, (0, self.padding))
        return F.linear(inputs, weight, bias)[..., :self.out_features]


def align_joint_classifier(model, multiple=64):
    linear = model.classifier.linear
    if isinstance(linear, AlignedJointLinear):
        assert linear.multiple == multiple
        return
    model.classifier.linear = AlignedJointLinear(linear, multiple)
