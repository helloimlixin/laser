"""BAR-inspired masked-bit prediction for an unchanged scalar quantizer.

The architecture follows the conditional masked-token idea, not BAR's tokenizer.
Bin IDs use ordinary little-endian binary encoding. A mask always replaces the
unknown bit value before the network sees it.
"""

import torch
from torch import nn
from torch.nn import functional as F


def ids_to_bits(ids, bits=11):
    return ((ids.long().unsqueeze(-1) >> torch.arange(bits, device=ids.device)) & 1).float()


def bits_to_ids(values):
    weights = 1 << torch.arange(values.shape[-1], device=values.device)
    return (values.long() * weights).sum(-1)


class ConditionedBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))
        self.up = nn.Linear(width, 4 * width)
        self.down = nn.Linear(2 * width, width)

    def forward(self, x, condition):
        shift, scale, gate = self.modulation(condition).chunk(3, -1)
        u, v = self.up(self.norm(x) * (1 + scale) + shift).chunk(2, -1)
        return x + torch.tanh(gate) * self.down(F.silu(u) * v)


class MaskedCoefficientHead(nn.Module):
    def __init__(self, context_dim=1024, width=512, layers=3, bits=11, depth=4):
        super().__init__()
        self.bits = bits
        self.context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, width))
        self.depth = nn.Embedding(depth, width)
        self.input = nn.Linear(2 * bits, width)
        self.blocks = nn.ModuleList([ConditionedBlock(width) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, bits))

    def forward(self, context, values, known, depth):
        # Unknown values have exactly zero influence, including during training.
        visible = torch.where(known, values * 2 - 1, torch.zeros_like(values))
        x = self.input(torch.cat((visible, known.to(values.dtype)), -1))
        condition = self.context(context) + self.depth(depth)
        for block in self.blocks:
            x = block(x, condition)
        return self.output(x)

    def loss(self, context, targets, depth, generator=None):
        values = ids_to_bits(targets, self.bits)
        # Uniform number of unknown bits in [1, bits], then uniform positions.
        count = torch.randint(1, self.bits + 1, targets.shape, device=targets.device,
                              generator=generator)
        ranks = torch.rand(values.shape, device=targets.device, generator=generator).argsort(-1).argsort(-1)
        unknown = ranks < count.unsqueeze(-1)
        logits = self(context, values, ~unknown, depth)
        loss = F.binary_cross_entropy_with_logits(logits.float(), values, reduction="none")
        per_token = (loss * unknown).sum(-1) / count
        return per_token.mean(), {"masked_bit_bce": per_token.detach().mean()}

    @torch.no_grad()
    def sample(self, context, depth, schedule=(2, 3, 3, 3), temperature=1.0,
               greedy=False, generator=None, return_trace=False):
        if sum(schedule) != self.bits or any(n <= 0 for n in schedule):
            raise ValueError("unmasking schedule must partition all bits")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        values = torch.zeros((*context.shape[:-1], self.bits), device=context.device)
        known = torch.zeros_like(values, dtype=torch.bool)
        trace = []
        for count in schedule:
            if return_trace:
                trace.append((values.clone(), known.clone()))
            logits = self(context, values, known, depth).float() / temperature
            probability = logits.sigmoid()
            proposal = (probability >= .5).float() if greedy else torch.bernoulli(probability, generator=generator)
            # Select confident unknown positions. Previously revealed bits stay fixed.
            confidence = torch.maximum(probability, 1 - probability).masked_fill(known, -1)
            locations = confidence.topk(count, -1).indices
            reveal = torch.zeros_like(known).scatter_(-1, locations, True)
            values = torch.where(reveal, proposal, values)
            known |= reveal
        ids = bits_to_ids(values)
        return (ids, trace) if return_trace else ids
