"""Atom-conditioned coefficient queries over strictly earlier sparse pairs."""
import torch
from torch import nn
from torch.nn import functional as F


class PairCrossAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.projection = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self._kv = None

    def forward(self, query, memory, *, cached=False):
        batch, length, width = query.shape
        heads = self.heads
        q = self.query(self.query_norm(query)).reshape(
            batch, length, heads, width // heads).transpose(1, 2)
        k, v = self.key_value(self.memory_norm(memory)).reshape(
            batch, memory.shape[1], 2, heads, width // heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        if cached:
            if self.training or torch.is_grad_enabled() or length != 1 or memory.shape[1] != 1:
                raise ValueError('cached cross-attention requires one event in evaluation without gradients')
            if self._kv is not None:
                k = torch.cat((self._kv[0], k), dim=-2)
                v = torch.cat((self._kv[1], v), dim=-2)
            self._kv = (k, v)
        elif length != memory.shape[1]:
            raise ValueError('dense queries and shifted pair memory must have equal lengths')
        # Memory position i holds pair i-1, with BOS at zero. Dense diagonal
        # attention is therefore strictly earlier-pair attention. Every cached
        # key is legal; is_causal=True would incorrectly retain only BOS there.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=not cached)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        query = query + self.projection(attended)
        return query + self.ffn(self.ffn_norm(query))


class CoefficientPairCrossAttention(nn.Module):
    """Zero-initialized residual; complete raster/depth history, without dropout.

    Query i uses its atom-predictor state and current dictionary atom. Keys and
    values are individual completed pairs j<i. The caller shifts those pairs
    once before passing previous_pair; current and future coefficients are
    never visible. No dropout is added, preserving the baseline RNG trajectory.
    """
    def __init__(self, hidden_dim, input_dim, *, width=512, layers=2, heads=8,
                 max_events=256):
        super().__init__()
        if min(width, layers, heads, max_events) < 1 or width % heads:
            raise ValueError('positive dimensions and a width divisible by heads are required')
        self.hidden_projection = nn.Linear(hidden_dim, width)
        self.atom_projection = nn.Linear(input_dim, width, bias=False)
        self.memory_projection = nn.Linear(input_dim, width)
        self.query_position = nn.Parameter(torch.empty(1, max_events, width))
        self.memory_position = nn.Parameter(torch.empty(1, max_events, width))
        self.blocks = nn.ModuleList([PairCrossAttentionBlock(width, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, hidden_dim, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.query_position, std=.02)
        nn.init.normal_(self.memory_position, std=.02)
        nn.init.zeros_(self.output.weight)
        self.reset_cache()

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_cache(self):
        self._next_event = 0
        for block in self.blocks:
            block._kv = None

    def forward(self, hidden, atom, previous_pair, prefix=None, *, event=None):
        # The shared adapter also supplies a local physical prefix. This
        # experiment isolates cross-attention and does not add that extra path.
        cached = event is not None
        start = event if cached else 0
        if cached and event != self._next_event:
            raise ValueError(f'expected cached event {self._next_event}, got {event}')
        length = hidden.shape[1]
        if start + length > self.query_position.shape[1]:
            raise ValueError('pair sequence exceeds cross-attention capacity')
        query = (self.hidden_projection(hidden) + self.atom_projection(atom)
                 + self.query_position[:, start:start + length])
        memory = (self.memory_projection(previous_pair)
                  + self.memory_position[:, start:start + length])
        for block in self.blocks:
            query = block(query, memory, cached=cached)
        if cached:
            self._next_event += length
        return self.output(self.output_norm(query))
