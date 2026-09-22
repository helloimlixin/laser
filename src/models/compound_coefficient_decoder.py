"""Causal history decoder for atom-conditioned sparse coefficients."""
import torch
from torch import nn
from torch.nn import functional as F


class HistoryDecoderBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        if width % heads:
            raise ValueError('decoder width must be divisible by head count')
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))
        self.dropout = nn.Dropout(dropout)
        self._kv = None

    def forward(self, x, *, cached=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, width // self.heads,
        ).permute(2, 0, 3, 1, 4).unbind(0)
        if cached:
            if self.training or length != 1 or torch.is_grad_enabled():
                raise ValueError('cached decoding requires one event in evaluation without gradients')
            if self._kv is not None:
                k = torch.cat((self._kv[0], k), dim=-2)
                v = torch.cat((self._kv[1], v), dim=-2)
            self._kv = (k, v)
        # A cached query is the final position and may see its entire prefix.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=not cached)
        y = y.transpose(1, 2).reshape(batch, length, width)
        x = x + self.dropout(self.projection(y))
        return x + self.dropout(self.mlp(self.norm2(x)))


class CompoundCoefficientHistoryDecoder(nn.Module):
    """One decoder position per pair; output is initially an exact zero residual.

    Inputs are the atom-predictor state, current atom vector, previous completed
    pair embedding, and the current site's strictly earlier reconstruction.
    Attention spans the complete raster/depth event sequence. No coefficient
    from the current event may enter these inputs.
    """
    def __init__(self, hidden_dim, input_dim, *, width=512, layers=2, heads=8,
                 dropout=0.1, max_events=256):
        super().__init__()
        if layers < 1 or max_events < 1:
            raise ValueError('positive layer and event counts are required')
        self.hidden_projection = nn.Linear(hidden_dim, width)
        self.atom_projection = nn.Linear(input_dim, width, bias=False)
        self.pair_projection = nn.Linear(input_dim, width, bias=False)
        self.prefix_projection = nn.Linear(input_dim, width, bias=False)
        self.position = nn.Parameter(torch.empty(1, max_events, width))
        self.blocks = nn.ModuleList([HistoryDecoderBlock(width, heads, dropout)
                                     for _ in range(layers)])
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, hidden_dim, bias=False)
        self.apply(self._initialize)
        nn.init.normal_(self.position, std=0.02)
        nn.init.zeros_(self.output.weight)
        self.reset_cache()

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_cache(self):
        self._next_event = 0
        for block in self.blocks:
            block._kv = None

    def forward(self, hidden, atom, previous_pair, prefix, *, event=None):
        cached = event is not None
        if cached:
            if event != self._next_event:
                raise ValueError(f'expected cached event {self._next_event}, got {event}')
            start = event
        else:
            start = 0
        length = hidden.shape[1]
        if start + length > self.position.shape[1]:
            raise ValueError('coefficient sequence exceeds decoder event capacity')
        x = (self.hidden_projection(hidden) + self.atom_projection(atom)
             + self.pair_projection(previous_pair) + self.prefix_projection(prefix)
             + self.position[:, start:start + length])
        for block in self.blocks:
            x = block(x, cached=cached)
        if cached:
            self._next_event += length
        return self.output(self.output_norm(x))
