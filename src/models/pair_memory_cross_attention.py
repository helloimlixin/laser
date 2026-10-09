"""Two prediction queries over a causal memory of completed sparse pairs."""
import math

import torch
from torch import nn

from src.models.coefficient_cross_attention import PairCrossAttentionBlock
from src.models.compound_full_history import CausalBlock


class QueryMLPBlock(nn.Module):
    """Capacity control: 12*w*w weights, matching a cross-attention block."""
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 6 * width), nn.GELU(),
                                 nn.Linear(6 * width, width))

    def forward(self, query, memory, *, cached=False):
        return query + self.ffn(self.norm(query))


class PairMemoryQueries(nn.Module):
    """Preserve pair fields; ask separate atom and atom-conditioned value queries.

    Dense memory slot t contains BOS (t=0) or completed pair t-1. Its encoder
    and all cross reads are causal. The cache appends that slot once before
    predicting an atom, and reuses it for the corresponding coefficient.
    Both output projections start at zero to preserve the old checkpoint.
    """
    def __init__(self, hidden_dim, input_dim, block_size, *, width=512, heads=8,
                 memory_layers=1, query_layers=2, mode='cross'):
        super().__init__()
        if mode not in {'cross', 'mlp'}:
            raise ValueError('query mode must be cross or mlp')
        if width % heads or min(width, heads, memory_layers, query_layers) < 1:
            raise ValueError('positive dimensions and width divisible by heads required')
        self.mode = mode
        self.block_size = tuple(block_size)
        self.events = math.prod(block_size)
        self.atom_norm = nn.LayerNorm(input_dim)
        self.coefficient_norm = nn.LayerNorm(input_dim)
        self.contribution_norm = nn.LayerNorm(input_dim)
        self.memory_atom = nn.Linear(input_dim, width, bias=False)
        self.memory_coefficient = nn.Linear(input_dim, width, bias=False)
        self.memory_contribution = nn.Linear(input_dim, width, bias=False)
        # Keep signed physical magnitude outside the normalized vector fields.
        self.memory_scalar = nn.Linear(1, width, bias=False)
        self.row_position = nn.Parameter(torch.empty(block_size[0], width))
        self.column_position = nn.Parameter(torch.empty(block_size[1], width))
        self.depth_position = nn.Parameter(torch.empty(block_size[2], width))
        self.bos = nn.Parameter(torch.empty(1, 1, width))
        self.memory_blocks = nn.ModuleList([
            CausalBlock(width, heads, 0., self.events) for _ in range(memory_layers)])
        self.hidden_projection = nn.Linear(hidden_dim, width)
        self.prefix_projection = nn.Linear(input_dim, width, bias=False)
        self.coefficient_atom = nn.Linear(input_dim, width, bias=False)
        self.coefficient_query_norm = nn.LayerNorm(width)
        block = (lambda: PairCrossAttentionBlock(width, heads)) if mode == 'cross' else (lambda: QueryMLPBlock(width))
        self.atom_blocks = nn.ModuleList([block() for _ in range(query_layers)])
        self.coefficient_blocks = nn.ModuleList([block() for _ in range(query_layers)])
        self.atom_output_norm = nn.LayerNorm(width)
        self.coefficient_output_norm = nn.LayerNorm(width)
        self.atom_output = nn.Linear(width, hidden_dim, bias=False)
        self.coefficient_output = nn.Linear(width, hidden_dim, bias=False)
        self.apply(self._initialize)
        for value in [self.row_position, self.column_position, self.depth_position, self.bos]:
            nn.init.normal_(value, std=.02)
        nn.init.zeros_(self.atom_output.weight)
        nn.init.zeros_(self.coefficient_output.weight)
        self.reset_cache()

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_cache(self):
        self._next_event = 0
        self._pending = None
        for block in self.memory_blocks:
            block.reset_cache()
        for block in [*self.atom_blocks, *self.coefficient_blocks]:
            if isinstance(block, PairCrossAttentionBlock):
                block._kv = None

    def positions(self):
        return (self.row_position[:, None, None] + self.column_position[None, :, None]
                + self.depth_position[None, None, :]).reshape(1, self.events, -1)

    def pair_features(self, atoms, coefficients, contributions, scalars):
        return (self.memory_atom(self.atom_norm(atoms))
                + self.memory_coefficient(self.coefficient_norm(coefficients))
                + self.memory_contribution(self.contribution_norm(contributions))
                + self.memory_scalar(scalars)) * .5

    def encode_memory(self, atoms, coefficients, contributions, scalars, *, event=None):
        positions = self.positions()
        if event is None:
            records = self.pair_features(atoms, coefficients, contributions, scalars) + positions
            memory = torch.cat((self.bos.expand(records.shape[0], -1, -1), records[:, :-1]), dim=1)
        else:
            if self.training or torch.is_grad_enabled() or event != self._next_event or self._pending is not None:
                raise ValueError('cached atom query must follow the previous completed coefficient')
            if event == 0:
                memory = self.bos.expand(atoms.shape[0], -1, -1)
            else:
                memory = self.pair_features(atoms, coefficients, contributions, scalars) + positions[:, event-1:event]
        for block in self.memory_blocks:
            memory = block(memory, cached=event is not None)
        return memory

    def atom_query(self, hidden, prefix, memory, *, event=None):
        start = 0 if event is None else event
        query = (self.hidden_projection(hidden) + self.prefix_projection(prefix)
                 + memory + self.positions()[:, start:start + hidden.shape[1]]) * .5
        for block in self.atom_blocks:
            query = block(query, memory, cached=event is not None)
        refined = hidden + self.atom_output(self.atom_output_norm(query))
        if event is not None:
            self._pending = (query, prefix, memory, event)
        return refined, query

    def coefficient_query(self, query, atom, prefix, memory, *, event=None):
        query = (self.coefficient_query_norm(query) + self.coefficient_atom(self.atom_norm(atom))
                 + self.prefix_projection(prefix)) / math.sqrt(3.)
        for block in self.coefficient_blocks:
            query = block(query, memory, cached=event is not None)
        residual = self.coefficient_output(self.coefficient_output_norm(query))
        if event is not None:
            if self._pending is None or self._pending[-1] != event:
                raise ValueError('coefficient query requires its corresponding atom query')
            self._pending = None
            self._next_event += 1
        return residual

    def forward(self, hidden, current_atom, prefix, atoms, coefficients, contributions, scalars):
        memory = self.encode_memory(atoms, coefficients, contributions, scalars)
        refined, query = self.atom_query(hidden, prefix, memory)
        coefficient_residual = self.coefficient_query(query, current_atom, prefix, memory)
        return refined, coefficient_residual
