"""Flat causal GPT over completed LASER (atom, coefficient-bin) events.

The integer a * B + b is transport only. There is no joint codebook or
N*B embedding/classifier: p(a,b|past) = p(a|past) p(b|past,a).
"""
from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .transformer_core import TransformerBlock, init_transformer_module_weights
from .rqtransformer.transformers import sample_from_logits


@dataclass(frozen=True)
class CompoundGPTConfig:
    block_size: tuple[int, int, int] = (8, 8, 4)
    num_atoms: int = 16384
    coeff_vocab_size: int = 512
    dictionary_dim: int = 256
    num_classes: int = 1000
    width: int = 2048
    heads: int = 32
    layers: int = 26
    dropout: float = 0.1
    mask_seen_atoms_training: bool = True
    attention_dropout: float | None = None
    embedding_dropout: float | None = None
    ffn_intermediate_dropout: float | None = None
    coefficient_conditioner_layers: int = 0
    coefficient_conditioner_width: int = 512
    coefficient_conditioner_heads: int = 8


class CausalCoefficientConditioner(nn.Module):
    """Causal refinement from history, current atom and the previous full pair.

    A zero output projection makes insertion into a trained prior initially
    preserve its predictions. No current/future coefficient enters this path.
    """
    def __init__(self, context_width, width, heads, layers, depth, dropout,
                 attention_dropout=0., ffn_intermediate_dropout=0.):
        super().__init__()
        if layers < 1 or width < 1 or heads < 1 or width % heads:
            raise ValueError('invalid coefficient conditioner geometry')
        self.depth = depth
        self.context_projection = nn.Linear(context_width, width)
        self.atom_projection = nn.Linear(context_width, width)
        self.history_projection = nn.Linear(context_width, width)
        self.field_norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.fusion = nn.Sequential(nn.Linear(3 * width, width), nn.GELU())
        self.depth_embedding = nn.Embedding(depth, width)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads, 4 * width, dropout,
                attn_bias=True, out_proj_bias=True,
                attention_dropout=attention_dropout,
                ffn_intermediate_dropout=ffn_intermediate_dropout)
            for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        # Bin count is set by the parent when constructing the output head.
        self.output = None
        self.apply(init_transformer_module_weights)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=.02 / math.sqrt(2 * layers))
            nn.init.normal_(block.ffn[3].weight, std=.02 / math.sqrt(2 * layers))

    def set_output_bins(self, bins):
        self.output = nn.Linear(self.final_norm.normalized_shape[0], bins)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _inputs(self, context, atom, history, positions):
        fields = (self.context_projection(context), self.atom_projection(atom),
                  self.history_projection(history))
        normalized = [norm(field) for norm, field in zip(self.field_norms, fields)]
        return self.fusion(torch.cat(normalized, -1)) + self.depth_embedding(positions % self.depth)

    def forward(self, context, atom, history):
        positions = torch.arange(context.shape[1], device=context.device)
        hidden = self._inputs(context, atom, history, positions)
        for block in self.blocks:
            hidden = block(hidden)[0]
        return self.output(self.final_norm(hidden))

    def step(self, context, atom, history, position, cache=None):
        positions = torch.tensor([position], device=context.device)
        hidden = self._inputs(context[:, None], atom[:, None], history[:, None], positions)
        updated = []
        for index, block in enumerate(self.blocks):
            hidden, state = block(hidden, kv_cache=None if cache is None else cache[index])
            updated.append(state)
        return self.output(self.final_norm(hidden))[:, 0], updated


class CompoundGPT(nn.Module):
    def __init__(self, config: CompoundGPTConfig):
        super().__init__()
        if (len(config.block_size) != 3 or min(config.block_size) < 1
                or config.num_atoms < config.block_size[-1]
                or config.coeff_vocab_size < 2 or config.layers < 1
                or config.width % config.heads or not 0 <= config.dropout < 1):
            raise ValueError("invalid compound GPT configuration")
        for value in (config.attention_dropout, config.embedding_dropout, config.ffn_intermediate_dropout):
            if value is not None and not 0 <= value < 1:
                raise ValueError('dropout probabilities must be in [0,1)')
        if config.coefficient_conditioner_layers < 0:
            raise ValueError('coefficient conditioner layers must be nonnegative')
        self.config = config
        self.block_size = tuple(config.block_size)
        self.events = math.prod(self.block_size)
        self.num_atoms = config.num_atoms
        self.coeff_vocab_size = config.coeff_vocab_size
        self.capture_shared_hidden = False
        self.activation_checkpointing = False
        w = config.width
        # Atom identity remains visible even when a late-depth coefficient is
        # tiny. Physical contributions additionally preserve dictionary geometry.
        self.atom_embedding = nn.Embedding(config.num_atoms, w)
        self.coefficient_embedding = nn.Embedding(config.coeff_vocab_size, w)
        self.physical_projection = nn.Linear(config.dictionary_dim, w, bias=False)
        self.class_embedding = nn.Embedding(config.num_classes, w)
        self.position_embedding = nn.Embedding(self.events, w)
        self.embedding_dropout = nn.Dropout(config.dropout if config.embedding_dropout is None else config.embedding_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(w, config.heads, 4 * w, config.dropout,
                             attn_bias=True, out_proj_bias=True,
                             attention_dropout=config.attention_dropout,
                             ffn_intermediate_dropout=config.ffn_intermediate_dropout)
            for _ in range(config.layers)
        ])
        self.final_norm = nn.LayerNorm(w)
        self.atom_classifier = nn.Linear(w, config.num_atoms)
        self.coefficient_atom_projection = nn.Linear(config.dictionary_dim, w, bias=False)
        self.coefficient_atom_norm = nn.LayerNorm(w)
        self.coefficient_classifier = nn.Sequential(
            nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(),
            nn.Linear(w, config.coeff_vocab_size),
        )
        self.apply(init_transformer_module_weights)
        # Standard GPT residual scaling controls variance through a deep stack.
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=.02 / math.sqrt(2 * config.layers))
            nn.init.normal_(block.ffn[3].weight, std=.02 / math.sqrt(2 * config.layers))
        # Append the new module after all historical parameters, preserving
        # their optimizer order and checkpoint names.
        self.coefficient_conditioner = None
        if config.coefficient_conditioner_layers:
            self.coefficient_conditioner = CausalCoefficientConditioner(
                w, config.coefficient_conditioner_width, config.coefficient_conditioner_heads,
                config.coefficient_conditioner_layers, self.block_size[-1], config.dropout,
                0. if config.attention_dropout is None else config.attention_dropout,
                0. if config.ffn_intermediate_dropout is None else config.ffn_intermediate_dropout)
            self.coefficient_conditioner.set_output_bins(config.coeff_vocab_size)

    def unpack(self, packed):
        return packed.long() // self.coeff_vocab_size, packed.long() % self.coeff_vocab_size

    def pair_embeddings(self, atoms, coefficients, model_aux, depths):
        vectors = model_aux.dictionary.t()[atoms]
        values = model_aux.coeff_bins[coefficients] * model_aux.coeff_scales[depths]
        physical = self.physical_projection(vectors * values.unsqueeze(-1))
        return self.atom_embedding(atoms) + self.coefficient_embedding(coefficients) + physical

    def _coefficient_base(self, hidden, atoms, model_aux):
        atom = self.coefficient_atom_norm(
            self.atom_embedding(atoms)
            + self.coefficient_atom_projection(model_aux.dictionary.t()[atoms]))
        return self.coefficient_classifier(hidden + atom), atom

    def coefficient_logits(self, hidden, atoms, model_aux, history=None):
        base, atom = self._coefficient_base(hidden, atoms, model_aux)
        if self.coefficient_conditioner is None:
            return base
        if history is None or history.shape != hidden.shape:
            raise ValueError('coefficient refinement requires the shifted completed-pair history')
        batch, width = hidden.shape[0], hidden.shape[-1]
        residual = self.coefficient_conditioner(hidden.reshape(batch, -1, width),
            atom.reshape(batch, -1, width), history.reshape(batch, -1, width))
        return base + residual.reshape_as(base)

    def coefficient_logits_step(self, hidden, atoms, model_aux, history, position, cache=None):
        base, atom = self._coefficient_base(hidden, atoms, model_aux)
        if self.coefficient_conditioner is None:
            return base, None
        residual, cache = self.coefficient_conditioner.step(hidden, atom, history, position, cache)
        return base + residual, cache

    @staticmethod
    def mask_seen_atoms(logits, atoms):
        result = logits.clone()
        for depth in range(1, atoms.shape[-1]):
            result[..., depth, :].scatter_(-1, atoms[..., :depth], -float('inf'))
        return result

    def _context(self, amp):
        device = self.atom_embedding.weight.device
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if amp and device.type == 'cuda' else nullcontext())

    def forward(self, packed, model_aux=None, cond=None, amp=False):
        if tuple(packed.shape[1:]) != self.block_size:
            raise ValueError(f"expected packed [batch, {self.block_size}], got {tuple(packed.shape)}")
        if model_aux is None:
            raise ValueError("the frozen dictionary and coefficient bins are required")
        batch = packed.shape[0]
        atoms, coefficients = self.unpack(packed)
        positions = torch.arange(self.events, device=packed.device)
        if cond is None:
            cond = torch.zeros(batch, device=packed.device, dtype=torch.long)
        with self._context(amp):
            # Drop the final pair before shifting: the current pair never enters
            # its own history, including across spatial-site boundaries.
            pairs = self.pair_embeddings(atoms.flatten(1)[:, :-1],
                coefficients.flatten(1)[:, :-1], model_aux,
                positions[:-1] % self.block_size[-1])
            history = torch.cat((self.class_embedding(cond.reshape(batch)).unsqueeze(1), pairs), dim=1)
            hidden = self.embedding_dropout(history + self.position_embedding(positions))
            for block in self.blocks:
                if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                    hidden = checkpoint(lambda x, block=block: block(x)[0], hidden,
                                        use_reentrant=False)
                else:
                    hidden = block(hidden)[0]
            hidden = self.final_norm(hidden).reshape(batch, *self.block_size, -1)
            atom_logits = self.atom_classifier(hidden)
            if self.config.mask_seen_atoms_training:
                atom_logits = self.mask_seen_atoms(atom_logits, atoms)
            result = dict(atom_logits=atom_logits,
                          coeff_logits=self.coefficient_logits(hidden, atoms, model_aux,
                              history=history.reshape(batch, *self.block_size, -1)))
            if self.capture_shared_hidden:
                result['shared_hidden'] = hidden
            return result

    def decode_step(self, previous_pair, model_aux, cond, position, kv_cache=None, *, return_history=False):
        """Consume BOS or exactly one completed pair and return the next context."""
        if not 0 <= position < self.events:
            raise ValueError("decoding position is outside the image")
        if position == 0:
            if kv_cache is not None or previous_pair is not None:
                raise ValueError("BOS requires an empty cache and no previous pair")
            hidden = self.class_embedding(cond.reshape(-1)).unsqueeze(1)
        else:
            if kv_cache is None or previous_pair is None:
                raise ValueError("later positions require the completed preceding pair and cache")
            atoms, coefficients = self.unpack(previous_pair.reshape(-1, 1))
            depth = (position - 1) % self.block_size[-1]
            hidden = self.pair_embeddings(atoms, coefficients, model_aux, depth)
        history = hidden[:, 0]
        hidden = hidden + self.position_embedding.weight[position].view(1, 1, -1)
        cache = []
        for index, block in enumerate(self.blocks):
            hidden, state = block(hidden, kv_cache=None if kv_cache is None else kv_cache[index])
            cache.append(state)
        result = self.final_norm(hidden)[:, 0], cache
        return (*result, history) if return_history else result

    @torch.no_grad()
    def sample_compound(self, batch_size, model_aux, cond=None, temperature=1.,
                        atom_temperature=None, coeff_temperature=None,
                        atom_top_k=0, atom_top_p=1., coeff_top_k=0, coeff_top_p=1.,
                        amp=True):
        if self.training:
            raise RuntimeError("call eval() before cached sampling")
        device = self.atom_embedding.weight.device
        if cond is None:
            cond = torch.zeros(batch_size, device=device, dtype=torch.long)
        atoms = torch.empty(batch_size, self.events, device=device, dtype=torch.long)
        coefficients = torch.empty_like(atoms)
        at = temperature if atom_temperature is None else atom_temperature
        ct = temperature if coeff_temperature is None else coeff_temperature
        if at <= 0 or ct <= 0:
            raise ValueError("sampling temperatures must be positive")
        cache = previous = coefficient_cache = None
        depth_count = self.block_size[-1]
        with self._context(amp):
            for position in range(self.events):
                hidden, cache, history = self.decode_step(previous, model_aux, cond, position, cache, return_history=True)
                logits = self.atom_classifier(hidden)
                depth = position % depth_count
                if depth:
                    logits.scatter_(1, atoms[:, position-depth:position], -float('inf'))
                atom = sample_from_logits(logits, at, min(atom_top_k or self.num_atoms, self.num_atoms), atom_top_p)
                coeff_logits, coefficient_cache = self.coefficient_logits_step(
                    hidden, atom, model_aux, history, position, coefficient_cache)
                coefficient = sample_from_logits(coeff_logits, ct,
                    min(coeff_top_k or self.coeff_vocab_size, self.coeff_vocab_size), coeff_top_p)
                atoms[:, position], coefficients[:, position] = atom, coefficient
                previous = atom * self.coeff_vocab_size + coefficient
        return (atoms.reshape(batch_size, *self.block_size),
                coefficients.reshape(batch_size, *self.block_size))
