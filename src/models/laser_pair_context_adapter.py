"""Optional identity-rich pair context without replacing a trained decoder.

The residual begins at zero, preserving a scalar checkpoint's logits and its
existing parameter gradients. It learns only from the ordinary token CE.
Physical decoding and the alternating support/coefficient factorization stay
unchanged; no additional attention stack or prediction head is introduced.
"""
import torch
from torch import nn
from omegaconf import OmegaConf

from src.models.laser_rq_adapter import LaserPairRQTransformer
from src.rq_upstream import RQTransformerConfig


class LaserPairContextRQTransformer(LaserPairRQTransformer):
    def __init__(self, config, num_atoms, coefficient_bins):
        super().__init__(config, num_atoms)
        dimension = int(config.input_embed_dim)
        self.coefficient_identity = nn.Embedding(coefficient_bins, dimension)
        self.pair_context_residual = nn.Sequential(
            nn.LayerNorm(3 * dimension), nn.Linear(3 * dimension, dimension),
            nn.SiLU(), nn.Linear(dimension, dimension, bias=False))
        nn.init.zeros_(self.pair_context_residual[-1].weight)
        self.decoder_type = 'scalar_ce_zero_initialized_pair_context_v1'

    def _context_contributions(self, tokens, aux, contributions):
        atoms = tokens[..., 0::2]
        ids = tokens[..., 1::2] - self.num_atoms
        features = torch.cat((aux.dictionary.T[atoms],
                              self.coefficient_identity(ids), contributions), -1)
        return contributions + self.pair_context_residual(features)

    def load_scalar_state(self, state, *, assign=False):
        """Load an unchanged scalar checkpoint, rejecting other mismatches."""
        additions = {name for name in self.state_dict()
                     if name.startswith(('coefficient_identity.', 'pair_context_residual.'))}
        expected = set(self.state_dict()) - additions
        if set(state) != expected:
            raise ValueError('Expected the complete scalar state with no architecture changes')
        result = self.load_state_dict(state, strict=False, assign=assign)
        assert set(result.missing_keys) == additions and not result.unexpected_keys
        # Reusing this helper must also restore the identity initialization.
        nn.init.zeros_(self.pair_context_residual[-1].weight)


def build_pair_context_model(arch, *, num_atoms, coefficient_bins, depth=4):
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.block_size = [*arch.block_size[:2], 2 * depth]
    arch.vocab_size = num_atoms + coefficient_bins
    model = LaserPairContextRQTransformer(
        RQTransformerConfig.create(arch), num_atoms, coefficient_bins)
    model.apply(model._init_weights)
    nn.init.zeros_(model.pair_context_residual[-1].weight)
    return model
