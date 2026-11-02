"""Optional learned atom-ID inputs for the native compound prior.

The added table begins at zero. A continued compound checkpoint therefore
starts with identical logits and base-parameter gradients. Only the prior's
history representation changes; tokenizer targets and reconstruction do not.
"""
import torch
from torch import nn
from omegaconf import OmegaConf

from src.models.laser_native_compound_adapter import LaserNativeCompoundRQTransformer
from src.rq_upstream import RQTransformerConfig


class LaserAtomIdentityRQTransformer(LaserNativeCompoundRQTransformer):
    def __init__(self, config, num_atoms, coefficient_bins, depth=4):
        super().__init__(config, num_atoms, coefficient_bins, depth)
        self.atom_identity_embedding = nn.Embedding(num_atoms, int(config.input_embed_dim))
        nn.init.zeros_(self.atom_identity_embedding.weight)
        self.decoder_type = 'native_compound_zero_initialized_atom_identity_v1'

    def pair_embeddings(self, tokens, aux):
        return super().pair_embeddings(tokens, aux) + self.atom_identity_embedding(tokens[..., 0::2])

    def load_compound_state(self, state, *, assign=False):
        """Strictly transfer every existing tensor and zero the added table."""
        added = {'atom_identity_embedding.weight'}
        if set(state) != set(self.state_dict()) - added:
            raise ValueError('Expected a complete native compound checkpoint without identity additions')
        result = self.load_state_dict(state, strict=False, assign=assign)
        assert set(result.missing_keys) == added and not result.unexpected_keys
        if self.atom_identity_embedding.weight.is_meta:
            device = next(iter(state.values())).device
            self.atom_identity_embedding.to_empty(device=device)
        nn.init.zeros_(self.atom_identity_embedding.weight)


def build_atom_identity_model(arch, *, num_atoms, coefficient_bins, depth=4):
    """Construct a model for strict continuation from native compound weights."""
    arch = OmegaConf.create(OmegaConf.to_container(arch, resolve=True))
    arch.block_size = [*arch.block_size[:2], depth]
    arch.vocab_size = num_atoms
    return LaserAtomIdentityRQTransformer(RQTransformerConfig.create(arch), num_atoms,
                                         coefficient_bins, depth)
