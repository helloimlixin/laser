"""The archived FFHQ8.17 compound decoder/objective with ImageNet dimensions."""
import hashlib
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src import ffhq_v4_archived as archive
from src.models.rqtransformer.configs import RQTransformerConfig
from src.training.coefficient_support import sampling_coefficient_support

ARCHIVE_SHA256 = '9ba1b49b4e5e339f0076bebee6fbac5629f6c391601de467019a3723c9d3e33f'


def verify_archive():
    actual = hashlib.sha256(Path(archive.__file__).read_bytes()).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError('FFHQ8.17 archived implementation changed')
    return actual


class ImageNetFFHQCompound(archive.CompoundLaserRQTransformer):
    """Retain archived forward/cache behavior; accept the maintained sampler API."""

    def __init__(self, config, num_atoms, coeff_vocab_size, **kwargs):
        super().__init__(config, num_atoms, coeff_vocab_size, **kwargs)
        self.pair_autoregressive = True
        self.causal_prefix_state = False
        self.decoder_type = 'ffhq-v4-archived-compound-micro2-depth-heads'
        self.coefficient_sampling_limits = None

    def sample_compound(self, *args, coeff_top_k=0,
                        causal_prefix_sampling='predicted',
                        coefficient_sampling_limits=None, **kwargs):
        if coeff_top_k not in (0, self.coeff_vocab_size):
            raise ValueError('Archived FFHQ sampler uses the full coefficient vocabulary')
        if causal_prefix_sampling != 'predicted':
            raise ValueError('Archived FFHQ decoder has no causal-prefix auxiliary head')
        if kwargs.get('atom_top_k') == 0:
            kwargs['atom_top_k'] = self.num_atoms
        limits = (self.coefficient_sampling_limits if coefficient_sampling_limits is None
                  else coefficient_sampling_limits)
        aux = args[1] if len(args) > 1 else kwargs['model_aux']
        with sampling_coefficient_support(self, aux.coeff_bins, limits):
            return super().sample_compound(*args, **kwargs)


def build_model(total_vocab_size, num_atoms, **options):
    verify_archive()
    required = dict(compound=True, coeff_vocab_size=2048, sparsity_level=4,
        compound_micro_transformer_layers=2, compound_depth_specific_coeff_heads=True,
        compound_pair_autoregressive=True, physical_pair_context=False,
        model_preset='imagenet-1400m')
    for key, expected in required.items():
        if options.get(key) != expected:
            raise ValueError(f'ImageNet FFHQ adaptation requires {key}={expected}')
    if total_vocab_size != 18432 or num_atoms != 16384:
        raise ValueError('ImageNet K4 adaptation requires 16384 atoms and 2048 coefficient bins')
    if any(options.get(key) for key in ('orthogonal_compound', 'closed_loop_orthogonal',
            'levelwise_var', 'support_first_patterns', 'causal_prefix_patterns',
            'compound_causal_prefix_state', 'compound_geometry_head', 'compound_refiner_layers')):
        raise ValueError('Unexpected decoder extension to the archived FFHQ recipe')
    config = RQTransformerConfig.create(OmegaConf.create(dict(
        type='rq-transformer', block_size=[8, 8, 4], embed_dim=1536,
        input_embed_dim=256, shared_tok_emb=True, shared_cls_emb=True,
        input_emb_vqvae=True, head_emb_vqvae=True, cumsum_depth_ctx=True,
        vocab_size=num_atoms, vocab_size_cond=1000, block_size_cond=1,
        body=dict(n_layer=42, block=dict(n_head=24)),
        head=dict(n_layer=6, block=dict(n_head=24)))))
    return ImageNetFFHQCompound(config, num_atoms, 2048, refiner_layers=0,
        geometry_head=False, micro_transformer_layers=2, depth_specific_coeff_heads=True)


def compound_objective(*args, **kwargs):
    """Use the archived expected-atom/expected-coefficient geometry unchanged."""
    inactive = dict(causal_prefix_prediction=None, target_causal_prefix=None,
        causal_prefix_weight=0., coeff_regression_weight=0., coeff_crps_weight=0.,
        geometry_orthogonal=False, geometry_candidate_coeff_logits=None,
        target_atom_ids=None, target_atom_weights=None, target_atom_probabilities=None)
    for key, expected in inactive.items():
        value = kwargs.pop(key, expected)
        if value is not expected and (isinstance(value, torch.Tensor) or value != expected):
            raise ValueError(f'{key} is outside the archived FFHQ objective')
    # The maintained trainer supplies bins for optional regression/CRPS, whose
    # weights above must be zero. Archived geometry uses geometry_coeff_bins.
    kwargs.pop('coefficient_bins', None)
    loss, values = archive.compound_objective(*args, **kwargs)
    zero = loss.new_zeros(())
    values.update(atom_cross_entropy=values['atom_nll'], coefficient_regression=zero,
        coefficient_crps=zero, predicted_coefficients=None, target_coefficients=None,
        causal_prefix=zero, causal_prefix_mse=zero)
    return loss, values
