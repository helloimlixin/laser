"""Attach learned completed-site pooling without moving existing parameters."""
from src.models.site_pair_pooling import SitePairPooling
from src.training.rqtransformer import CompoundLaserRQTransformer


class SitePooledCompoundTransformer(CompoundLaserRQTransformer):
    def pool_spatial_pairs(self, pair_embeddings):
        return self.site_pooling(pair_embeddings)


def attach_site_pair_pooling(model, *, width=480, heads=8, mode='attention'):
    if type(model) is not CompoundLaserRQTransformer or not model.pair_autoregressive:
        raise ValueError('site pooling requires the plain full-pair compound model')
    if model.causal_prefix_state or model.contribution_head is not None:
        raise ValueError('site pooling requires the standard compound objective')
    parameter = next(model.parameters())
    model.__class__ = SitePooledCompoundTransformer
    model.site_pooling = SitePairPooling(
        model.config.embed_dim, model.block_size[-1], width=width, heads=heads, mode=mode,
    ).to(device=parameter.device, dtype=parameter.dtype)
    model.site_pooling.train(model.training)
    return model
