"""One globally normalized pair-ID classifier with a smaller output width.

Every Cartesian pair owns an independent output row. The projection reduces
the rank of the context-to-vocabulary matrix; it does not factor probabilities
into atom and coefficient decisions. Native RQ forward/cache/sampling consume
the ordinary dense vocabulary logits without modification.
"""
from contextlib import contextmanager
import sys
from torch import nn


class ProjectedPairIDLinear(nn.Module):
    def __init__(self, in_features, out_features, projection_dim, *, chunked_training_loss=False):
        super().__init__()
        if not 0 < projection_dim <= in_features or out_features < 2:
            raise ValueError('Positive projection width no larger than the input width required')
        self.in_features, self.out_features = int(in_features), int(out_features)
        self.projection_dim = int(projection_dim)
        self.chunked_training_loss = bool(chunked_training_loss)
        self.projection = nn.Linear(in_features, projection_dim, bias=False)
        self.classes = nn.Linear(projection_dim, out_features)

    def forward(self, inputs):
        projected = self.projection(inputs)
        if self.training and self.chunked_training_loss:
            return projected
        return self.classes(projected)


@contextmanager
def skip_redundant_native_topk(model):
    """Keep the native sampler, skipping only top-k when k is the whole vocab.

    Sorting every class cannot remove any logit, consumes no RNG, and does
    not change the subsequent native top-p/multinomial distribution.
    Sampling is single threaded in the shared training driver.
    """
    module = sys.modules[type(model).__module__]
    native = module.sample_from_logits
    def sample(logits,temperature=1.,top_k=None,top_p=None):
        if top_k is not None and top_k >= logits.shape[-1]:
            top_k = None
        return native(logits,temperature=temperature,top_k=top_k,top_p=top_p)
    module.sample_from_logits = sample
    try:
        yield
    finally:
        module.sample_from_logits = native
