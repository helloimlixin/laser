"""Fresh-model initialization matching the original RQ training driver."""
from torch import nn


def initialize_rq_prior(model):
    """Initialize the completed model once, retaining zero history residuals."""
    model.apply(model._init_weights)
    history = getattr(model, 'coefficient_history_decoder', None)
    if history is not None:
        # The history adapter deliberately starts as an exact zero residual.
        # The generic RQ initializer would otherwise turn it on immediately.
        nn.init.zeros_(history.output.weight)
    return model
