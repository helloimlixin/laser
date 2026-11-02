"""Control density gradients entering shared features without reweighting heads."""
import math


def scale_gradient(value, scale):
    """Identity forward; multiply only the input derivative by ``scale``."""
    scale = float(scale)
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('density gradient scale must be finite and in [0, 1]')
    if scale == 1:
        return value
    return value.detach() + scale * (value - value.detach())


def install_density_gradient_scale(model, scale):
    """Scale private GMM input gradients; preserve sign and head gradients.

    Install once before training. The handles can be removed to restore the
    original backward behavior. Forward values and state-dict keys are unchanged.
    The caller must save this training policy in the experiment configuration.
    """
    scale_gradient(next(model.parameters()).new_zeros(()), scale)
    heads = getattr(model, 'gmm_heads', None)
    if heads is None or not len(heads):
        raise ValueError('model has no private gmm_heads')
    if getattr(model, '_density_gradient_handles', None):
        raise ValueError('density gradient policy is already installed')

    def hook(module, inputs):
        if len(inputs) != 1:
            raise ValueError('expected one shared-feature input per density head')
        return (scale_gradient(inputs[0], scale),)

    handles = [head.register_forward_pre_hook(hook) for head in heads]
    model._density_gradient_handles = handles
    return handles
