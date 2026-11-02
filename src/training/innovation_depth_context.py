"""Feed individual completed pairs to the depth transformer, without repetition."""


def install_innovation_depth_context(model):
    """Replace cumulative depth tokens with innovations; retain spatial inputs.

    The depth transformer already attends to every earlier pair. Repeating the
    entire prefix at each position repeats early coefficient errors as well.
    This policy changes no parameters, targets, or completed-site embeddings.
    Install before training and again on inference/checkpoint recovery.
    """
    if getattr(model, '_innovation_depth_context_installed', False):
        raise ValueError('innovation depth context is already installed')
    state = {'previous': None}

    def head_input(module, inputs):
        value, = inputs
        if value.ndim == 4:
            import torch
            return (torch.cat((value[..., :1, :], value[..., 1:, :]-value[..., :-1, :]), -2),)
        if value.ndim != 2:
            raise ValueError('unexpected depth projection input shape')
        previous = state['previous']
        state['previous'] = value.detach()
        return (value if previous is None else value-previous,)

    handle = model.head_mlp.register_forward_pre_hook(head_input)
    cached = model.cached_head_output

    def cached_with_innovations(packed, model_aux, cond, sample_loc, amp=True):
        if sample_loc[-1] == 0:
            state['previous'] = None
        return cached(packed, model_aux, cond, sample_loc, amp)

    model.cached_head_output = cached_with_innovations
    model._innovation_depth_context_installed = True
    return handle
