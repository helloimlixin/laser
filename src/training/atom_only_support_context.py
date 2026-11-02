"""Separate stable atom history from continuous coefficient-density history."""


def install_atom_only_support_context(model):
    """Use an atom-only depth pass for support; keep the physical density pass.

    Shares the existing head weights. The spatial backbone keeps physical
    completed sites, and the density head keeps the original physical prefix.
    No current/future atom is included in its own classifier context.
    """
    import torch
    if getattr(model, '_atom_only_support_context_installed', False):
        raise ValueError('atom-only support context is already installed')
    if getattr(model, '_innovation_depth_context_installed', False):
        raise ValueError('atom-only and innovation-depth policies are separate trials')
    state = {}

    def inputs(module, args, kwargs):
        packed = args[0] if args else kwargs['packed']
        aux = args[1] if len(args)>1 else kwargs['model_aux']
        state.update(mode='full', atoms=packed[..., 0].long(), aux=aux)

    def spatial(module, args, output):
        state['spatial'] = output

    def support(module, args):
        aux = state['aux']
        if state['mode'] == 'full':
            atoms = state['atoms']
            batch, height, width, depth = atoms.shape
            vectors = aux.dictionary.T[atoms].reshape(batch, height*width, depth, -1)
            history = model.head_mlp(vectors.cumsum(-2))
            tokens = torch.cat((state['spatial'].unsqueeze(-2), history[:, :, :-1]), -2)
            tokens = tokens.reshape(batch*height*width, depth, -1)+model.pos_emb_d
            hidden = model.head_transformer(tokens).reshape(batch, height, width, depth, -1)
        else:
            packed, (h, w, d) = state['packed'], state['location']
            context = model._cache['spatial_ctx_hw']
            if d:
                vectors = aux.dictionary.T[packed[:, h, w, :d, 0].long()]
                history = model.head_mlp(vectors.cumsum(-2))
                context = torch.cat((context, history), 1)
            tokens = context+model.pos_emb_d[:, :d+1]
            hidden = model.head_transformer(tokens)[:, -1]
        return (hidden,)

    handles = [model.register_forward_pre_hook(inputs, with_kwargs=True),
               model.body_transformer.register_forward_hook(spatial),
               model.classifier.register_forward_pre_hook(support)]
    cached = model.cached_head_output

    def cached_with_support(packed, model_aux, cond, sample_loc, amp=True):
        state.update(mode='cached', packed=packed, aux=model_aux, location=sample_loc)
        return cached(packed, model_aux, cond, sample_loc, amp)

    model.cached_head_output = cached_with_support
    model._atom_only_support_context_installed = True
    return handles
