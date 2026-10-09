"""Inference-only controls for the added rich pair-memory retrieval path."""


def use_bos_only_memory(decoder):
    """Replace encoded records and the latest-record query seed with encoded BOS.

    The original causal encoder/cache still advances once per completed pair.
    The backbone history, local reconstruction prefix, current-atom conditioning,
    trained weights and query FFNs remain available. No RNG draws are added.
    """
    import torch
    assert not getattr(decoder, '_bos_memory_ablation', False)
    native_encode = decoder.encode_memory
    native_reset = decoder.reset_cache
    decoder._ablation_bos = None
    decoder._bos_memory_ablation = True

    def reset():
        native_reset()
        decoder._ablation_bos = None

    def encode(*args, **kwargs):
        assert not decoder.training and not torch.is_grad_enabled(), 'BOS substitution is inference only'
        memory = native_encode(*args, **kwargs)
        event = kwargs.get('event')
        if event is None:
            return memory[:, :1].expand_as(memory)
        if event == 0:
            decoder._ablation_bos = memory
        assert decoder._ablation_bos is not None
        assert decoder._ablation_bos.shape == memory.shape
        return decoder._ablation_bos

    decoder.encode_memory = encode
    decoder.reset_cache = reset
