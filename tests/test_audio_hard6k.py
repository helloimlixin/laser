import itertools
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.audio_hard6k_bitstream import (FRAME_BITS,frame_budget,max_packet_bytes,
    sparse_words,sparse_from_words,pack_packet,unpack_packet)


def test_combinatorial_rank_preserves_atoms_coefficients_and_unordered_sum():
    rng=np.random.default_rng(21)
    supports=np.array(list(itertools.combinations(range(9),4))+[[4092,4093,4094,4095]])
    bins=rng.integers(0,9,supports.shape);bins[-1]=[0,4,7,8]
    shuffled=supports[:,[2,0,3,1]];shuffled_bins=bins[:,[2,0,3,1]]
    words=sparse_words(shuffled,shuffled_bins)
    restored,coefficients=sparse_from_words(words)
    np.testing.assert_array_equal(restored,supports)
    np.testing.assert_array_equal(coefficients,bins)
    with pytest.raises(ValueError):sparse_words(np.array([[0,0,2,3]]),np.zeros((1,4),dtype=int))


@pytest.mark.parametrize('arm',['laser','rvq'])
def test_hard_cap_for_short_tail_boundaries_and_adversarial_tokens(arm):
    rng=np.random.default_rng(5)
    for samples in [1536,1537,1599,1600,7960,8003,47999,48000,48001,96000,339600]:
        frames=frame_budget(samples,arm)
        if arm=='laser':
            atoms=np.tile([4092,4093,4094,4095],(frames,1));coefficients=rng.integers(0,9,(frames,4))
        else:atoms=rng.integers(0,1024,(frames,4));coefficients=None
        packet=pack_packet(arm,samples,atoms,coefficients)
        assert len(packet)<=max_packet_bytes(samples)
        assert len(packet)*8*48000<=6000*samples  # exact integer inequality, no tolerance
        data=unpack_packet(packet,arm)
        np.testing.assert_array_equal(data['codes'],atoms)
        if arm=='laser':np.testing.assert_array_equal(data['coefficient_bins'],coefficients)
        assert data['samples']==samples
        with pytest.raises(ValueError):pack_packet(arm,samples,np.concatenate([atoms,atoms[:1]]),
            np.concatenate([coefficients,coefficients[:1]]) if coefficients is not None else None)
        with pytest.raises(ValueError):unpack_packet(packet[:-1],arm)
        damaged=bytearray(packet);damaged[-1]^=1
        with pytest.raises(ValueError):unpack_packet(bytes(damaged),arm)
    with pytest.raises(ValueError):frame_budget(1535,arm)


@pytest.mark.parametrize('arm',['laser','rvq'])
@pytest.mark.parametrize('samples',[7960,31960])
def test_rate_controller_is_in_training_gradient_and_packet_decoder(arm,samples):
    from src.mdctcodec_hard6k import HardRateLASER,HardRateRVQ
    initial=Path('outputs/mdctcodec_matched_6kbps_rangefix')/f'{arm}_initial.pt'
    kwargs=torch.load(initial,map_location='cpu',weights_only=False)['hyper_parameters']
    kwargs.update(audio_mdct_vae_hidden_channels=16,audio_mdct_vae_convnext_intermediate_channels=32,
        audio_mdct_vae_num_residual_layers=2,adversarial_weight=0.,log_images_every_n_steps=0)
    kwargs['sparsity_level']=4
    if arm=='laser':kwargs.update(num_embeddings=4096,coefficient_quantization_bits=4,coefficient_quantization_max=1.)
    model=(HardRateLASER if arm=='laser' else HardRateRVQ)(**kwargs)
    x=torch.randn(1,1,samples)*.02
    y,loss,codes=model(x)
    assert y.shape==x.shape and codes.support.shape[-2:]==(frame_budget(samples,arm),4)
    objective=(y-x).square().mean()+loss
    if arm=='rvq':objective=objective+model.bottleneck._last_dictionary_loss_for_backward
    objective.backward()
    for component in [model.encoder,model.decoder]:
        gradients=[p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0
    model.eval();packet=model.encode_packet(x);decoded=model.decode_packet(packet)
    torch.testing.assert_close(decoded,model(x)[0].clamp(-1,1),atol=2e-5,rtol=2e-4)
    restored=(HardRateLASER if arm=='laser' else HardRateRVQ)(**dict(model.hparams))
    restored.load_state_dict(model.state_dict(),strict=True);restored.eval()
    torch.testing.assert_close(restored.decode_packet(packet),decoded,atol=0,rtol=0)
