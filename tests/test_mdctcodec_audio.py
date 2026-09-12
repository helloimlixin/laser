from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import pytest

from src.models.audio_codec import (
    MDCTCodecOrthonormalAnalysis, MDCTCodecOrthonormalSynthesis,
    build_mdctcodec_official_backbone,
)
from src.models.dictionary_learner import DictionaryLearning
from src.models.discriminator import MDCTCodecOfficialDiscriminator
from train import _make_selected_checkpoint_artifact_callback


def test_mdct_reconstructs_boundaries_and_preserves_energy():
    x = torch.randn(2, 1, 7960)
    analysis, synthesis = MDCTCodecOrthonormalAnalysis(40), MDCTCodecOrthonormalSynthesis(40)
    coefficients = analysis(x)
    torch.testing.assert_close(synthesis(coefficients), x, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(coefficients.square().sum(), x.square().sum(), rtol=2e-6, atol=1e-4)


def test_mdct_laser_replaces_rvq_and_backpropagates():
    encoder, decoder = build_mdctcodec_official_backbone(
        latent_dim=8, hidden_channels=16, intermediate_channels=32, num_layers=2,
    )
    bottleneck = DictionaryLearning(num_embeddings=32, embedding_dim=8, sparsity_level=4, omp_ridge=0.05)
    x = torch.randn(2, 1, 1240) * 0.1
    z = encoder(x)
    sparse, commitment, codes = bottleneck(z.unsqueeze(-2))
    y = decoder(sparse.squeeze(-2))
    loss = (y - x).square().mean() + commitment + bottleneck._last_dictionary_loss_for_backward
    loss.backward()
    assert y.shape == x.shape
    assert codes.support.shape[-1] == 4
    for module in (encoder, decoder, bottleneck):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


def test_mdct_discriminator_has_three_finite_branches():
    critic = MDCTCodecOfficialDiscriminator(num_filters=4)
    logits, features = critic(torch.randn(2, 1, 1240), return_features=True)
    assert len(logits) == len(features) == 3
    assert all(torch.isfinite(x).all() for x in logits)
    assert all(len(f) == 6 for f in features)


def test_upload_happens_after_every_five_completed_epochs():
    callback = _make_selected_checkpoint_artifact_callback(object)(SimpleNamespace(), every_n_epochs=5)
    uploaded = []
    callback._upload = lambda trainer, **kw: uploaded.append(trainer.current_epoch)
    trainer = SimpleNamespace(current_epoch=0, sanity_checking=False)
    for epoch in range(12):
        trainer.current_epoch = epoch
        callback.on_train_epoch_start(trainer, None)
    assert uploaded == [5, 10]


def test_encodec_short_tail_padding_preserves_original_samples():
    from scripts.benchmark_mdctcodec_vctk import pad_encodec_segment_tail
    for length, expected in [(47520, 47520), (47521, 48000), (47999, 48000), (48000, 48000)]:
        x = torch.randn(1, 2, length)
        padded = pad_encodec_segment_tail(x, 48000, 47520)
        assert padded.shape[-1] == expected
        torch.testing.assert_close(padded[..., :length], x)


def test_six_kbps_payload_roundtrip_and_invalid_tokens():
    from src.mdctcodec_bitstream import pack_frames, unpack_frames
    rng = np.random.default_rng(1)
    atoms = rng.integers(0, 8192, (150, 2))
    integers = rng.integers(-63, 64, (150, 2))
    atoms[0] = [0, 8191]; integers[0] = [-63, 63]
    payload = pack_frames(atoms, integers)
    assert len(payload) * 8 == 6000  # One second of 150 Hz frames.
    decoded_atoms, decoded_integers = unpack_frames(payload)
    np.testing.assert_array_equal(decoded_atoms, atoms)
    np.testing.assert_array_equal(decoded_integers, integers)
    assert pack_frames(np.array([[0, 0]]), np.array([[-63, -63]])) == bytes(5)
    assert pack_frames(np.array([[8191, 8191]]), np.array([[63, 63]])) == bytes.fromhex('ffffeffffe')
    with pytest.raises(ValueError): unpack_frames(payload[:-1])
    with pytest.raises(ValueError): unpack_frames(bytes([255] * 5))
    with pytest.raises(ValueError): pack_frames(atoms + 8192, integers)
    with pytest.raises(ValueError): pack_frames(atoms, integers + 127)


def test_six_kbps_quantized_bottleneck_grid_and_gradients():
    bound = 18.45128059387207
    bottleneck = DictionaryLearning(num_embeddings=8192, embedding_dim=8,
        sparsity_level=2, omp_ridge=0.05, coefficient_quantization_bits=7,
        coefficient_quantization_max=bound)
    z = torch.randn(1, 8, 1, 3, requires_grad=True)
    reconstructed, loss, codes = bottleneck(z)
    assert codes.support.shape[-1] == 2
    scaled = codes.values.detach() / (bound / 63)
    torch.testing.assert_close(scaled, scaled.round(), atol=1e-5, rtol=1e-5)
    assert scaled.abs().max() <= 63
    assert bottleneck._last_coefficient_quantization_fraction.item() == 1.0
    (reconstructed.square().mean() + loss).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
