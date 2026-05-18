"""Core stage-1 behavior: both autoencoders reconstruct and expose cache APIs."""

import numpy as np
import torch

from src.codebook_visuals import (
    _fixed_square_axis_limits,
    _pca_project_snapshots,
    render_codebook_scatter,
    save_codebook_trajectory_gif,
)
from src.models.laser import LASER
from src.models.vqvae import VQVAE


def test_laser_forward_and_sparse_token_decode():
    torch.manual_seed(0)
    model = LASER(
        in_channels=3,
        num_hiddens=16,
        num_embeddings=16,
        embedding_dim=4,
        sparsity_level=2,
        num_residual_blocks=1,
        num_residual_hiddens=8,
        commitment_cost=0.25,
        learning_rate=1e-3,
        beta=0.9,
        perceptual_weight=0.0,
        compute_fid=False,
        log_images_every_n_steps=0,
        enable_val_latent_visuals=True,
        codebook_visual_max_vectors=4,
    )
    x = torch.randn(2, 3, 16, 16)

    recon, bottleneck_loss, codes = model(x)
    # Cache extraction depends on these public token helpers, not on Lightning
    # logging or trainer plumbing.
    tokens, latent_hw = model.encode_to_tokens(x, coeff_vocab_size=5, coeff_max=2.0)
    decoded = model.decode_from_tokens(
        tokens,
        latent_hw=latent_hw,
        coeff_vocab_size=5,
        coeff_max=2.0,
    )

    assert recon.shape == x.shape
    assert decoded.shape == x.shape
    assert torch.isfinite(bottleneck_loss)
    assert codes.support.shape[-1] == 2
    assert tokens.shape[-1] == 4
    heatmaps = model._sparse_heatmaps(codes, image_hw=x.shape[-2:])
    assert len(heatmaps) == x.size(0)
    assert heatmaps[0].shape == tuple(x.shape[-2:])
    assert 0.0 <= float(heatmaps[0].min()) <= float(heatmaps[0].max()) <= 1.0
    model.on_fit_start()
    model._snapshot_dictionary()
    assert len(model._dict_snapshots) == 1


def test_vqvae_forward_and_index_decode():
    torch.manual_seed(0)
    model = VQVAE(
        in_channels=3,
        num_hiddens=16,
        num_embeddings=16,
        embedding_dim=4,
        num_residual_blocks=1,
        num_residual_hiddens=8,
        commitment_cost=0.25,
        decay=0.99,
        perceptual_weight=0.0,
        learning_rate=1e-3,
        beta=0.9,
        compute_fid=False,
        enable_codebook_visuals=True,
        codebook_visual_max_vectors=4,
    )
    x = torch.randn(2, 3, 16, 16)

    recon, vq_loss, perplexity = model(x)
    # VQ-VAE caches store flat codebook indices and later decode them back to
    # latents/images during stage-2 preview generation.
    indices, h_z, w_z = model.encode_to_indices(x)
    decoded = model.decode_from_indices(indices, h_z, w_z)

    assert recon.shape == x.shape
    assert decoded.shape == x.shape
    assert indices.shape == (2, h_z * w_z)
    assert torch.isfinite(vq_loss)
    assert torch.isfinite(perplexity)
    model.on_fit_start()
    model._snapshot_codebook()
    assert len(model._codebook_snapshots) == 1


def test_laser_simple_backbone_matches_vqvae_encoder_decoder_depth():
    torch.manual_seed(0)
    common = dict(
        in_channels=3,
        num_hiddens=16,
        num_embeddings=16,
        embedding_dim=4,
        num_residual_blocks=1,
        num_residual_hiddens=8,
        commitment_cost=0.25,
        learning_rate=1e-3,
        beta=0.9,
        compute_fid=False,
        num_downsamples=4,
    )
    laser = LASER(
        **common,
        sparsity_level=2,
        perceptual_weight=0.0,
        backbone="simple",
        log_images_every_n_steps=0,
    )
    vqvae = VQVAE(
        **common,
        decay=0.99,
        perceptual_weight=0.0,
    )
    laser.eval()
    vqvae.eval()
    x = torch.randn(2, 3, 32, 32)

    assert type(laser.encoder) is type(vqvae.encoder)
    assert type(laser.decoder) is type(vqvae.decoder)
    assert laser.encoder.num_downsamples == vqvae.encoder.num_downsamples == 4
    assert laser.decoder.num_upsamples == vqvae.decoder.num_upsamples == 4
    assert laser.pre_bottleneck(laser.encoder(x)).shape[-2:] == (2, 2)
    assert vqvae.pre_bottleneck(vqvae.encoder(x)).shape[-2:] == (2, 2)


def test_laser_nonoverlap_patch_tokens_have_expected_grid_and_depth():
    torch.manual_seed(0)
    model = LASER(
        in_channels=3,
        num_hiddens=16,
        num_embeddings=32,
        embedding_dim=4,
        sparsity_level=2,
        num_residual_blocks=1,
        num_residual_hiddens=8,
        commitment_cost=0.25,
        learning_rate=1e-3,
        beta=0.9,
        perceptual_weight=0.0,
        compute_fid=False,
        backbone="simple",
        num_downsamples=2,
        patch_based=True,
        patch_size=2,
        patch_stride=2,
        patch_reconstruction="tile",
        log_images_every_n_steps=0,
    )
    x = torch.randn(2, 3, 32, 32)

    tokens, latent_hw = model.encode_to_tokens(x, coeff_vocab_size=8, coeff_max=2.0)

    assert latent_hw == (8, 8)
    assert tokens.shape == (2, 4, 4, 4)
    assert model.bottleneck.patch_reconstruction == "tile"


def test_codebook_progression_visual_helpers(tmp_path):
    torch.manual_seed(0)
    first = torch.randn(8, 4)
    second = first + 0.05 * torch.randn(8, 4)
    snapshots = [first, second]
    steps = [0, 10]

    scatter = render_codebook_scatter(snapshots, steps, title="test vectors")
    gif_path = save_codebook_trajectory_gif(
        snapshots,
        steps,
        tmp_path / "vectors.gif",
        title="test vector trajectories",
        fps=1,
    )

    assert scatter is not None
    assert scatter.ndim == 3
    assert scatter.shape[-1] == 3
    assert gif_path is not None
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_codebook_animation_uses_fixed_square_axis_limits():
    projected = [
        np.array([[0.0, 0.0], [10.0, 1.0]]),
        np.array([[2.0, -1.0], [8.0, 3.0]]),
    ]

    x_lim, y_lim = _fixed_square_axis_limits(projected, margin_fraction=0.1)

    assert x_lim[0] <= 0.0
    assert x_lim[1] >= 10.0
    assert y_lim[0] <= -1.0
    assert y_lim[1] >= 3.0
    assert np.isclose(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0])


def test_codebook_pca_uses_full_trajectory_not_only_latest_snapshot():
    first = torch.tensor([[-10.0, 0.0], [10.0, 0.0]])
    second = torch.tensor([[0.0, -1.0], [0.0, 1.0]])

    projected, pc1_var, pc2_var = _pca_project_snapshots([first, second])
    first_span = float(np.ptp(projected[0][:, 0]))
    second_span = float(np.ptp(projected[1][:, 0]))

    assert first_span > 10.0
    assert second_span < 1.0e-5
    assert pc1_var > pc2_var
