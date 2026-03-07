import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


scratch_laser = _load_module("scratch_laser_coeff_regression_test_module", "scratch/laser.py")

DistributedContext = scratch_laser.DistributedContext
DictionaryLearning = scratch_laser.DictionaryLearning
LASER = scratch_laser.LASER
Prior = scratch_laser.Prior
PriorConfig = scratch_laser.PriorConfig
train_stage2_transformer = scratch_laser.train_stage2_transformer
precompute_tokens = scratch_laser.precompute_tokens
safe_atanh = scratch_laser.safe_atanh


def test_prior_predict_coefficients_returns_dual_outputs():
    cfg = PriorConfig(
        vocab_size=8,
        H=1,
        W=1,
        D=2,
        predict_coefficients=True,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    prior = Prior(cfg, bos_token_id=6, pad_token_id=7).eval()

    x = torch.tensor([[6, 1]], dtype=torch.long)
    coeff_tokens = torch.tensor([[1, 2]], dtype=torch.long)
    logits, coeff_pred = prior(x, coeff_tokens=coeff_tokens)

    assert logits.shape == (1, 2, 8)
    assert coeff_pred.shape == (1, 2)
    assert torch.isfinite(coeff_pred).all()


def test_prior_coefficient_prediction_depends_on_aligned_tokens():
    cfg = PriorConfig(
        vocab_size=6,
        H=1,
        W=1,
        D=2,
        predict_coefficients=True,
        d_model=2,
        n_heads=1,
        n_layers=1,
        d_ff=4,
        dropout=0.0,
        coeff_max=10.0,
    )
    prior = Prior(cfg, bos_token_id=4, pad_token_id=5).eval()

    with torch.no_grad():
        for param in prior.parameters():
            param.zero_()
        prior.token_emb.weight[1] = torch.tensor([1.0, 0.0])
        prior.token_emb.weight[2] = torch.tensor([0.0, 2.0])
        prior.coeff_head[0].weight.zero_()
        prior.coeff_head[0].bias.zero_()
        prior.coeff_head[0].weight[0, 2] = 1.0
        prior.coeff_head[0].weight[1, 3] = 1.0
        prior.coeff_head[2].weight.zero_()
        prior.coeff_head[2].bias.zero_()
        prior.coeff_head[2].weight[0, 0] = 1.0
        prior.coeff_head[2].weight[0, 1] = 1.0

    x = torch.zeros((1, 2), dtype=torch.long)
    _, coeff_a = prior(x, coeff_tokens=torch.tensor([[1, 1]], dtype=torch.long))
    _, coeff_b = prior(x, coeff_tokens=torch.tensor([[2, 2]], dtype=torch.long))

    assert not torch.allclose(coeff_a, coeff_b)


def test_prior_generate_returns_atom_tokens_and_float_coeffs():
    torch.manual_seed(0)

    cfg = PriorConfig(
        vocab_size=5,
        H=1,
        W=1,
        D=2,
        predict_coefficients=True,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    prior = Prior(cfg, bos_token_id=3, pad_token_id=4).eval()
    with torch.no_grad():
        for param in prior.parameters():
            param.zero_()

    atom_ids, coeffs = prior.generate(batch_size=4, show_progress=False)

    assert atom_ids.shape == (4, 2)
    assert coeffs.shape == (4, 2)
    assert torch.isfinite(coeffs).all()
    assert not torch.isin(atom_ids, torch.tensor([3, 4])).any()


def test_stage2_training_smoke_with_float_coeff_regression(tmp_path):
    torch.manual_seed(0)

    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=1,
        latent_patch_stride=1,
        quantize_sparse_coeffs=False,
    ).to(torch.device("cpu"))

    tokens_flat = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    coeffs_flat = torch.tensor([[0.5, -0.25], [-0.1, 0.8]], dtype=torch.float32)
    loader = DataLoader(TensorDataset(tokens_flat, coeffs_flat), batch_size=2, shuffle=False)

    prior = Prior(
        PriorConfig(
            vocab_size=laser.bottleneck.vocab_size,
            H=1,
            W=1,
            D=2,
            predict_coefficients=True,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            dropout=0.0,
        ),
        bos_token_id=laser.bottleneck.bos_token_id,
        pad_token_id=laser.bottleneck.pad_token_id,
    ).to(torch.device("cpu"))

    train_stage2_transformer(
        transformer=prior,
        token_loader=loader,
        dist_ctx=DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device("cpu"),
            backend=None,
        ),
        epochs=1,
        lr=1e-3,
        pad_token_id=laser.bottleneck.pad_token_id,
        out_dir=str(tmp_path),
        ae_for_decode=laser,
        H=1,
        W=1,
        D=2,
        sample_every_steps=0,
        coeff_loss_weight=1.0,
        wandb_run=None,
    )

    assert (tmp_path / "transformer_last.pt").exists()


def test_laser_forward_outputs_are_finite():
    torch.manual_seed(0)

    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=1,
        latent_patch_stride=1,
        quantize_sparse_coeffs=False,
    ).to(torch.device("cpu"))

    x = torch.randn(2, 3, 8, 8)
    recon, _, _ = laser(x)

    assert torch.isfinite(recon).all()


def test_bottleneck_loss_is_rescaled_back_to_pre_normalized_latent_scale():
    torch.manual_seed(0)

    bottleneck = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=2,
        sparsity_level=2,
        patch_size=2,
        patch_stride=2,
        quantize_sparse_coeffs=False,
        commitment_cost=0.25,
        epsilon=1e-6,
    ).eval()

    z_e = torch.randn(1, 2, 4, 4)

    with torch.no_grad():
        support, coeffs = bottleneck._encode_sparse_codes(z_e)
        z_q = bottleneck._reconstruct_sparse(support, coeffs)
        _, loss, _ = bottleneck(z_e)

    expected = float(bottleneck.patch_dim) * (
        F.mse_loss(z_q, z_e) + bottleneck.commitment_cost * F.mse_loss(z_q, z_e)
    )

    assert torch.allclose(loss, expected, atol=1e-6)


def test_latent_transform_scale_matches_sparse_signal_dimension():
    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=2,
        latent_patch_stride=2,
        quantize_sparse_coeffs=False,
    ).eval()

    z = torch.randn(3, 2, 4, 4)

    scale = laser._latent_transform_scale(z)

    assert torch.allclose(scale, torch.tensor(float(laser.bottleneck.patch_dim) ** 0.5, dtype=z.dtype))


def test_latent_transform_helpers_are_inverse_pairs():
    laser = LASER(
        in_channels=3,
        num_hiddens=2,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=1,
        latent_patch_stride=1,
        quantize_sparse_coeffs=False,
    ).eval()

    with torch.no_grad():
        laser.pre_bottleneck.weight.zero_()
        laser.pre_bottleneck.bias.zero_()
        laser.pre_bottleneck.weight[0, 0, 0, 0] = 1.0
        laser.pre_bottleneck.weight[1, 1, 0, 0] = 1.0

    z = torch.tensor(
        [[[[0.25, -0.5], [0.1, -0.2]], [[-0.3, 0.4], [0.2, -0.1]]]],
        dtype=torch.float32,
    )

    encoded = laser._to_bottleneck_input(z)
    decoded = laser._from_bottleneck_output(encoded)

    assert torch.allclose(decoded, z, atol=1e-6)


def test_latent_transform_keeps_each_sparse_signal_norm_within_one():
    torch.manual_seed(0)

    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=2,
        latent_patch_stride=2,
        quantize_sparse_coeffs=False,
    ).eval()

    z = torch.randn(3, 8, 4, 4)

    with torch.no_grad():
        encoded = laser._to_bottleneck_input(z)

    patches, _, _ = laser.bottleneck._extract_patches(encoded)
    signal_norm = patches.view(-1, laser.bottleneck.patch_dim).norm(dim=1)

    assert torch.all(signal_norm <= 1.0 + 1e-6)


def test_laser_encode_is_batch_invariant():
    torch.manual_seed(0)

    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=1,
        latent_patch_stride=1,
        quantize_sparse_coeffs=False,
    ).eval()

    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        codes_batch, coeffs_batch, _, _ = laser.encode(x)
        codes_single, coeffs_single, _, _ = laser.encode(x[:1])

    assert torch.equal(codes_batch[:1], codes_single)
    assert torch.allclose(coeffs_batch[:1], coeffs_single, atol=1e-6)


def test_laser_encode_decode_matches_forward_non_quantized():
    torch.manual_seed(0)

    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=1,
        latent_patch_stride=1,
        quantize_sparse_coeffs=False,
    ).eval()

    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        recon_forward, _, tokens_forward = laser(x)
        codes, coeffs, h_tokens, w_tokens = laser.encode(x)
        recon_decode = laser.decode(codes, coeffs)

    assert coeffs is not None
    assert h_tokens == tokens_forward.shape[1]
    assert w_tokens == tokens_forward.shape[2]
    assert torch.equal(codes, tokens_forward)
    assert torch.allclose(recon_decode, recon_forward, atol=1e-5)


def test_laser_encode_decode_matches_forward_quantized():
    torch.manual_seed(0)

    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=1,
        latent_patch_stride=1,
        quantize_sparse_coeffs=True,
        n_bins=5,
        coef_max=1.0,
    ).eval()

    x = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        recon_forward, _, tokens_forward = laser(x)
        codes, coeffs, h_tokens, w_tokens = laser.encode(x)
        recon_decode = laser.decode(codes)

    assert coeffs is None
    assert h_tokens == tokens_forward.shape[1]
    assert w_tokens == tokens_forward.shape[2]
    assert torch.equal(codes, tokens_forward)
    assert torch.allclose(recon_decode, recon_forward, atol=1e-5)


def test_precompute_tokens_matches_encode_transform_contract():
    torch.manual_seed(0)

    laser = LASER(
        in_channels=3,
        num_hiddens=8,
        num_downsamples=1,
        num_residual_layers=1,
        num_residual_hiddens=4,
        embedding_dim=2,
        num_embeddings=4,
        sparsity_level=2,
        latent_patch_size=1,
        latent_patch_stride=1,
        quantize_sparse_coeffs=False,
    ).eval()

    x = torch.randn(3, 3, 8, 8)
    loader = DataLoader(TensorDataset(x, torch.zeros(3)), batch_size=2, shuffle=False)

    tokens_flat, coeffs_flat, indices_flat, h_tokens, w_tokens, token_depth, raw_coeff_min, raw_coeff_max = (
        precompute_tokens(laser, loader, torch.device("cpu"), show_progress=False)
    )

    expected_tokens = []
    expected_coeffs = []
    for sample in x:
        codes, coeffs, h_i, w_i = laser.encode(sample.unsqueeze(0))
        expected_tokens.append(codes.view(1, -1).to(torch.int32))
        expected_coeffs.append(coeffs.view(1, -1).to(torch.float32))
        assert h_i == h_tokens
        assert w_i == w_tokens

    expected_tokens = torch.cat(expected_tokens, dim=0)
    expected_coeffs = torch.cat(expected_coeffs, dim=0)

    assert indices_flat is None
    assert token_depth == laser.bottleneck.token_depth
    assert torch.equal(tokens_flat, expected_tokens)
    assert torch.allclose(coeffs_flat, expected_coeffs, atol=1e-6)
    assert abs(raw_coeff_min - float(expected_coeffs.min().item())) < 1e-6
    assert abs(raw_coeff_max - float(expected_coeffs.max().item())) < 1e-6


def test_safe_atanh_stays_finite_at_and_beyond_bounds():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    y = safe_atanh(x)

    assert torch.isfinite(y).all()
    assert y[0].item() == y[1].item()
    assert y[-1].item() == y[-2].item()
