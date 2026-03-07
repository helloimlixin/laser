import importlib.util
import sys
from pathlib import Path

import torch
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
LASER = scratch_laser.LASER
Prior = scratch_laser.Prior
PriorConfig = scratch_laser.PriorConfig
train_stage2_transformer = scratch_laser.train_stage2_transformer
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


def test_safe_atanh_stays_finite_at_and_beyond_bounds():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    y = safe_atanh(x)

    assert torch.isfinite(y).all()
    assert y[0].item() == y[1].item()
    assert y[-1].item() == y[-2].item()
