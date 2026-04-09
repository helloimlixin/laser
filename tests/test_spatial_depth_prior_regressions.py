import torch
from src.models.sparse_token_prior import compute_quantized_rq_losses
from src.models.spatial_prior import SpatialDepthPrior, SpatialDepthPriorConfig


def test_spatial_depth_prior_real_valued_generation_masks_duplicate_atoms():
    torch.manual_seed(0)

    cfg = SpatialDepthPriorConfig(
        vocab_size=3,
        H=1,
        W=1,
        D=3,
        d_model=6,
        n_heads=2,
        n_spatial_layers=1,
        n_depth_layers=1,
        d_ff=12,
        dropout=0.0,
    )
    model = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    atom_ids, coeffs = model.generate(batch_size=32, show_progress=False)

    assert atom_ids.shape == (32, 1, 3)
    assert coeffs.shape == (32, 1, 3)
    for sample in atom_ids[:, 0]:
        assert torch.unique(sample).numel() == cfg.D


def test_spatial_depth_prior_forward_soft_clamps_real_coeff_predictions():
    cfg = SpatialDepthPriorConfig(
        vocab_size=4,
        H=1,
        W=1,
        D=2,
        d_model=8,
        n_heads=2,
        n_spatial_layers=0,
        n_depth_layers=0,
        d_ff=16,
        dropout=0.0,
        coeff_max=2.5,
    )
    model = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.coeff_head[2].bias.fill_(1e3)

    atom_ids = torch.tensor([[[1, 2]]], dtype=torch.long)
    coeffs = torch.zeros(1, 1, 2)

    _, coeff_pred = model(atom_ids, coeffs)

    assert coeff_pred.shape == (1, 1, 2)
    assert torch.isfinite(coeff_pred).all()
    assert torch.all(coeff_pred.abs() <= cfg.coeff_max + 1e-6)
    assert torch.allclose(
        coeff_pred.abs(),
        torch.full_like(coeff_pred.abs(), cfg.coeff_max),
        atol=1e-4,
    )


def test_quantized_rq_loss_weights_change_total_loss():
    per_token_ce = torch.tensor(
        [[[1.0, 5.0, 3.0, 7.0]]],
        dtype=torch.float32,
    )

    token_ce, atom_ce, coeff_ce, total = compute_quantized_rq_losses(
        per_token_ce,
        atom_loss_weight=2.0,
        coeff_loss_weight=0.25,
    )

    assert torch.isclose(token_ce, torch.tensor(4.0))
    assert torch.isclose(atom_ce, torch.tensor(2.0))
    assert torch.isclose(coeff_ce, torch.tensor(6.0))
    assert torch.isclose(total, torch.tensor((2.0 * 2.0 + 0.25 * 6.0) / 2.25))
    assert not torch.isclose(total, token_ce)


def test_quantized_rq_loss_defaults_preserve_unweighted_mean():
    per_token_ce = torch.tensor(
        [[[1.0, 5.0, 3.0, 7.0]]],
        dtype=torch.float32,
    )

    token_ce, atom_ce, coeff_ce, total = compute_quantized_rq_losses(
        per_token_ce,
        atom_loss_weight=1.0,
        coeff_loss_weight=1.0,
    )

    assert torch.isclose(atom_ce, torch.tensor(2.0))
    assert torch.isclose(coeff_ce, torch.tensor(6.0))
    assert torch.isclose(token_ce, torch.tensor(4.0))
    assert torch.isclose(total, token_ce)


def test_spatial_depth_prior_quantized_generation_preserves_parity_and_unique_atoms():
    torch.manual_seed(0)

    cfg = SpatialDepthPriorConfig(
        vocab_size=5,
        atom_vocab_size=3,
        coeff_vocab_size=2,
        H=1,
        W=1,
        D=4,
        real_valued_coeffs=False,
        d_model=8,
        n_heads=2,
        n_spatial_layers=0,
        n_depth_layers=0,
        d_ff=16,
        dropout=0.0,
    )
    model = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    tokens = model.generate(batch_size=32, show_progress=False)

    assert tokens.shape == (32, 1, 4)
    atom_tokens = tokens[:, 0, 0::2]
    coeff_tokens = tokens[:, 0, 1::2]
    assert torch.all((atom_tokens >= 0) & (atom_tokens < cfg.atom_vocab_size))
    assert torch.all((coeff_tokens >= cfg.atom_vocab_size) & (coeff_tokens < cfg.vocab_size))
    for sample in atom_tokens:
        assert torch.unique(sample).numel() == sample.numel()


def test_spatial_depth_prior_quantized_training_masks_previously_used_atoms():
    cfg = SpatialDepthPriorConfig(
        vocab_size=5,
        atom_vocab_size=3,
        coeff_vocab_size=2,
        H=1,
        W=1,
        D=4,
        real_valued_coeffs=False,
        d_model=8,
        n_heads=2,
        n_spatial_layers=0,
        n_depth_layers=0,
        d_ff=16,
        dropout=0.0,
    )
    model = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    tokens = torch.tensor([[[1, 3, 2, 4]]], dtype=torch.long)

    logits = model(tokens)

    assert logits.shape == (1, 1, 4, cfg.vocab_size)
    assert not torch.isfinite(logits[0, 0, 2, 1])
    assert torch.isfinite(logits[0, 0, 2, 0])
    assert torch.isfinite(logits[0, 0, 2, 2])
    assert not torch.isfinite(logits[0, 0, 1, 0])
    assert not torch.isfinite(logits[0, 0, 1, 2])
    assert torch.isfinite(logits[0, 0, 1, 3])
    assert torch.isfinite(logits[0, 0, 1, 4])


def test_spatial_depth_prior_quantized_rejects_too_few_atoms_for_unique_support():
    try:
        SpatialDepthPrior(
            SpatialDepthPriorConfig(
                vocab_size=3,
                atom_vocab_size=1,
                coeff_vocab_size=2,
                H=1,
                W=1,
                D=4,
                real_valued_coeffs=False,
                d_model=8,
                n_heads=2,
                n_spatial_layers=0,
                n_depth_layers=0,
                d_ff=16,
                dropout=0.0,
            )
        )
    except ValueError as exc:
        assert "unique atom id per atom step" in str(exc)
    else:
        raise AssertionError("Expected ValueError for impossible quantized atom support config")


def test_spatial_depth_prior_global_spatial_tokens_load_old_checkpoints():
    old_cfg = SpatialDepthPriorConfig(
        vocab_size=4,
        H=1,
        W=2,
        D=2,
        d_model=8,
        n_heads=2,
        n_spatial_layers=1,
        n_depth_layers=1,
        d_ff=16,
        dropout=0.0,
        n_global_spatial_tokens=0,
    )
    new_cfg = SpatialDepthPriorConfig(
        vocab_size=4,
        H=1,
        W=2,
        D=2,
        d_model=8,
        n_heads=2,
        n_spatial_layers=1,
        n_depth_layers=1,
        d_ff=16,
        dropout=0.0,
        n_global_spatial_tokens=3,
    )
    old_model = SpatialDepthPrior(old_cfg).eval()
    new_model = SpatialDepthPrior(new_cfg).eval()
    expected_global = new_model.global_spatial_tokens.detach().clone()

    new_model.load_state_dict(old_model.state_dict())

    assert new_model.global_spatial_tokens is not None
    assert torch.allclose(new_model.global_spatial_tokens, expected_global)


def test_spatial_depth_prior_global_spatial_tokens_influence_first_position_logits():
    cfg = SpatialDepthPriorConfig(
        vocab_size=4,
        H=1,
        W=2,
        D=1,
        d_model=4,
        n_heads=1,
        n_spatial_layers=1,
        n_depth_layers=0,
        n_global_spatial_tokens=1,
        d_ff=8,
        dropout=0.0,
    )
    model = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        blk = model.spatial_blocks[0]
        blk.ln1.weight.fill_(1.0)
        blk.ln2.weight.fill_(1.0)
        model.spatial_ln.weight.fill_(1.0)
        model.depth_ln.weight.fill_(1.0)
        blk.attn.qkv.weight.zero_()
        blk.attn.qkv.weight[2 * cfg.d_model:3 * cfg.d_model].copy_(torch.eye(cfg.d_model))
        blk.attn.out_proj.weight.copy_(torch.eye(cfg.d_model))
        for param in blk.ffn.parameters():
            param.zero_()
        model.token_head.weight.copy_(torch.eye(cfg.vocab_size, cfg.d_model))
        model.token_head.bias.zero_()

    tokens = torch.zeros(1, cfg.H * cfg.W, cfg.D, dtype=torch.long)
    coeffs = torch.zeros(1, cfg.H * cfg.W, cfg.D)

    with torch.no_grad():
        model.global_spatial_tokens.zero_()
        logits_a, _ = model(tokens, coeffs)
        model.global_spatial_tokens[0, 0].copy_(torch.tensor([2.0, 0.0, -2.0, 1.0]))
        logits_b, _ = model(tokens, coeffs)

    assert not torch.allclose(logits_a[0, 0, 0], logits_b[0, 0, 0])
