import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


scratch_laser = _load_module("scratch_laser_test_module", "scratch/laser.py")
scratch_laser_transformer = _load_module(
    "scratch_laser_transformer_test_module",
    "scratch/laser_transformer.py",
)
sys.modules["laser_transformer"] = scratch_laser_transformer
scratch_laser_diffusion_prior = _load_module(
    "scratch_laser_diffusion_prior_test_module",
    "scratch/laser_diffusion_prior.py",
)
sys.modules["laser_diffusion_prior"] = scratch_laser_diffusion_prior

DictionaryLearning = scratch_laser.DictionaryLearning
LASER = scratch_laser.LASER
Prior = scratch_laser.Prior
PriorConfig = scratch_laser.PriorConfig
Stage2Module = scratch_laser.Stage2Module
build_sparse_site_tokenizer = scratch_laser.build_sparse_site_tokenizer
SpatialDepthPrior = scratch_laser_transformer.SpatialDepthPrior
SpatialDepthPriorConfig = scratch_laser_transformer.SpatialDepthPriorConfig


def test_spatial_depth_prior_uses_previous_coefficients():
    torch.manual_seed(0)

    cfg = SpatialDepthPriorConfig(
        vocab_size=8,
        H=1,
        W=2,
        D=2,
        d_model=8,
        n_heads=2,
        n_spatial_layers=0,
        n_depth_layers=0,
        d_ff=16,
        dropout=0.0,
    )
    model = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        model.token_emb.weight.zero_()
        model.coeff_proj.weight.copy_(torch.arange(1, cfg.d_model + 1, dtype=torch.float32).view(cfg.d_model, 1))
        model.coeff_proj.bias.zero_()

    atom_ids = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.long)
    coeffs_a = torch.zeros(1, cfg.H * cfg.W, cfg.D)
    coeffs_b = coeffs_a.clone()
    coeffs_b[0, 0, 0] = 1.0

    with torch.no_grad():
        logits_a, coeff_pred_a = model(atom_ids, coeffs_a)
        logits_b, coeff_pred_b = model(atom_ids, coeffs_b)

    assert not torch.allclose(logits_a[0, 0, 1], logits_b[0, 0, 1])
    assert not torch.allclose(logits_a[0, 1, 0], logits_b[0, 1, 0])
    assert not torch.allclose(coeff_pred_a[0, 0, 1], coeff_pred_b[0, 0, 1])
    assert not torch.allclose(coeff_pred_a[0, 1, 0], coeff_pred_b[0, 1, 0])


def test_spatial_depth_generation_masks_duplicate_atoms_per_location():
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

    atom_ids, coeffs = model.generate(batch_size=4, show_progress=False)

    assert atom_ids.shape == (4, 1, 3)
    assert coeffs.shape == (4, 1, 3)
    for sample in atom_ids[:, 0]:
        assert torch.unique(sample).numel() == cfg.D


def test_prior_generate_masks_special_tokens_even_without_coeff_prediction():
    torch.manual_seed(0)

    cfg = PriorConfig(
        vocab_size=6,
        H=1,
        W=2,
        D=1,
        predict_coefficients=False,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        dropout=0.0,
    )
    prior = Prior(cfg, bos_token_id=4, pad_token_id=5).eval()
    with torch.no_grad():
        for param in prior.parameters():
            param.zero_()

    toks = prior.generate(batch_size=8, show_progress=False)

    assert toks.shape == (8, 2)
    assert not torch.isin(toks, torch.tensor([4, 5])).any()


def test_sparse_bottleneck_encode_decode_smoke_with_duplicate_atoms():
    torch.manual_seed(0)

    bottleneck = DictionaryLearning(
        num_embeddings=3,
        embedding_dim=2,
        sparsity_level=2,
        quantize_sparse_coeffs=False,
        epsilon=1e-6,
    )
    with torch.no_grad():
        bottleneck.dictionary.copy_(
            torch.tensor(
                [
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    z_e = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32)

    atom_ids, coeffs = bottleneck._encode(z_e)
    z_q = bottleneck._decode(atom_ids, coeffs)
    z_q_ste, loss, tokens = bottleneck(z_e)

    assert atom_ids.shape == (1, 1, 1, 2)
    assert coeffs.shape == (1, 1, 1, 2)
    assert tokens.shape == atom_ids.shape
    assert torch.isfinite(coeffs).all()
    assert torch.isfinite(z_q).all()
    assert torch.isfinite(z_q_ste).all()
    assert torch.isfinite(loss)


def test_sparse_bottleneck_project_codes_recovers_valid_support():
    torch.manual_seed(0)

    bottleneck = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=2,
        sparsity_level=2,
        quantize_sparse_coeffs=False,
        epsilon=1e-6,
    )
    with torch.no_grad():
        bottleneck.dictionary.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, -1.0],
                ]
            )
        )

    support = torch.tensor([[[[0, 0]]]], dtype=torch.long)
    coeffs = torch.tensor([[[[2.0, -1.0]]]], dtype=torch.float32)

    proj_support, proj_coeffs = bottleneck.project_codes(support, coeffs)
    proj_latent = bottleneck._decode(proj_support, proj_coeffs)

    assert proj_support.shape == support.shape
    assert proj_coeffs.shape == coeffs.shape
    assert proj_latent.shape == (1, 2, 1, 1)
    assert torch.isfinite(proj_coeffs).all()
    assert torch.isfinite(proj_latent).all()
    assert ((proj_support >= 0) & (proj_support < bottleneck.num_embeddings)).all()


def test_build_sparse_site_tokenizer_round_trips_sparse_sites():
    tokens_flat = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.int32,
    )
    coeffs_flat = torch.tensor(
        [
            [-1.0, 1.0, 0.0, 0.5],
            [1.0, -1.0, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    site_tokens, tokenizer, oov_rate = build_sparse_site_tokenizer(
        tokens_flat=tokens_flat,
        coeffs_flat=coeffs_flat,
        H=1,
        W=2,
        D=2,
        num_atoms=4,
        coeff_bins=5,
        coeff_max=1.0,
        coeff_quantization="uniform",
        coeff_mu=255.0,
        vocab_size=8,
        chunk_images=1,
    )

    assert site_tokens.shape == (2, 2)
    assert tokenizer.num_site_tokens == 4
    assert oov_rate == 0.0

    dec_atom_ids, dec_coeffs = tokenizer.decode_tokens(site_tokens.view(2, 1, 2, 1))
    assert dec_atom_ids.shape == (2, 1, 2, 2)
    assert dec_coeffs.shape == (2, 1, 2, 2)
    assert torch.equal(dec_atom_ids.view(2, 4), tokens_flat.to(torch.long))
    assert torch.allclose(dec_coeffs.view(2, 4), coeffs_flat, atol=1e-5)


def test_stage2_latent_projection_helper_returns_finite_latents():
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
        quantize_sparse_coeffs=False,
    )
    cfg = PriorConfig(
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
    )
    prior = Prior(
        cfg,
        bos_token_id=laser.bottleneck.bos_token_id,
        pad_token_id=laser.bottleneck.pad_token_id,
    )
    module = Stage2Module(
        transformer=prior,
        lr=1e-3,
        pad_token_id=laser.bottleneck.pad_token_id,
        out_dir=str(ROOT / "tests" / "artifacts"),
        laser=laser,
        H=1,
        W=1,
        D=2,
        image_size=4,
        sample_every_steps=0,
        coeff_mean=0.0,
        coeff_std=1.0,
        latent_loss_weight=0.25,
    )
    module.to(torch.device("cpu"))

    atom_logits = torch.randn(1, 2, laser.bottleneck.vocab_size)
    coeff_pred = torch.randn(1, 2)
    atom_ids = torch.tensor([[[0, 1]]], dtype=torch.long)
    coeffs_raw = torch.tensor([[[0.5, -0.25]]], dtype=torch.float32)

    pred_latent = module._soft_code_to_latent(atom_logits, coeff_pred)
    target_latent = module._target_code_to_latent(atom_ids, coeffs_raw)

    assert pred_latent.shape == target_latent.shape
    assert torch.isfinite(pred_latent).all()
    assert torch.isfinite(target_latent).all()


def test_stage2_coefficient_energy_loss_penalizes_low_amplitude_predictions():
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
        quantize_sparse_coeffs=False,
    )
    cfg = PriorConfig(
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
    )
    prior = Prior(
        cfg,
        bos_token_id=laser.bottleneck.bos_token_id,
        pad_token_id=laser.bottleneck.pad_token_id,
    )
    module = Stage2Module(
        transformer=prior,
        lr=1e-3,
        pad_token_id=laser.bottleneck.pad_token_id,
        out_dir=str(ROOT / "tests" / "artifacts"),
        laser=laser,
        H=1,
        W=1,
        D=2,
        image_size=4,
        sample_every_steps=0,
        coeff_mean=0.0,
        coeff_std=1.0,
        coeff_norm_clip=100.0,
        coeff_energy_loss_weight=0.25,
    )
    module.to(torch.device("cpu"))

    coeff_target_raw = torch.tensor([[[1.0, -2.0]]], dtype=torch.float32)
    zero_pred_norm = torch.zeros_like(coeff_target_raw)

    zero_loss, zero_pred_energy, target_energy = module._coefficient_energy_loss(
        zero_pred_norm,
        coeff_target_raw,
    )
    match_loss, match_pred_energy, match_target_energy = module._coefficient_energy_loss(
        coeff_target_raw,
        coeff_target_raw,
    )

    assert zero_loss.item() > 0.0
    assert match_loss.item() < zero_loss.item()
    assert torch.isclose(zero_pred_energy, torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(target_energy, torch.tensor(3.0), atol=1e-6)
    assert torch.isclose(match_pred_energy, match_target_energy, atol=1e-3)


def test_stage2_spatial_depth_sampling_projects_each_site():
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
        quantize_sparse_coeffs=False,
    )
    cfg = SpatialDepthPriorConfig(
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
        coeff_max=3.0,
    )
    prior = SpatialDepthPrior(cfg).eval()
    with torch.no_grad():
        for param in prior.parameters():
            param.zero_()

    module = Stage2Module(
        transformer=prior,
        lr=1e-3,
        pad_token_id=0,
        out_dir=str(ROOT / "tests" / "artifacts"),
        laser=laser,
        H=1,
        W=2,
        D=2,
        image_size=4,
        sample_every_steps=0,
        sample_batch_size=2,
        coeff_mean=0.0,
        coeff_std=1.0,
    )
    module.to(torch.device("cpu"))

    original_project_codes = module.laser.project_codes
    call_count = {"n": 0}

    def counting_project_codes(atom_ids, coeffs):
        call_count["n"] += 1
        return original_project_codes(atom_ids, coeffs)

    module.laser.project_codes = counting_project_codes

    gen, raw_gen = module._sample_spatial_depth_batch(step=1, capture_raw=True)

    assert raw_gen is not None
    atom_ids, coeffs = gen
    raw_atom_ids, raw_coeffs = raw_gen
    assert atom_ids.shape == (2, cfg.H * cfg.W, cfg.D)
    assert coeffs.shape == (2, cfg.H * cfg.W, cfg.D)
    assert raw_atom_ids.shape == atom_ids.shape
    assert raw_coeffs.shape == coeffs.shape
    assert call_count["n"] == cfg.H * cfg.W


def test_stage2_sparse_debug_snapshot_flags_duplicate_and_projected_codes(tmp_path):
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
        quantize_sparse_coeffs=False,
    )
    with torch.no_grad():
        laser.bottleneck.dictionary.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, -1.0],
                ]
            )
        )
    cfg = PriorConfig(
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
    )
    prior = Prior(
        cfg,
        bos_token_id=laser.bottleneck.bos_token_id,
        pad_token_id=laser.bottleneck.pad_token_id,
    )
    module = Stage2Module(
        transformer=prior,
        lr=1e-3,
        pad_token_id=laser.bottleneck.pad_token_id,
        out_dir=str(tmp_path),
        laser=laser,
        H=1,
        W=1,
        D=2,
        image_size=4,
        sample_every_steps=0,
        coeff_mean=0.0,
        coeff_std=1.0,
        dump_sparse_debug=True,
        sparse_debug_topk=4,
    )
    module.to(torch.device("cpu"))

    raw_atom_ids = torch.tensor([[[0, 0]]], dtype=torch.long)
    raw_coeff_norm = torch.tensor([[[2.0, -1.0]]], dtype=torch.float32)
    raw_imgs = laser.decode(
        raw_atom_ids.view(1, 1, 1, 2),
        raw_coeff_norm.view(1, 1, 1, 2),
    )

    snapshot = module._build_sparse_debug_snapshot((raw_atom_ids, raw_coeff_norm), raw_imgs)

    assert snapshot is not None
    assert snapshot["summary"]["duplicate_rate"] > 0.0
    assert snapshot["summary"]["atom_changed_rate"] > 0.0
    assert snapshot["summary"]["coeff_change_max"] > 0.0
    assert snapshot["summary"]["projection_image_l1_mean"] > 0.0
    assert len(snapshot["top_sites"]) == 1
    site = snapshot["top_sites"][0]
    assert site["duplicate"] is True
    assert site["atom_changed"] is True

    module._dump_sparse_debug(step=7, gen=(raw_atom_ids, raw_coeff_norm), raw_imgs=raw_imgs)

    assert (tmp_path / "stage2_step000007_sparse_debug.pt").exists()
    assert (tmp_path / "stage2_step000007_sparse_debug.txt").exists()
    assert (tmp_path / "stage2_step000007_sparse_debug_raw.png").exists()
    assert (tmp_path / "stage2_step000007_sparse_debug_projected.png").exists()
    assert (tmp_path / "stage2_step000007_sparse_debug_absdiff.png").exists()
    assert (tmp_path / "stage2_step000007_sparse_debug_compare.png").exists()
