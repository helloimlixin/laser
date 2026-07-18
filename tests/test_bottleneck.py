import pytest
import torch


from src.models.bottleneck import (
    DictionaryLearning,
    SparseCodes,
    VectorQuantizer,
    VectorQuantizerEMA,
)
from src.sparse_token_codec import sparse_codes_to_tokens, tokens_to_sparse_codes


def test_bottleneck_smoke():
    torch.manual_seed(0)

    # VectorQuantizer: basic shape + gradient check.
    vq_input = torch.randn(2, 4, 8, 8, requires_grad=True)
    vq = VectorQuantizer(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    vq_out, vq_loss, _, _ = vq(vq_input)
    assert vq_out.shape == vq_input.shape
    assert torch.isfinite(vq_loss)
    (vq_out.mean() + vq_loss).backward()
    assert vq.embedding.weight.grad is not None

    # VectorQuantizerEMA: basic shape + no codebook gradients.
    vq_ema_input = torch.randn(2, 4, 8, 8, requires_grad=True)
    vq_ema = VectorQuantizerEMA(
        num_embeddings=8,
        embedding_dim=4,
        commitment_cost=0.25,
        ema_decay=0.99,
    )
    vq_ema_out, vq_ema_loss, _, _ = vq_ema(vq_ema_input)
    assert vq_ema_out.shape == vq_ema_input.shape
    assert torch.isfinite(vq_ema_loss)
    (vq_ema_out.mean() + vq_ema_loss).backward()
    assert vq_ema.embedding.weight.grad is None

    # DictionaryLearning: per-pixel sparse coding smoke test.
    dl_input = torch.randn(2, 4, 8, 8, requires_grad=True)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
    )
    dl_out, dl_loss, sparse_codes = dl(dl_input)
    assert dl_out.shape == dl_input.shape
    assert torch.isfinite(dl_loss)
    expected_n = dl_input.shape[0] * dl_input.shape[2] * dl_input.shape[3]
    assert isinstance(sparse_codes, SparseCodes)
    assert sparse_codes.support.shape == (
        dl_input.shape[0],
        dl_input.shape[2],
        dl_input.shape[3],
        dl.sparsity_level,
    )
    assert sparse_codes.values.shape == sparse_codes.support.shape
    assert sparse_codes.num_embeddings == dl.num_embeddings
    assert sparse_codes.support.view(expected_n, dl.sparsity_level).dtype == torch.long
    assert torch.allclose(dl_loss.detach(), dl._last_bottleneck_loss)
    assert torch.allclose(
        dl._last_bottleneck_objective,
        dl._last_dictionary_loss + dl._last_bottleneck_loss,
    )
    (dl_out.mean() + dl._last_bottleneck_objective_for_backward).backward()
    assert dl.dictionary.grad is not None


def test_dictionary_learning_normalizes_and_projects_dictionary():
    torch.manual_seed(0)
    dl = DictionaryLearning(num_embeddings=6, embedding_dim=4, sparsity_level=2)

    with torch.no_grad():
        dl.dictionary.mul_(3.0)
    dl.normalize_dictionary_()
    norm_error = dl.dictionary.norm(dim=0) - 1.0
    assert torch.allclose(norm_error, torch.zeros_like(norm_error), atol=1e-5)

    atoms = dl.dictionary.detach().clone()
    radial_grad = atoms * 2.0
    tangential_grad = torch.randn_like(atoms)
    dl.dictionary.grad = radial_grad + tangential_grad
    dl.project_dictionary_gradient_()

    projected_grad = dl.dictionary.grad
    radial_component = (atoms * projected_grad).sum(dim=0)
    assert torch.allclose(
        radial_component,
        torch.zeros_like(radial_component),
        atol=1e-5,
    )


def test_dictionary_data_sampling_jitters_short_batches_instead_of_duplicates():
    torch.manual_seed(0)
    dl = DictionaryLearning(num_embeddings=8, embedding_dim=4, sparsity_level=2)
    signals = torch.eye(4, 2)

    atoms = dl._sample_atoms_from_signals(signals, dl.num_embeddings)

    assert atoms.shape == (4, 8)
    assert torch.isfinite(atoms).all()
    assert torch.allclose(atoms.norm(dim=0), torch.ones(8), atol=1e-5)
    cosine = atoms.t() @ atoms
    off_diag = cosine - torch.eye(8)
    assert float(off_diag.abs().max()) < 0.9999


def test_dictionary_learning_data_initializes_from_first_batch():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=2,
        sparsity_level=1,
        data_init_from_first_batch=True,
    )
    before = dl.dictionary.detach().clone()
    z = torch.randn(1, 2, 2, 4)

    dl(z)

    assert bool(dl._data_initialized.item())
    assert not torch.allclose(dl.dictionary.detach(), before)
    assert torch.allclose(
        dl.dictionary.detach().norm(dim=0),
        torch.ones(dl.num_embeddings),
        atol=1e-5,
    )


def test_dictionary_learning_revives_dead_atoms_after_optimizer_step():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=1,
        dead_atom_revival=True,
        dead_atom_revival_interval=1,
        dead_atom_revival_max_fraction=0.5,
        dead_atom_revival_noise=0.0,
        dead_atom_revival_patience=1,
    )
    dl.train()
    with torch.no_grad():
        dictionary = torch.zeros_like(dl.dictionary)
        dictionary[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for col in range(1, dl.num_embeddings):
            dictionary[(col - 1) % 3 + 1, col] = 1.0
        dl.dictionary.copy_(dictionary)
        before = dl.dictionary.detach().clone()

    z = torch.zeros(2, 4, 2, 2)
    z[:, 0] = 1.0
    _z_out, _loss, codes = dl(z)

    assert set(codes.support.reshape(-1).tolist()) == {0}
    revived = dl.revive_dead_atoms_after_step_()

    assert revived == 4
    assert int(dl._last_dead_atom_count.item()) == 7
    assert int(dl._last_revived_atom_count.item()) == 4
    assert torch.allclose(
        dl.dictionary.detach().norm(dim=0),
        torch.ones(dl.num_embeddings),
        atol=1e-5,
    )
    changed = (dl.dictionary.detach() - before).abs().sum(dim=0) > 1e-6
    assert int(changed.sum().item()) >= revived


def test_ste_keeps_encoder_grad_and_loss_trains_dictionary():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
    )
    z = torch.randn(2, 4, 3, 3, requires_grad=True)

    z_out, loss, _ = dl(z)
    # The straight-through estimator routes decoder gradients to the encoder
    # input, while the dictionary atoms learn from the full bottleneck objective.
    (z_out.square().mean() + dl._last_bottleneck_objective_for_backward).backward()

    assert z.grad is not None
    assert z.grad.abs().sum() > 0
    assert dl.dictionary.grad is not None
    assert dl.dictionary.grad.abs().sum() > 0


def test_batch_omp_support_matches_abs_correlations_on_orthogonal_dictionary():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=4,
        sparsity_level=2,
    )
    dictionary = torch.eye(4)
    signals = torch.tensor(
        [
            [3.0, -0.5],
            [-1.0, 2.5],
            [2.0, 0.25],
            [0.5, -4.0],
        ]
    )

    support, values = dl.batch_omp_with_support(signals, dictionary)

    assert support.tolist() == [[0, 2], [3, 1]]
    assert torch.allclose(
        values,
        torch.tensor(
            [
                [3.0, 2.0],
                [-4.0, 2.5],
            ]
        ),
        atol=1e-6,
    )


def test_batch_omp_uses_unclipped_least_squares_coefficients():
    dl = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=2,
    )
    dictionary = torch.eye(2)
    signals = torch.tensor([[3.0], [-2.0]], dtype=torch.float32)

    support, values = dl.batch_omp_with_support(signals, dictionary)

    assert support.tolist() == [[0, 1]]
    assert torch.allclose(values, torch.tensor([[3.0, -2.0]]), atol=1e-4)


def test_batch_omp_fixed_sparsity_does_not_reselect_atoms_on_zero_ties():
    dl = DictionaryLearning(
        num_embeddings=3,
        embedding_dim=3,
        sparsity_level=3,
    )
    dictionary = torch.eye(3)
    signals = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float32)

    support, values = dl.batch_omp_with_support(signals, dictionary)

    assert support.tolist() == [[0, 1, 2]]
    assert torch.allclose(values, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6)


def test_dictionary_learning_forward_uses_unclipped_patch_coefficients():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=1,
        sparsity_level=4,
        patch_based=True,
        patch_size=2,
        patch_stride=2,
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(4))

    z = torch.tensor(
        [[[[2.0, -1.5],
           [0.25, -3.0]]]],
        dtype=torch.float32,
    )

    z_out, loss, sparse_codes = dl(z)

    assert torch.isfinite(loss)
    assert torch.allclose(z_out, z, atol=1e-4)
    assert float(sparse_codes.values.abs().max()) > 1.0


def test_patch_dictionary_learning_rejects_overlapping_stride():
    with pytest.raises(ValueError, match="must equal patch_size"):
        DictionaryLearning(
            num_embeddings=4,
            embedding_dim=1,
            sparsity_level=4,
            patch_based=True,
            patch_size=2,
            patch_stride=1,
        )


def test_patch_dictionary_learning_rejects_overlap_reconstruction_modes():
    with pytest.raises(ValueError, match="must be 'tile'"):
        DictionaryLearning(
            num_embeddings=16,
            embedding_dim=1,
            sparsity_level=4,
            patch_based=True,
            patch_size=4,
            patch_stride=4,
            patch_reconstruction="hann",
        )


def test_patch_dictionary_learning_preserves_latent_shape():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
        patch_based=True,
        patch_size=2,
        patch_stride=2,
    )
    z = torch.randn(2, 4, 4, 4, requires_grad=True)

    z_out, loss, sparse_codes = dl(z)

    assert z_out.shape == z.shape
    assert torch.isfinite(loss)
    assert sparse_codes.support.shape == (2, 2, 2, 2)
    assert sparse_codes.values.shape == sparse_codes.support.shape
    (z_out.mean() + dl._last_bottleneck_objective_for_backward).backward()
    assert dl.dictionary.grad is not None


def test_patch_dictionary_learning_keeps_sparse_solve_finite_for_half_inputs():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=4,
        sparsity_level=4,
        patch_based=True,
        patch_size=2,
        patch_stride=2,
    )
    z = torch.randn(2, 4, 4, 4, dtype=torch.float16, requires_grad=True)

    z_out, loss, sparse_codes = dl(z)

    assert z_out.dtype == z.dtype
    assert sparse_codes.values.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(z_out.float()).all()
    (z_out.float().mean() + dl._last_bottleneck_objective_for_backward).backward()
    assert dl.dictionary.grad is not None
    assert torch.isfinite(dl.dictionary.grad).all()


def test_patch_dictionary_learning_tile_reconstructs_exact_signal_with_full_identity_dictionary():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=1,
        sparsity_level=16,
        patch_based=True,
        patch_size=4,
        patch_stride=4,
        patch_reconstruction="tile",
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(16))

    z = torch.randn(1, 1, 8, 8)
    patches, nph, npw, height, width = dl._extract_patches(z)
    coeffs = patches.permute(0, 2, 1).reshape(-1, 16)
    support = torch.arange(16, dtype=torch.long).view(1, 1, 16).expand(coeffs.size(0), -1, -1)
    support = support.squeeze(1).view(1, nph, npw, 16)
    coeffs = coeffs.view(1, nph, npw, 16)

    z_out = dl._reconstruct_sparse(support, coeffs, height, width)

    assert z_out.shape == z.shape
    assert torch.allclose(z_out, z, atol=1e-5)


def test_ste_keeps_encoder_gradient_path():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
    )
    z = torch.randn(1, 4, 2, 2, requires_grad=True)

    z_out, loss, _sparse_codes = dl(z)
    (z_out.sum() + dl._last_bottleneck_objective_for_backward).backward()

    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0
    assert dl.dictionary.grad is not None
    assert float(dl.dictionary.grad.abs().sum()) > 0.0


def test_patch_toggle_disables_patch_dictionary_learning_even_with_patch_params():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
        patch_based=False,
        patch_size=4,
        patch_stride=2,
    )
    z = torch.randn(2, 4, 4, 4)

    z_out, loss, sparse_codes = dl(z)

    assert torch.isfinite(loss)
    assert z_out.shape == z.shape
    assert sparse_codes.support.shape == (2, 4, 4, 2)


def test_sparse_token_codec_decodes_quantized_per_site_tokens():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=4,
        sparsity_level=2,
        patch_based=False,
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(4))

    tokens = torch.tensor([[[[0, 5, 2, 6]]]], dtype=torch.long)
    coeff_bin_values = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32)

    support, values = tokens_to_sparse_codes(
        tokens,
        num_embeddings=4,
        sparsity_level=2,
        atom_vocab_size=4,
        coeff_vocab_size=3,
        coeff_bin_values=coeff_bin_values,
    )
    z = dl._reconstruct_sparse(support, values, height=1, width=1)

    expected = torch.tensor([[[[2.0]], [[0.0]], [[-1.0]], [[0.0]]]])
    assert z.shape == expected.shape
    assert torch.allclose(z, expected, atol=1e-6)


def test_sparse_token_codec_quantizes_support_and_values():
    sparse_codes = SparseCodes(
        support=torch.tensor([[[[1, 3]]]], dtype=torch.long),
        values=torch.tensor([[[[-2.0, 1.6]]]], dtype=torch.float32),
        num_embeddings=4,
    )

    tokens, coeff_q = sparse_codes_to_tokens(
        sparse_codes,
        num_embeddings=4,
        sparsity_level=2,
        coeff_vocab_size=5,
        coeff_max=2.0,
        coeff_quantization="uniform",
    )

    assert tokens.shape == (1, 1, 1, 4)
    assert tokens.tolist() == [[[[1, 4, 3, 8]]]]
    assert torch.allclose(coeff_q, torch.tensor([[[[-2.0, 2.0]]]]))
