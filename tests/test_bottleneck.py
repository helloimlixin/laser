import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path_str in (str(ROOT), str(SRC)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from src.models.bottleneck import (
    DictionaryLearning,
    SparseCodes,
    VectorQuantizer,
    VectorQuantizerEMA,
)


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
    (dl_out.mean() + dl_loss).backward()
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


def test_one_shot_topk_support_matches_abs_correlations_on_orthogonal_dictionary():
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

    support, values = dl.batch_topk_with_support(signals, dictionary)

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

def test_dictionary_learning_tolerates_removed_omp_kwargs():
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
        fast_omp=False,
        omp_diag_eps=1e-4,
        omp_cholesky_eps=1e-6,
        sparse_coding_scheme="lista",
        lista_steps=3,
        lista_step_size_init=0.1,
        lista_threshold_init=0.05,
        dictionary_update_mode="ema",
        dict_ema_decay=0.95,
        dict_ema_eps=1e-6,
    )

    assert dl.num_embeddings == 8


def test_batch_omp_with_coef_max_hard_bounds_coefficients():
    dl = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=2,
        coef_max=1.0,
        bounded_omp_refine_steps=16,
    )
    dictionary = torch.eye(2)
    signals = torch.tensor([[3.0], [-2.0]], dtype=torch.float32)

    support, values = dl.batch_omp_with_support(signals, dictionary)

    assert support.tolist() == [[0, 1]]
    assert torch.allclose(values, torch.tensor([[1.0, -1.0]]), atol=1e-4)
    assert float(values.abs().max()) <= 1.0 + 1e-6


def test_dictionary_learning_forward_hard_bounds_patch_coefficients():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=1,
        sparsity_level=4,
        patch_based=True,
        patch_size=2,
        patch_stride=2,
        coef_max=1.0,
        bounded_omp_refine_steps=16,
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(4))

    z = torch.tensor(
        [[[[2.0, -1.5],
           [0.25, -3.0]]]],
        dtype=torch.float32,
    )

    z_out, loss, sparse_codes = dl(z)

    expected = z.clamp(-1.0, 1.0)
    assert torch.isfinite(loss)
    assert torch.allclose(z_out, expected, atol=1e-4)
    assert float(sparse_codes.values.abs().max()) <= 1.0 + 1e-6
    assert float(dl._last_diag["coeff_abs_max"]) <= 1.0 + 1e-6
    assert float(dl._last_diag["coeff_clip_frac"]) > 0.0


def test_patch_dictionary_learning_preserves_latent_shape():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=4,
        sparsity_level=2,
        patch_based=True,
        patch_size=2,
        patch_stride=1,
    )
    z = torch.randn(2, 4, 4, 4, requires_grad=True)

    z_out, loss, sparse_codes = dl(z)

    assert z_out.shape == z.shape
    assert torch.isfinite(loss)
    assert sparse_codes.support.shape == (2, 4, 4, 2)
    assert sparse_codes.values.shape == sparse_codes.support.shape
    (z_out.mean() + loss).backward()
    assert dl.dictionary.grad is not None


def test_patch_dictionary_learning_hann_reconstructs_exact_signal_with_full_identity_dictionary():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=1,
        sparsity_level=16,
        patch_based=True,
        patch_size=4,
        patch_stride=2,
        patch_reconstruction="hann",
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


def test_dictionary_learning_tokens_to_latent_decodes_quantized_per_site_tokens():
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

    z = dl.tokens_to_latent(
        tokens,
        atom_vocab_size=4,
        coeff_vocab_size=3,
        coeff_bin_values=coeff_bin_values,
    )

    expected = torch.tensor([[[[2.0]], [[0.0]], [[-1.0]], [[0.0]]]])
    assert z.shape == expected.shape
    assert torch.allclose(z, expected, atol=1e-6)


def test_dictionary_learning_sparse_codes_to_tokens_quantizes_support_and_values():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=4,
        sparsity_level=2,
        patch_based=False,
    )
    sparse_codes = SparseCodes(
        support=torch.tensor([[[[1, 3]]]], dtype=torch.long),
        values=torch.tensor([[[[-2.0, 1.6]]]], dtype=torch.float32),
        num_embeddings=4,
    )

    tokens, coeff_q = dl.sparse_codes_to_tokens(
        sparse_codes,
        coeff_vocab_size=5,
        coeff_max=2.0,
        coeff_quantization="uniform",
    )

    assert tokens.shape == (1, 1, 1, 4)
    assert tokens.tolist() == [[[[1, 4, 3, 8]]]]
    assert torch.allclose(coeff_q, torch.tensor([[[[-2.0, 2.0]]]]))


def test_patch_dictionary_learning_tokens_to_latent_decodes_quantized_patch_tokens():
    dl = DictionaryLearning(
        num_embeddings=16,
        embedding_dim=1,
        sparsity_level=16,
        patch_based=True,
        patch_size=4,
        patch_stride=2,
        patch_reconstruction="hann",
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(16))

    z = torch.tensor(
        [[[[0.0, 1.0, 0.0, 1.0],
           [1.0, 0.0, 1.0, 0.0],
           [0.0, 1.0, 0.0, 1.0],
           [1.0, 0.0, 1.0, 0.0]]]],
        dtype=torch.float32,
    )
    patches, nph, npw, height, width = dl._extract_patches(z)
    coeffs = patches.permute(0, 2, 1).reshape(-1, 16).to(torch.long)
    support = torch.arange(16, dtype=torch.long).view(1, 1, 16).expand(coeffs.size(0), -1, -1)
    support = support.squeeze(1).view(1, nph, npw, 16)
    coeffs = coeffs.view(1, nph, npw, 16)
    tokens = torch.empty(1, nph, npw, 32, dtype=torch.long)
    tokens[..., 0::2] = support
    tokens[..., 1::2] = coeffs + 16

    z_out = dl.tokens_to_latent(
        tokens,
        latent_hw=(height, width),
        atom_vocab_size=16,
        coeff_vocab_size=2,
        coeff_bin_values=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )

    assert z_out.shape == z.shape
    assert torch.allclose(z_out, z, atol=1e-5)


def test_patch_dictionary_learning_dispatches_to_omp_not_topk():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=1,
        sparsity_level=1,
        patch_based=True,
        patch_size=2,
        patch_stride=1,
    )
    z = torch.randn(1, 1, 2, 2)
    calls = {"omp": 0, "topk": 0}

    def _omp(X, D):
        del X, D
        calls["omp"] += 1
        return (
            torch.zeros(4, 1, dtype=torch.long),
            torch.zeros(4, 1, dtype=torch.float32),
        )

    def _topk(X, D):
        del X, D
        calls["topk"] += 1
        raise AssertionError("patch path should not dispatch to top-k")

    dl.batch_omp_with_support = _omp
    dl.batch_topk_with_support = _topk

    z_out, loss, sparse_codes = dl(z)

    assert torch.isfinite(loss)
    assert z_out.shape == z.shape
    assert sparse_codes.support.shape == (1, 2, 2, 1)
    assert calls == {"omp": 1, "topk": 0}
