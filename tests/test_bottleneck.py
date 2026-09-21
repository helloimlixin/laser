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


def test_dictionary_learning_delays_and_pools_data_initialization():
    torch.manual_seed(0)
    dl = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=2,
        sparsity_level=1,
        data_init_from_first_batch=True,
        data_init_start_step=2,
        data_init_accumulation_steps=2,
        dictionary_update_mode="alternating_residual",
    )
    before = dl.dictionary.detach().clone()

    dl(torch.randn(1, 2, 1, 4))
    assert dl.alternating_dictionary_update_after_step_() == 0
    dl(torch.randn(1, 2, 1, 4))
    assert dl.alternating_dictionary_update_after_step_() == 0
    assert not bool(dl._data_initialized.item())
    assert torch.equal(dl.dictionary.detach(), before)

    dl(torch.randn(1, 2, 1, 4))
    assert dl.alternating_dictionary_update_after_step_() == 0
    assert not bool(dl._data_initialized.item())
    assert len(dl._data_init_accumulator) == 1

    dl(torch.randn(1, 2, 1, 4))
    assert bool(dl._data_initialized.item())
    assert len(dl._data_init_accumulator) == 0
    assert not torch.allclose(dl.dictionary.detach(), before)


def test_dictionary_learning_resume_does_not_reinitialize_loaded_dictionary():
    torch.manual_seed(0)
    trained = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=2,
        sparsity_level=1,
        data_init_from_first_batch=True,
        dead_atom_revival=True,
    )
    trained(torch.randn(1, 2, 2, 4))
    with torch.no_grad():
        trained._revival_step.fill_(37)
        trained._atom_unused_intervals.copy_(torch.arange(8))

    resumed = DictionaryLearning(
        num_embeddings=8,
        embedding_dim=2,
        sparsity_level=1,
        data_init_from_first_batch=True,
        dead_atom_revival=True,
    )
    resumed.load_state_dict(trained.state_dict())
    loaded_dictionary = resumed.dictionary.detach().clone()

    resumed(torch.randn(1, 2, 2, 4))

    assert bool(resumed._data_initialized.item())
    assert torch.equal(resumed.dictionary.detach(), loaded_dictionary)
    assert int(resumed._revival_step.item()) == 37
    assert torch.equal(resumed._atom_unused_intervals, torch.arange(8))


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


def test_alternating_residual_update_freezes_dictionary_and_improves_fixed_codes():
    dl = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=1,
        dictionary_update_mode="alternating_residual",
        dictionary_update_relaxation=1.0,
        dictionary_update_max_atoms_per_step=2,
        dictionary_update_min_usage=1,
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(2))
    z = torch.tensor(
        [[
            [[1.0, 1.0], [0.4, 0.5]],
            [[0.5, 0.4], [1.0, 1.0]],
        ]],
        requires_grad=True,
    )

    z_out, _loss, _codes = dl(z)
    cached = dl._last_dictionary_update_batch
    signals = cached["signals"]
    support = cached["support"]
    values = cached["values"]

    def fixed_code_error(dictionary):
        atoms = dictionary.t()[support]
        reconstruction = (atoms * values.unsqueeze(-1)).sum(dim=1).t()
        return (signals - reconstruction).square().sum()

    before_dictionary = dl.dictionary.detach().clone()
    before_error = fixed_code_error(before_dictionary)
    (z_out.square().mean() + dl._last_bottleneck_objective_for_backward).backward()

    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0.0
    assert torch.allclose(
        dl._last_bottleneck_objective_for_backward.detach(),
        dl._last_commitment_loss,
    )
    assert not dl.dictionary.requires_grad
    assert dl.dictionary.grad is None

    updated = dl.alternating_dictionary_update_after_step_()
    after_dictionary = dl.dictionary.detach().clone()
    after_error = fixed_code_error(after_dictionary)

    assert updated == 2
    assert not torch.allclose(after_dictionary, before_dictionary)
    assert after_error <= before_error + 1e-6
    assert torch.allclose(
        after_dictionary.norm(dim=0),
        torch.ones(2),
        atol=1e-6,
    )
    assert int(dl._dictionary_update_step.item()) == 1
    assert float(dl._last_dictionary_update_relative_improvement.item()) > 0.0


def test_alternating_residual_update_accumulates_complete_windows():
    dl = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=1,
        dictionary_update_mode="alternating_residual",
        dictionary_update_relaxation=1.0,
        dictionary_update_max_atoms_per_step=2,
        dictionary_update_min_usage=2,
        dictionary_update_accumulation_steps=2,
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(2))
    z = torch.tensor([[[[1.0, 0.4]], [[0.5, 1.0]]]])

    before = dl.dictionary.detach().clone()
    dl(z)
    assert dl.alternating_dictionary_update_after_step_() == 0
    assert torch.equal(dl.dictionary, before)
    assert int(dl._last_dictionary_update_accumulated_step_count.item()) == 1

    dl(z)
    assert dl.alternating_dictionary_update_after_step_() == 2
    assert not torch.equal(dl.dictionary, before)
    assert int(dl._dictionary_update_step.item()) == 2
    assert int(dl._last_dictionary_update_step.item()) == 2
    assert int(dl._last_dictionary_update_accumulated_step_count.item()) == 0

    last_updated = int(dl._last_dictionary_updated_atom_count.item())
    last_improvement = float(dl._last_dictionary_update_relative_improvement.item())
    dl(z)
    assert dl.alternating_dictionary_update_after_step_() == 0
    assert int(dl._last_dictionary_updated_atom_count.item()) == last_updated
    assert float(dl._last_dictionary_update_relative_improvement.item()) == pytest.approx(
        last_improvement
    )


def test_alternating_update_uses_fixed_gather_then_broadcast_protocol(monkeypatch):
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=2,
        sparsity_level=1,
        dictionary_update_mode="alternating_residual",
        dictionary_update_max_atoms_per_step=1,
        dictionary_update_min_usage=1,
    )
    dl._last_dictionary_update_batch = {
        "signals": torch.tensor([[0.1], [1.0]]),
        "support": torch.tensor([[1]]),
        "values": torch.tensor([[1.0]]),
    }
    collective_calls = []

    def fake_all_reduce(tensor, op=None):
        if op == torch.distributed.ReduceOp.SUM:
            tensor.mul_(2)
        collective_calls.append(("reduce", tuple(tensor.shape)))

    def fake_all_gather(outputs, tensor, group=None):
        del group
        collective_calls.append(("gather", tuple(tensor.shape)))
        if tensor.shape == (1,):
            outputs[0].fill_(1)
            outputs[1].fill_(1)
        elif tensor.dtype == torch.long:
            outputs[0].fill_(0)
            outputs[1].copy_(tensor)
        elif tensor.shape == (2, 1):
            outputs[0].copy_(torch.tensor([[1.0], [0.0]]))
            outputs[1].copy_(tensor)
        else:
            outputs[0].fill_(1.0)
            outputs[1].copy_(tensor)

    def fake_broadcast(tensor, src, group=None):
        del src, group
        collective_calls.append(("broadcast", tuple(tensor.shape)))

    monkeypatch.setattr(dl, "_distributed_is_initialized", lambda: True)
    monkeypatch.setattr(dl, "_distributed_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)

    updated = dl.alternating_dictionary_update_after_step_()

    assert updated == 1
    assert collective_calls == [
        ("reduce", ()),
        ("reduce", ()),
        ("gather", (1,)),
        ("gather", (2, 1)),
        ("gather", (1, 1)),
        ("gather", (1, 1)),
        ("broadcast", (2, 4)),
        ("broadcast", (3,)),
    ]


def test_dictionary_learning_rejects_unknown_update_mode():
    with pytest.raises(ValueError, match="dictionary_update_mode"):
        DictionaryLearning(dictionary_update_mode="adam-but-not-really")


def test_dictionary_learning_rejects_unknown_collective_backend():
    with pytest.raises(ValueError, match="dictionary_collective_backend"):
        DictionaryLearning(dictionary_collective_backend="mixed-and-unsafe")


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


def test_coherence_gated_omp_avoids_collinear_support_without_clamping_values():
    unrestricted = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=3,
        sparsity_level=2,
        omp_ridge=0.1,
    )
    conditioned = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=3,
        sparsity_level=2,
        omp_ridge=0.1,
        omp_max_support_coherence=0.25,
    )
    dictionary = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.05, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    dictionary = torch.nn.functional.normalize(dictionary, dim=0)
    signal = torch.tensor([[20.0], [1.0], [0.0]], dtype=torch.float32)

    plain_support, _ = unrestricted.batch_omp_with_support(signal, dictionary)
    support, values = conditioned.batch_omp_with_support(signal, dictionary)

    assert set(plain_support[0].tolist()) == {0, 1}
    assert set(support[0].tolist()) == {1, 2}
    selected = dictionary[:, support[0]]
    assert float((selected[:, 0] @ selected[:, 1]).abs()) <= 0.25
    assert float(values.abs().max()) > 10.0
    assert float(conditioned._last_support_coherence_max) <= 0.25
    assert float(conditioned._last_support_coherence_fallback_fraction) == 0.0


def test_dictionary_learning_validates_support_coherence_bound():
    with pytest.raises(ValueError, match="omp_max_support_coherence"):
        DictionaryLearning(omp_max_support_coherence=0.0)


def test_progressive_dictionary_loss_matches_rqvae_depth_average():
    dl = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=2,
        commitment_cost=1.0,
        progressive_loss=True,
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(2))

    # OMP reconstructs [2, 1] as [2, 0] at depth one and exactly at depth two.
    # The per-element MSEs are therefore 0.5 and 0.0, matching RQ-VAE's mean
    # over cumulative reconstruction depths: (0.5 + 0.0) / 2 = 0.25.
    z = torch.tensor([[[[2.0]], [[1.0]]]], requires_grad=True)
    _, commitment_loss, _ = dl(z)

    assert dl._last_dictionary_loss.item() == pytest.approx(0.25)
    assert dl._last_commitment_loss.item() == pytest.approx(0.25)
    assert dl._last_final_dictionary_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert commitment_loss.item() == pytest.approx(0.25)
    assert dl._last_bottleneck_objective.item() == pytest.approx(0.5)

    dl._last_bottleneck_objective_for_backward.backward()
    assert z.grad is not None
    assert dl.dictionary.grad is not None


def test_nonprogressive_dictionary_loss_uses_final_batch_omp_k_only():
    dl = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=3,
        sparsity_level=2,
        commitment_cost=1.0,
        progressive_loss=False,
    )
    with torch.no_grad():
        dl.dictionary.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                ]
            )
        )

    # Final Batch OMP reconstructs [2, 1, 1] as [2, 1, 0]. The training
    # objective must therefore be its final K=2 MSE, 1 / 3, with no K=1 term.
    z = torch.tensor([[[[2.0]], [[1.0]], [[1.0]]]], requires_grad=True)
    _, commitment_loss, sparse_codes = dl(z)

    assert sparse_codes.support.shape[-1] == 2
    assert dl._last_final_dictionary_loss.item() == pytest.approx(1.0 / 3.0)
    assert dl._last_dictionary_loss.item() == pytest.approx(1.0 / 3.0)
    assert dl._last_commitment_loss.item() == pytest.approx(1.0 / 3.0)
    assert commitment_loss.item() == pytest.approx(1.0 / 3.0)


def test_variance_normalized_commitment_is_scale_invariant():
    base = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=1,
        commitment_cost=1.0,
        commitment_normalize_by_variance=True,
    )
    scaled = DictionaryLearning(
        num_embeddings=2,
        embedding_dim=2,
        sparsity_level=1,
        commitment_cost=1.0,
        commitment_normalize_by_variance=True,
    )
    with torch.no_grad():
        base.dictionary.copy_(torch.eye(2))
        scaled.dictionary.copy_(torch.eye(2))

    signal = torch.tensor([[[[2.0]], [[1.0]]]])
    _, base_loss, _ = base(signal)
    _, scaled_loss, _ = scaled(7.0 * signal)

    assert base_loss.item() == pytest.approx(scaled_loss.item(), rel=1e-5)


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


def test_dictionary_learning_quantizes_and_bounds_codec_coefficients():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=4,
        sparsity_level=4,
        coefficient_quantization_bits=4,
        coefficient_quantization_max=2.0,
    )
    with torch.no_grad():
        dl.dictionary.copy_(torch.eye(4))
    z = torch.tensor([[[[3.0]], [[-1.1]], [[0.31]], [[-0.02]]]])

    _z_out, loss, sparse_codes = dl(z)

    qmax = 2 ** (4 - 1) - 1
    step = 2.0 / qmax
    scaled = sparse_codes.values / step
    assert torch.isfinite(loss)
    assert float(sparse_codes.values.abs().max()) <= 2.0
    assert torch.allclose(scaled, scaled.round(), atol=1e-5)
    assert float(dl._last_coefficient_saturation_fraction) > 0.0


def test_dictionary_learning_requires_bound_for_coefficient_quantizer():
    with pytest.raises(ValueError, match="coefficient_quantization_max"):
        DictionaryLearning(coefficient_quantization_bits=8)


def test_disabled_stage_one_quantizer_preserves_unbounded_coefficients_exactly():
    dl = DictionaryLearning(
        coefficient_quantization_bits=0,
        coefficient_quantization_max=None,
    )
    values = torch.tensor([[-1000.0, -0.125, 0.0, 2400.0]])

    output = dl._quantize_coefficients(values)

    assert torch.equal(output, values)
    assert float(dl._last_coefficient_quantization_fraction) == 0.0
    assert float(dl._last_coefficient_saturation_fraction) == 0.0
    assert float(dl._last_coefficient_abs_p99) > 2000.0
    assert float(dl._last_coefficient_abs_max) == 2400.0


def test_dictionary_learning_coefficient_quantization_curriculum():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=4,
        sparsity_level=2,
        coefficient_quantization_bits=4,
        coefficient_quantization_max=2.0,
        coefficient_quantization_start_step=2,
        coefficient_quantization_warmup_steps=2,
    )
    values = torch.tensor([[0.2, 1.7]], dtype=torch.float32)
    quantized = torch.round(values / (2.0 / 7.0)) * (2.0 / 7.0)

    assert torch.equal(dl._quantize_coefficients(values), values)
    assert float(dl._last_coefficient_quantization_fraction) == 0.0

    dl._dictionary_update_step.fill_(3)
    halfway = dl._quantize_coefficients(values)
    assert torch.allclose(halfway, torch.lerp(values, quantized, 0.5))
    assert float(dl._last_coefficient_quantization_fraction) == pytest.approx(0.5)

    dl._dictionary_update_step.fill_(4)
    assert torch.allclose(dl._quantize_coefficients(values), quantized)
    assert float(dl._last_coefficient_quantization_fraction) == 1.0


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"coefficient_quantization_start_step": -1}, "start_step"),
        ({"coefficient_quantization_warmup_steps": -1}, "warmup_steps"),
    ],
)
def test_dictionary_learning_rejects_negative_quantization_schedule(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        DictionaryLearning(**kwargs)


def test_dictionary_learning_ridge_stabilizes_collinear_omp_coefficients():
    dictionary = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0e-4, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    dictionary = torch.nn.functional.normalize(dictionary, dim=0)
    signal = torch.tensor([[1.0], [0.5], [0.0]], dtype=torch.float32)
    unregularized = DictionaryLearning(
        num_embeddings=3,
        embedding_dim=3,
        sparsity_level=2,
        omp_ridge=0.0,
    )
    regularized = DictionaryLearning(
        num_embeddings=3,
        embedding_dim=3,
        sparsity_level=2,
        omp_ridge=1.0e-2,
    )

    _support_plain, values_plain = unregularized.batch_omp_with_support(
        signal, dictionary
    )
    _support_ridge, values_ridge = regularized.batch_omp_with_support(
        signal, dictionary
    )

    assert torch.isfinite(values_ridge).all()
    assert values_ridge.abs().max() < values_plain.abs().max()


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


def test_dictionary_learning_rejects_unknown_omp_compute_precision():
    with pytest.raises(ValueError, match="omp_compute_precision"):
        DictionaryLearning(omp_compute_precision="float16")


def test_bfloat16_omp_uses_fp32_coefficients_and_preserves_clear_support():
    dl = DictionaryLearning(
        num_embeddings=4,
        embedding_dim=4,
        sparsity_level=2,
        omp_compute_precision="bfloat16",
    )
    dictionary = torch.eye(4)
    signals = torch.tensor(
        [[3.0, 0.0], [0.0, -4.0], [1.5, 2.0], [0.0, 0.5]],
        dtype=torch.float32,
    )

    support, values = dl.batch_omp_with_support(signals, dictionary)

    assert support.tolist() == [[0, 2], [1, 2]]
    assert values.dtype == torch.float32
    assert torch.allclose(values, torch.tensor([[3.0, 1.5], [-4.0, 2.0]]))


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
