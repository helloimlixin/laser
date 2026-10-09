import copy

import torch
import pytest

from src.training.physical_compound_geometry import compound_energy_distance
from tests.test_physical_compound_prior import setup


def test_energy_is_zero_for_identical_nontrivial_distribution_and_positive_for_wrong_atom():
    dictionary = torch.eye(2)
    bins = torch.linspace(-1, 1, 8)
    target = torch.tensor([[.05, .1, .1, .25, .2, .1, .1, .1]])
    a = torch.zeros(1, 1, requires_grad=True)
    c = target.log()[:, None].clone().requires_grad_(True)
    def score(atom):
        return compound_energy_distance(a, c, torch.tensor([[atom]]), torch.tensor([0]),
                                        target, dictionary, bins, groups=8)
    assert abs(float(score(0))) < 2e-6
    assert float(score(1)) > .01


def test_energy_distinguishes_different_distributions_with_the_same_mean():
    bins = torch.tensor([-1., 0., 1.])
    c = torch.tensor([[[0., -80., 0.]]], requires_grad=True)
    target = torch.tensor([[0., 1., 0.]])
    loss = compound_energy_distance(torch.zeros(1, 1), c, torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, dtype=torch.long), target, torch.ones(1, 1), bins, groups=3)
    assert float(loss) > .5  # Both means are zero; a mean-only objective would vanish.
    loss.backward()
    assert torch.isfinite(c.grad).all()


def test_signed_equivalent_pair_has_zero_energy_and_both_conditionals_have_gradients():
    dictionary = torch.tensor([[1., -1.], [0., 0.]])
    bins = torch.tensor([-1., 0., 1.])
    a = torch.tensor([[0., 0.]], requires_grad=True)
    c = torch.tensor([[[-80., -80., 0.], [0., -80., -80.]]], requires_grad=True)
    target = torch.tensor([[0., 0., 1.]])
    loss = compound_energy_distance(a, c, torch.tensor([[0, 1]]), torch.tensor([0]),
                                    target, dictionary, bins, groups=3)
    assert abs(float(loss)) < 2e-6
    wrong = c.detach().clone(); wrong[:, 1] = torch.tensor([-1., 1., 0.])
    wrong.requires_grad_(True)
    loss = compound_energy_distance(a, wrong, torch.tensor([[0, 1]]), torch.tensor([0]),
                                    target, dictionary, bins, groups=3)
    loss.backward()
    assert a.grad.norm() > 0 and wrong.grad.norm() > 0
    assert torch.isfinite(a.grad).all() and torch.isfinite(wrong.grad).all()


def test_candidate_conditionals_match_counterfactual_dense_forward_and_keep_native_predictions():
    model, aux, packed, cond, _ = setup()
    reference = copy.deepcopy(model).eval()
    expected = reference(packed, aux, cond)
    model.train()
    model.compound_geometry_config = dict(top_k=4, sites_per_image=2)
    model.compound_geometry_cursor = 0
    rng = torch.get_rng_state().clone()
    output = model(packed, aux, cond)
    assert torch.equal(rng, torch.get_rng_state())
    for key in ('atom_logits', 'coeff_logits'):
        torch.testing.assert_close(output[key], expected[key], rtol=0, atol=0)
    rows = output['geometry_selected_rows']
    candidates = output['geometry_candidate_atoms'].reshape(2, 4, 5)
    actual = output['geometry_coefficient_logits'].reshape(2, 4, 5, 5)
    for row in range(2):
        for depth in range(4):
            for candidate in range(5):
                changed = packed.clone().reshape(1, 4, 4)
                changed[0, rows[row], depth] = (candidates[row, depth, candidate] * 5 +
                                               changed[0, rows[row], depth] % 5)
                predicted = reference(changed.reshape_as(packed), aux, cond)['coeff_logits']
                torch.testing.assert_close(actual[row, depth, candidate],
                    predicted.reshape(1, 4, 4, 5)[0, rows[row], depth], rtol=2e-5, atol=2e-6)
    target_atoms = output['geometry_teacher_atoms']
    matching = output['geometry_candidate_atoms'] == target_atoms[:, None]
    weights = output['geometry_atom_logits'].softmax(-1)
    assert (weights.masked_fill(~matching, 0).sum(-1) > 0).all()
    assert ((weights > 0) & matching).sum(-1).eq(1).all()
    loss = compound_energy_distance(output['geometry_atom_logits'], output['geometry_coefficient_logits'],
        output['geometry_candidate_atoms'], target_atoms,
        torch.randn(8, 5).softmax(-1), aux.dictionary, aux.coeff_bins, groups=5)
    loss.backward()
    for name in ('history_gates', 'classifier.linear.weight', 'selected_atom_projection.weight',
                 'full_history.blocks.0.attn.query.weight', 'head_transformer.blocks.0.attn.query.weight'):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0, name


def test_candidate_replay_preserves_native_dropout_predictions_and_rng():
    model, aux, packed, cond, _ = setup()
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = .1
    reference = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    expected = reference(packed, aux, cond)
    expected_rng = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    model.compound_geometry_config = dict(top_k=4, sites_per_image=2)
    model.compound_geometry_cursor = 123
    actual = model(packed, aux, cond)
    for key in ('atom_logits', 'coeff_logits'):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())


def test_loss_tracking_includes_declared_geometry_and_rejects_missing_components():
    from src.training.training_loss_tracker import TrainingLossTracker
    tracker = TrainingLossTracker(100)
    result = tracker.update(6.51, 6.5, .1, .05, 2048, 101, additional_loss=.005)
    assert result['train/loss'] == 6.51 and result['train/cross_entropy'] == 6.5
    restored = TrainingLossTracker(101, tracker.state_dict())
    with pytest.raises(ValueError, match='objective components'):
        restored.update(6.51, 6.5, .1, .05, 2048, 102)
    restored.update(6.51, 6.5, .1, .05, 2048, 102, additional_loss=.005)


def test_official_evaluator_keeps_frozen_signature_and_preserves_rng():
    from src.training.compound_geometry_trial_hook import seeded_official_evaluation
    model = torch.nn.Linear(1, 1)
    calls = []
    # This signature intentionally has no fid_seed: the frozen production
    # evaluator predates that keyword in the working checkout.
    def evaluator(model, aux, val_loader, num_samples, batch_size=64,
            num_condition_classes=1000, atom_temperature=1., atom_top_k=0, atom_top_p=.92,
            coeff_temperature=1., coeff_top_k=0, coeff_top_p=.92,
            causal_prefix_sampling='predicted', compute_inception_score=True,
            metric_backend='original-rqvae', fid_reference_stats=None):
        calls.append((num_samples, batch_size, metric_backend, compute_inception_score))
        return torch.rand(3)
    rng = torch.get_rng_state().clone()
    def evaluate(seed):
        return seeded_official_evaluation(evaluator, model, None, None, 50000, 64,
            seed=seed, process_rank=3, metric_backend='original-rqvae',
            compute_inception_score=True)
    first = evaluate(261001)
    torch.testing.assert_close(evaluate(261001), first, rtol=0, atol=0)
    assert not torch.equal(evaluate(261101), first)
    assert torch.equal(torch.get_rng_state(), rng)
    assert calls == [(50000, 64, 'original-rqvae', True)] * 3


def test_trial_snapshot_precedes_evaluation_and_patch_is_idempotent(tmp_path):
    from scripts.tools.run_compound_energy_trial import patch_trial_runtime
    runtime = tmp_path/'runtime'
    target = runtime/'src/training/rqtransformer.py'
    target.parent.mkdir(parents=True)
    target.write_text('        if run_fid:\n'
        '            with optimizer_state_offloaded_for_generation(model, optimizer, device):\n'
        '                evaluate_generation_metrics(model)\n')
    patch_trial_runtime(runtime)
    first = target.read_text()
    patch_trial_runtime(runtime)
    assert first == target.read_text()
    assert first.index('save_before_official_evaluation(') < first.index('optimizer_state_offloaded')
