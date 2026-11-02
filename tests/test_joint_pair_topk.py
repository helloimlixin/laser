import copy

import pytest
import torch

from src.joint_pair_topk import joint_pair_topk, sample_joint_pair_topk
from tests.test_offset_pair_prior import case


def test_joint_launch_recipe_reaches_the_cli_and_uses_one_complete_pair_policy():
    from omegaconf import OmegaConf
    from src.training.cli import load_config
    from src.training.options import options_to_argv
    from src.training.rqtransformer import build_parser, preview_sampling_settings
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root/'configs/stage2/imagenet-compound512-joint-pair-8a100-20261027.yaml')
    args = build_parser().parse_args(options_to_argv(OmegaConf.to_container(cfg.options)))
    settings = preview_sampling_settings(args)
    assert len(settings) == 1 and settings[0]['joint_pair_top_k'] == 131072
    assert settings[0]['joint_pair_temperature'] == .8
    assert not any(key.startswith(('atom_', 'coeff_')) for key in settings[0])
    legacy = load_config(root/'configs/stage2/imagenet-compound512-quality-repair-8a100-20261026.yaml')
    assert legacy.stage == 'stage2' and legacy.options.atom_temperature == .8


@pytest.mark.parametrize('chunk', [1, 2, 5, 20])
@pytest.mark.parametrize('k', [1, 3, 11, 35])
def test_certified_search_matches_full_atom_coefficient_enumeration(chunk, k):
    torch.manual_seed(31)
    a = torch.randn(3, 7)
    c = torch.randn(3, 7, 5)
    a[1, 0] = -torch.inf
    def callback(rows, ids):
        return c[rows[:, None], ids]
    scores, ids, stats = joint_pair_topk(a, callback, top_k=k, candidate_chunk=chunk)
    full = (a.log_softmax(-1)[..., None] + c.log_softmax(-1)).flatten(1)
    expected, positions = full.topk(k, -1)
    torch.testing.assert_close(scores, expected)
    finite = torch.isfinite(expected)
    assert torch.equal(ids[finite], positions[finite])
    torch.testing.assert_close(stats['retained_mass'], expected.logsumexp(-1).exp())
    assert stats['certified']


def test_top_pair_can_use_an_atom_outside_the_marginal_top_one():
    a = torch.tensor([[.55, .45]]).log()
    c = torch.tensor([[[.5, .5], [.99, .01]]]).log()
    scores, ids, stats = joint_pair_topk(a, lambda rows, atoms: c[rows[:, None], atoms],
                                        top_k=1, candidate_chunk=1)
    assert int(ids[0, 0]) == 2  # atom 1, coefficient 0 beats both atom-0 pairs
    torch.testing.assert_close(scores.exp(), torch.tensor([[.45 * .99]]))
    assert int(stats['evaluated_atoms'][0]) == 2


def test_candidate_memory_budget_does_not_change_the_global_ranking():
    torch.manual_seed(77)
    a, c = torch.randn(3, 41), torch.randn(3, 41, 17)
    def callback(rows, ids):
        assert ids.numel() <= 3
        return c[rows[:, None], ids]
    scores, ids, _ = joint_pair_topk(a, callback, top_k=31,
        candidate_chunk=1000, max_candidate_batch=3)
    expected, indices = (a.log_softmax(-1)[..., None] + c.log_softmax(-1)).flatten(1).topk(31, -1)
    torch.testing.assert_close(scores, expected)
    assert torch.equal(ids, indices)


def test_search_stops_only_when_unseen_joint_probabilities_are_bounded_out():
    a = torch.tensor([[.99, .009, .001]]).log()
    c = torch.tensor([[[.9, .1], [.5, .5], [.5, .5]]]).log()
    scores, ids, stats = joint_pair_topk(a, lambda rows, atoms: c[rows[:, None], atoms],
                                        top_k=1, candidate_chunk=1)
    assert int(ids[0, 0]) == 0 and int(stats['evaluated_atoms'][0]) == 1
    with pytest.raises(RuntimeError, match='not certified'):
        joint_pair_topk(torch.tensor([[.55, .45]]).log(),
            lambda rows, atoms: c[:1, :2][rows[:, None], atoms], top_k=3,
            candidate_chunk=1, max_candidate_atoms=1)


@pytest.mark.parametrize('temperature', [.5, 1., 1.5])
def test_temperature_and_nucleus_act_on_complete_pairs(monkeypatch, temperature):
    a = torch.tensor([[.55, .45]]).log()
    c = torch.tensor([[[.5, .5], [.99, .01]]]).log()
    captured = []
    def sample(p, *args, **kwargs):
        captured.append(p)
        return p.argmax(-1, keepdim=True)
    monkeypatch.setattr(torch, 'multinomial', sample)
    atom, coefficient, _ = sample_joint_pair_topk(a,
        lambda rows, atoms: c[rows[:, None], atoms], top_k=3, temperature=temperature, top_p=1.)
    expected = torch.tensor([[.4455, .275, .275]]).pow(1 / temperature)
    expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(captured[0], expected)
    assert int(atom) == 1 and int(coefficient) == 0
    sample_joint_pair_topk(a, lambda rows, atoms: c[rows[:, None], atoms], top_k=3,
                          temperature=temperature, top_p=.1)
    assert int(captured[-1].count_nonzero()) == 1


@torch.no_grad()
def test_candidate_logits_equal_independent_native_branches_without_cache_mutation():
    _, model, aux, target, cond = case()
    # Activate the normally zero-initialized global decoder so this test
    # exercises its attention history, not just the local conditioner.
    torch.nn.init.normal_(model.coefficient_history_decoder.output.weight, std=.08)
    model.init_cache()
    generated = target.clone()
    for h in range(2):
        for w in range(2):
            for depth in range(4):
                model.cached_forward(generated, aux, cond, amp=False, sample_loc=(h, w, 2*depth))
                history = model.coefficient_history_decoder
                cache = [block._kv for block in history.blocks]
                event = history._next_event
                candidates = torch.tensor([[0, 3, 6]])
                actual = model.joint_coefficient_scorer(aux)(torch.tensor([0]), candidates)
                assert history._next_event == event
                assert all(block._kv is old for block, old in zip(history.blocks, cache))
                for column, atom in enumerate(candidates[0]):
                    branch = copy.deepcopy(model)
                    branch_tokens = generated.clone()
                    branch_tokens[:, h, w, 2*depth] = atom
                    expected = branch.cached_forward(branch_tokens, aux, cond, amp=False,
                        sample_loc=(h, w, 2*depth+1))[:, 7:]
                    torch.testing.assert_close(actual[:, column], expected, atol=3.e-6, rtol=2.e-5)
                model.cached_forward(generated, aux, cond, amp=False, sample_loc=(h, w, 2*depth+1))


@torch.no_grad()
def test_joint_sampler_commits_distinct_atoms_offset_coefficients_and_preserves_weights():
    _, model, aux, _, _ = case()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    tokens, stats = model.sample_joint_sparse(2, aux, torch.tensor([1, 999]), top_k=9,
        temperature=.9, top_p=.95, candidate_chunk=2, amp=False,
        generator=torch.Generator().manual_seed(81), return_stats=True)
    again = model.sample_joint_sparse(2, aux, torch.tensor([1, 999]), top_k=9,
        temperature=.9, top_p=.95, candidate_chunk=5, amp=False,
        generator=torch.Generator().manual_seed(81))
    assert torch.equal(tokens, again)
    atoms, coefficients = model.fields(tokens)
    assert ((coefficients >= 0) & (coefficients < 5)).all()
    assert (atoms.sort(-1).values.diff(dim=-1) > 0).all()
    assert stats['certified_global_top_k'] and stats['pairs'] == 32
    assert model._offset_hidden is None and model.coefficient_history_decoder._next_event == 0
    assert all(block._kv is None for block in model.coefficient_history_decoder.blocks)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
