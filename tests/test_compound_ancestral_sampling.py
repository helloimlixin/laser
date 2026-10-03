"""The sampling adapter must preserve the trained native compound factorization."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf

from src.compound_ancestral_sampling import categorical_probabilities, sample_dc_ancestral
from src.models.rqtransformer.attentions import AttentionStack
from src.models.rqtransformer.configs import RQTransformerConfig
from src.models.rqtransformer.transformers import RQTransformer, sample_from_logits


def compound_class():
    # The training CLI also imports an unrelated legacy third-party dataclass
    # package incompatible with Python3.11. Compile the actual maintained class
    # unchanged rather than importing that CLI or replacing model behavior.
    source = Path(__file__).resolve().parents[1] / 'src/training/rqtransformer.py'
    tree = ast.parse(source.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'CompoundLaserRQTransformer')
    namespace = dict(torch=torch, nn=nn, RQTransformer=RQTransformer,
                     AttentionStack=AttentionStack, sample_from_logits=sample_from_logits)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['CompoundLaserRQTransformer']


def setup():
    torch.manual_seed(63)
    config = RQTransformerConfig.create(OmegaConf.create(dict(
        type='rq-transformer', block_size=[2, 2, 3], embed_dim=12, input_embed_dim=4,
        shared_tok_emb=True, shared_cls_emb=True, input_emb_vqvae=True,
        head_emb_vqvae=True, cumsum_depth_ctx=True, vocab_size=7,
        vocab_size_cond=1, block_size_cond=1,
        body=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)),
        head=dict(n_layer=1, block=dict(n_head=3, resid_pdrop=0.)))))
    model = compound_class()(config, num_atoms=7, coeff_vocab_size=5,
        micro_transformer_layers=1, depth_specific_coeff_heads=True,
        pair_autoregressive=True).eval()
    aux = SimpleNamespace(dictionary=torch.randn(4, 7),
        coeff_bins=torch.tensor([-2., -1., 0., 1., 2.]), coeff_scales=torch.tensor([2., 5., 9.]))
    aux.compound_embeddings = lambda atoms, coefficients: (
        aux.dictionary.T[atoms.long()] *
        (aux.coeff_bins[coefficients.long()] * aux.coeff_scales)[..., None])
    return model, aux


def assert_caches_clear(model):
    assert model._cache['spatial_ctx_hw'] is None
    for block in [*model.body_transformer.blocks, *model.head_transformer.blocks]:
        assert block._cache['past_kv'] is None


@pytest.mark.parametrize('settings', [
    {},
    dict(atom_top_k=7, coeff_top_k=5),
    dict(atom_top_k=0, coeff_top_k=0),
    dict(atom_top_k=3, atom_top_p=1., coeff_top_p=.85, coeff_temperature=.9),
    dict(atom_top_k=4, coeff_top_k=3, atom_top_p=.8, coeff_top_p=1., atom_temperature=.7),
])
def test_actual_model_samples_and_rng_match_native_sampler(settings):
    model, aux = setup()
    native = dict(atom_temperature=1., coeff_temperature=1., atom_top_k=model.num_atoms,
                  coeff_top_k=model.coeff_vocab_size, atom_top_p=None, coeff_top_p=None, amp=False)
    native.update(settings)
    native['atom_top_k'] = native['atom_top_k'] or model.num_atoms
    native['coeff_top_k'] = native['coeff_top_k'] or model.coeff_vocab_size
    torch.manual_seed(931)
    expected = model.sample_compound(8, aux, **native)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(931)
    actual = sample_dc_ancestral(model, 8, aux, amp=False, **settings)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert_caches_clear(model)
    for depth in range(1, 3):
        assert not (actual[0][..., depth, None] == actual[0][..., :depth]).any()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('top_p', [None, .85, 1.])
def test_probabilities_match_native_nan_filter_and_threshold_arithmetic(dtype, top_p):
    logits = torch.tensor([[float('nan'), -1., 2., -float('inf'), .2],
                           [float('inf'), -.1, .1, 0., -float('inf')],
                           [1., 1., 1., -25., -30.]], dtype=dtype)
    for top_k in [2, logits.shape[-1]]:
        captured = []
        def capture(probabilities, num_samples):
            captured.append(probabilities.clone())
            return probabilities.argmax(-1, keepdim=True)
        with patch('torch.multinomial', side_effect=capture):
            sample_from_logits(logits, temperature=.9, top_k=top_k, top_p=top_p)
        actual = categorical_probabilities(logits, temperature=.9, top_k=top_k, top_p=top_p)
        torch.testing.assert_close(actual, captured[0], rtol=0, atol=0, equal_nan=True)


def test_full_vocabulary_skips_redundant_topk_but_p_one_retains_native_rounding():
    logits = torch.tensor([[0., -25., -30.]])
    with patch('torch.topk', side_effect=AssertionError('redundant full-vocabulary topk')):
        full = categorical_probabilities(logits, top_k=3, top_p=None)
        larger = categorical_probabilities(logits, top_k=100, top_p=None)
    assert torch.equal(full, larger)
    assert bool((full > 0).all())
    p_one = categorical_probabilities(logits, top_p=1.)
    assert p_one[0, 1:].count_nonzero() == 0


def test_explicit_generator_is_reproducible_and_does_not_advance_global_rng():
    model, aux = setup()
    before = torch.get_rng_state()
    first = sample_dc_ancestral(model, 4, aux, amp=False, generator=torch.Generator().manual_seed(11))
    second = sample_dc_ancestral(model, 4, aux, amp=False, generator=torch.Generator().manual_seed(11))
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert torch.equal(before, torch.get_rng_state())


def test_convenience_method_changes_no_weights_auxiliaries_modes_or_native_defaults():
    model, aux = setup()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    frozen = [value.clone() for value in (aux.dictionary, aux.coeff_bins, aux.coeff_scales)]
    modes = [module.training for module in model.modules()]
    model.sample_dc_ancestral(3, aux, amp=False)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(torch.equal(a, b) for a, b in zip(frozen, (aux.dictionary, aux.coeff_bins, aux.coeff_scales)))
    assert modes == [module.training for module in model.modules()]
    import inspect
    defaults = inspect.signature(model.sample_compound).parameters
    assert defaults['atom_top_p'].default == defaults['coeff_top_p'].default == .92
    assert defaults['atom_top_k'].default == 16384
    assert_caches_clear(model)


class RecordingCompound(nn.Module):
    """A controlled categorical model that exposes each complete-pair handoff."""
    pair_autoregressive = True
    causal_prefix_state = False
    num_atoms = 7
    coeff_vocab_size = 5
    block_size = (1, 2, 3)

    def __init__(self, fail=False):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls, self.selected, self.physical = [], [], []
        self.reset_count = 0
        self.fail = fail

    def init_cache(self):
        self.cache = None
        self.reset_count += 1

    def cached_head_output(self, packed, aux, cond, sample_loc, amp=True):
        assert not torch.is_grad_enabled()
        self.cache = sample_loc
        self.calls.append((sample_loc, packed.clone(), amp))
        atoms, coefficients = packed // 5, packed % 5
        self.physical.append(aux.compound_embeddings(atoms, coefficients).clone())
        return torch.zeros(packed.shape[0], 4)

    def classifier(self, hidden):
        return torch.arange(7., 0., -1.)[None].expand(hidden.shape[0], -1)

    def refine_coefficient_hidden(self, hidden, atom_vector):
        self.selected.append(atom_vector.clone())
        return atom_vector

    def classify_coefficients(self, refined, depth_index=None):
        if self.fail:
            raise RuntimeError('injected coefficient failure')
        selected_atom = refined[:, 0].long()
        coefficient = (selected_atom + 2 * depth_index) % 5
        logits = torch.full((len(refined), 5), -100.)
        return logits.scatter_(1, coefficient[:, None], 100.)


def test_selected_atom_conditions_value_and_complete_signed_scaled_pairs_are_committed_in_order():
    model = RecordingCompound().eval()
    aux = SimpleNamespace(dictionary=torch.stack([torch.arange(7.)] * 4),
        coeff_bins=torch.tensor([-2., -1., 0., 1., 2.]), coeff_scales=torch.tensor([2., 5., 9.]))
    aux.compound_embeddings = lambda a, c: aux.dictionary.T[a] * (aux.coeff_bins[c] * aux.coeff_scales)[..., None]
    atoms, coefficients = sample_dc_ancestral(model, 2, aux, amp=False, atom_top_k=1, coeff_top_k=1)
    expected_atoms = torch.tensor([0, 1, 2, 0, 1, 2]).reshape(1, 1, 2, 3).expand(2, -1, -1, -1)
    expected_coefficients = torch.tensor([0, 3, 1, 0, 3, 1]).reshape_as(expected_atoms[:1]).expand_as(expected_atoms)
    assert torch.equal(atoms, expected_atoms) and torch.equal(coefficients, expected_coefficients)
    committed = atoms * 5 + coefficients
    expected_physical = aux.compound_embeddings(atoms, coefficients).reshape(2, 6, 4)
    assert expected_physical[:, 1].gt(0).all() and expected_physical[:, 2].lt(0).all()
    for event, ((location, packed, amp), selected, physical) in enumerate(zip(model.calls, model.selected, model.physical)):
        assert location == (0, event // 3, event % 3) and amp is False
        assert torch.equal(packed.reshape(2, 6)[:, :event], committed.reshape(2, 6)[:, :event])
        assert torch.equal(packed.reshape(2, 6)[:, event:], torch.full((2, 6 - event), 2))
        assert torch.equal(selected, aux.dictionary.T[atoms.reshape(2, 6)[:, event]])
        assert torch.equal(physical.reshape(2, 6, 4)[:, :event], expected_physical[:, :event])
    assert model.reset_count == 2 and model.cache is None


def test_sampler_clears_real_native_caches_when_conditional_head_fails():
    model, aux = setup()
    modes = [module.training for module in model.modules()]
    with patch.object(model, 'refine_coefficient_hidden', side_effect=RuntimeError('injected failure')):
        with pytest.raises(RuntimeError, match='injected failure'):
            sample_dc_ancestral(model, 2, aux, amp=False)
    assert_caches_clear(model)
    assert modes == [module.training for module in model.modules()]


@pytest.mark.parametrize('settings', [
    dict(atom_temperature=0), dict(coeff_temperature=float('nan')),
    dict(atom_temperature=True), dict(atom_top_k=-1), dict(coeff_top_k=1.2),
    dict(atom_top_p=0), dict(coeff_top_p=1.1), dict(coeff_top_p=True),
    dict(amp='yes'), dict(generator=12),
])
def test_invalid_sampling_arguments_are_rejected(settings):
    model, aux = setup()
    with pytest.raises(ValueError):
        sample_dc_ancestral(model, 2, aux, **settings)


def test_training_and_other_compound_factorizations_are_rejected_without_flag_changes():
    model, aux = setup()
    for field, value in [('pair_autoregressive', False), ('causal_prefix_state', True)]:
        original = getattr(model, field)
        setattr(model, field, value)
        with pytest.raises(ValueError):
            sample_dc_ancestral(model, 2, aux, amp=False)
        setattr(model, field, original)
    model.coeff_micro_transformer.train()
    modes = [m.training for m in model.modules()]
    with pytest.raises(ValueError, match='eval'):
        sample_dc_ancestral(model, 2, aux, amp=False)
    assert modes == [m.training for m in model.modules()]
