from dataclasses import asdict, replace

import torch
import torch.nn.functional as F

from src.models.compound_gpt import CompoundGPT
from src.training.compound_gpt_upgrade import upgrade_checkpoint
from tests.test_compound_gpt import fixture


def strong_case(nonzero=False):
    old, aux, packed, labels = fixture()
    config = replace(old.config, dropout=.1, attention_dropout=0., embedding_dropout=0.,
        ffn_intermediate_dropout=0., coefficient_conditioner_layers=2,
        coefficient_conditioner_width=16, coefficient_conditioner_heads=4)
    model = CompoundGPT(config).eval()
    missing, unexpected = model.load_state_dict(old.state_dict(), strict=False)
    assert missing and all(name.startswith('coefficient_conditioner.') for name in missing)
    assert not unexpected
    if nonzero:
        torch.nn.init.normal_(model.coefficient_conditioner.output.weight, std=.1)
    return old, model, aux, packed, labels


def test_inserted_conditioner_preserves_both_predictions_exactly():
    old, model, aux, packed, labels = strong_case()
    expected = old(packed, model_aux=aux, cond=labels)
    result = model(packed, model_aux=aux, cond=labels)
    for key in ('atom_logits', 'coeff_logits'):
        torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)


def test_new_conditioner_cannot_see_current_coefficient_or_future_atoms():
    _, model, aux, packed, labels = strong_case(nonzero=True)
    baseline = model(packed, model_aux=aux, cond=labels)
    changed = packed.clone().flatten(1)
    changed[:, 3] = (changed[:, 3] // 8) * 8 + (changed[:, 3] % 8 + 2) % 8
    changed[:, 4:] = 18 * 8 + (changed[:, 4:] % 8)
    result = model(changed.reshape_as(packed), model_aux=aux, cond=labels)
    for key in ('atom_logits', 'coeff_logits'):
        torch.testing.assert_close(result[key].flatten(1, 3)[:, :4],
            baseline[key].flatten(1, 3)[:, :4], rtol=0, atol=0)


@torch.no_grad()
def test_nonzero_conditioner_cached_logits_match_full_sequence():
    _, model, aux, packed, labels = strong_case(nonzero=True)
    full = model(packed, model_aux=aux, cond=labels)
    atoms, _ = model.unpack(packed)
    cache = coefficient_cache = previous = None
    for position in range(model.events):
        h, cache, history = model.decode_step(previous, aux, labels, position, cache, return_history=True)
        logits, coefficient_cache = model.coefficient_logits_step(
            h, atoms.flatten(1)[:, position], aux, history, position, coefficient_cache)
        torch.testing.assert_close(logits, full['coeff_logits'].flatten(1, 3)[:, position], atol=5e-7, rtol=1e-5)
        assert all(k.shape[2] == position + 1 for k, v in coefficient_cache)
        previous = packed.flatten(1)[:, position]


def test_rq_dropout_policy_keeps_residual_dropout():
    _, model, _, _, _ = strong_case()
    assert model.embedding_dropout.p == 0
    for block in [*model.blocks, *model.coefficient_conditioner.blocks]:
        assert block.attn.attn_drop_p == 0
        assert block.attn.resid_dropout.p == .1
        assert block.ffn[2].p == 0
        assert block.ffn[-1].p == .1


def test_conditioner_starts_learning_then_reaches_internal_blocks():
    _, model, aux, packed, labels = strong_case()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    atoms, ids = model.unpack(packed)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(packed, model_aux=aux, cond=labels)
        loss = F.cross_entropy(result['coeff_logits'].flatten(0, 3), ids.flatten())
        loss.backward()
        assert model.coefficient_conditioner.output.weight.grad.norm() > 0
        if step:
            assert model.coefficient_conditioner.blocks[0].attn.qkv.weight.grad.norm() > 0
            assert model.physical_projection.weight.grad.norm() > 0
        optimizer.step()


def test_full_state_upgrade_preserves_old_moments_and_ages_new_parameters_separately():
    old, aux, packed, labels = fixture()
    optimizer = torch.optim.AdamW(old.parameters(), lr=.001, betas=(.9, .95))
    atoms, ids = old.unpack(packed)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = old(packed, model_aux=aux, cond=labels)
        loss = sum(F.cross_entropy(result[key].flatten(0, 3), target.flatten())
            for key, target in [('atom_logits', atoms), ('coeff_logits', ids)])
        loss.backward(); optimizer.step()
    original = dict(state_dict=old.state_dict(), optimizer=optimizer.state_dict(), global_step=3,
        epoch=1, batch_idx=0, config=dict(backbone_config=asdict(old.config)),
        scheduler=dict(last_epoch=3, T_max=100), rng_state_by_rank=[dict(torch_cpu=torch.get_rng_state())])
    before = torch.get_rng_state().clone()
    upgraded, receipt = upgrade_checkpoint(original, width=16, heads=4)
    assert torch.equal(before, torch.get_rng_state())
    for key, value in original['state_dict'].items():
        assert torch.equal(value, upgraded['state_dict'][key])
    for key, state in original['optimizer']['state'].items():
        for field, value in state.items():
            assert torch.equal(value, upgraded['optimizer']['state'][key][field])
    assert upgraded['scheduler'] == original['scheduler']
    assert upgraded['rng_state_by_rank'] is original['rng_state_by_rank']
    old_count = len(list(old.parameters()))
    for key, state in upgraded['optimizer']['state'].items():
        assert int(state['step']) == (3 if key < old_count else 0)
    model = CompoundGPT(type(old.config)(**upgraded['config']['backbone_config'])).eval()
    model.load_state_dict(upgraded['state_dict'], strict=True)
    for key, value in old(packed, model_aux=aux, cond=labels).items():
        torch.testing.assert_close(model(packed, model_aux=aux, cond=labels)[key], value, rtol=0, atol=0)
    resumed = torch.optim.AdamW(model.parameters(), lr=.001)
    resumed.load_state_dict(upgraded['optimizer'])
    resumed.zero_grad(set_to_none=True)
    out = model(packed, model_aux=aux, cond=labels)
    (F.cross_entropy(out['atom_logits'].flatten(0, 3), atoms.flatten())
     + F.cross_entropy(out['coeff_logits'].flatten(0, 3), ids.flatten())).backward()
    resumed.step()
    for name, parameter in model.named_parameters():
        assert int(resumed.state[parameter]['step']) == (1 if name.startswith('coefficient_conditioner.') else 4)


def test_upgraded_parameter_budget_stays_approximately_1_4b():
    from src.models.compound_gpt import CompoundGPTConfig
    with torch.device('meta'):
        model = CompoundGPT(CompoundGPTConfig(attention_dropout=0., embedding_dropout=0.,
            ffn_intermediate_dropout=0., coefficient_conditioner_layers=2))
    count = sum(p.numel() for p in model.parameters())
    assert 1.39e9 < count < 1.41e9, count
