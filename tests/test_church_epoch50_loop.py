import copy
import pytest
import torch

from test_compound_pair_autoregressive import tiny_aux, tiny_config
from scripts.train_official_rqtransformer_laser_stage2 import CompoundLaserRQTransformer
from src.church_epoch50_loop import GatedDepthLoop, continuation_lr


def pair():
    torch.manual_seed(601)
    cfg = tiny_config(4)
    cfg.block_size = [2, 2, 4]
    cfg.head.n_layer = 4
    original = CompoundLaserRQTransformer(cfg, 7, 5, pair_autoregressive=True,
        micro_transformer_layers=2, depth_specific_coeff_heads=True).eval()
    looped = copy.deepcopy(original)
    looped.head_transformer = GatedDepthLoop(list(looped.head_transformer.blocks))
    looped.eval()
    atoms = torch.tensor([[[[0, 2, 4, 6], [1, 3, 5, 6]], [[0, 1, 2, 3], [2, 3, 4, 5]]]])
    packed = atoms * 5 + torch.arange(16).reshape_as(atoms) % 5
    return original, looped, tiny_aux(4), atoms, packed


def test_zero_gates_preserve_all_original_weights_outputs_and_samples():
    original, looped, aux, _, packed = pair()
    assert set(looped.state_dict()) - set(original.state_dict()) == {'head_transformer.loop_gates'}
    for key, value in original.state_dict().items():
        assert torch.equal(value, looped.state_dict()[key])
    with torch.no_grad():
        a, b = [m(packed, model_aux=aux) for m in (original, looped)]
        for key in a:
            assert torch.equal(a[key], b[key])
        generated = []
        for m in (original, looped):
            torch.manual_seed(77)
            generated.append(m.sample_compound(2, aux, atom_top_k=7, atom_top_p=1., coeff_top_p=None, amp=False))
        assert all(torch.equal(a, b) for a, b in zip(*generated))


def test_closed_gate_learns_and_pretrained_gradients_are_preserved():
    original, looped, aux, _, packed = pair()
    original.train(); looped.train()
    for m in (original, looped):
        out = m(packed, model_aux=aux)
        out['coeff_logits'].square().mean().backward()
    assert torch.isfinite(looped.head_transformer.loop_gates.grad).all()
    assert looped.head_transformer.loop_gates.grad.abs().sum() > 0
    for name, parameter in original.named_parameters():
        other = dict(looped.named_parameters())[name]
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-7, rtol=2e-5)


def test_active_loops_full_grid_cache_matches_teacher_and_is_causal():
    _, m, aux, atoms, packed = pair()
    m.head_transformer.loop_gates.data.copy_(torch.tensor([.12, -.08]))
    with torch.no_grad():
        teacher = m(packed, model_aux=aux)
        changed = packed.clone(); changed[:, 0, 0, 0] += 1
        other = m(changed, model_aux=aux)
        for name in teacher:
            assert torch.equal(teacher[name][:, 0, 0, 0], other[name][:, 0, 0, 0])
            assert not torch.equal(teacher[name][:, 0, 0, 1], other[name][:, 0, 0, 1])
            assert not torch.equal(teacher[name][:, 0, 1, 0], other[name][:, 0, 1, 0])
        m.init_cache()
        for h in range(2):
            for w in range(2):
                for d in range(4):
                    hidden = m.cached_head_output(packed, aux, None, (h, w, d), amp=False)
                    logits = m.classifier(hidden)
                    if d: logits.scatter_(1, atoms[:, h, w, :d], -float('inf'))
                    coefficient = m.coefficient_logits(hidden, aux.dictionary.t()[atoms[:, h, w, d]], d)
                    torch.testing.assert_close(logits, teacher['atom_logits'][:, h, w, d], atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(coefficient, teacher['coeff_logits'][:, h, w, d], atol=2e-6, rtol=2e-5)
        caches = [c for row in m.head_transformer._pass_caches for c in row]
        assert len(caches) == len({id(c) for c in caches}) == 12
        assert all(c['past_kv'] is not None for c in caches)
        m.init_cache()
        assert all(c['past_kv'] is None for row in m.head_transformer._pass_caches for c in row)


def test_schedule_decays_and_does_not_restart():
    assert continuation_lr(8, 128) == 1e-5
    assert 1e-6 < continuation_lr(64, 128) < 1e-5
    assert continuation_lr(128, 128) == continuation_lr(200, 128) == 1e-6
    with pytest.raises(ValueError):
        continuation_lr(1, 4)
