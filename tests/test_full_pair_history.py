"""Behavioral checks for the FFHQ compound recipe with four sparse depths."""

import torch

from src.training.rqtransformer import CompoundLaserRQTransformer
from tests.test_compound_pair_autoregressive import tiny_aux, tiny_config


def make_case():
    torch.manual_seed(71)
    config = tiny_config(depth=4)
    config.block_size = [2, 2, 4]
    config.vocab_size_cond = 1000
    model = CompoundLaserRQTransformer(
        config, 7, 5, micro_transformer_layers=2,
        depth_specific_coeff_heads=True, pair_autoregressive=True,
        mask_seen_atoms_training=False,
    ).eval()
    atoms = torch.arange(16).reshape(1, 2, 2, 4).remainder(7)
    coefficients = torch.arange(16).reshape_as(atoms).remainder(5)
    return model, tiny_aux(depth=4), atoms * 5 + coefficients, torch.tensor([999])


@torch.no_grad()
def test_every_earlier_atom_and_coefficient_affects_both_predictors():
    model, aux, tokens, labels = make_case()
    baseline = model(tokens, model_aux=aux, cond=labels)
    for changed_event in range(tokens.numel()):
        for component in ("atom", "coefficient"):
            changed = tokens.clone()
            atom, coefficient = divmod(int(changed.flatten()[changed_event]), 5)
            if component == "atom":
                atom = (atom + 1) % 7
            else:
                coefficient = (coefficient + 2) % 5
            changed.flatten()[changed_event] = atom * 5 + coefficient
            output = model(changed, model_aux=aux, cond=labels)
            for key in ("atom_logits", "coeff_logits"):
                original = baseline[key].flatten(0, -2)
                modified = output[key].flatten(0, -2)
                # An atom can affect its own coefficient, but no predictor
                # sees its own coefficient or any future event.
                visible_at_current = component == "atom" and key == "coeff_logits"
                unchanged_prefix = changed_event + (not visible_at_current)
                torch.testing.assert_close(
                    original[:unchanged_prefix], modified[:unchanged_prefix],
                    rtol=0, atol=0,
                )
                for target_event in range(unchanged_prefix, tokens.numel()):
                    assert not torch.equal(original[target_event], modified[target_event]), (
                        component, changed_event, key, target_event,
                    )


@torch.no_grad()
def test_cached_predictions_match_teacher_forcing_with_only_generated_history():
    model, aux, targets, labels = make_case()
    teacher = model(targets, model_aux=aux, cond=labels)
    # Future entries are placeholders, not teacher-forced future values.
    generated = torch.full_like(targets, 2)
    model.init_cache()
    try:
        for row in range(2):
            for column in range(2):
                for depth in range(4):
                    location = (row, column, depth)
                    hidden = model.cached_head_output(
                        generated, aux, labels, location, amp=False,
                    )
                    atom_logits = model.classifier(hidden)
                    atom = targets[:, row, column, depth].div(5, rounding_mode="floor")
                    coefficient_logits = model.coefficient_logits(
                        hidden, aux.dictionary.t()[atom], depth_index=depth,
                    )
                    for key, actual in (("atom_logits", atom_logits),
                                        ("coeff_logits", coefficient_logits)):
                        torch.testing.assert_close(
                            actual, teacher[key][:, row, column, depth],
                            rtol=2e-6, atol=3e-7,
                        )
                    # Complete this pair before predicting the next atom.
                    generated[:, row, column, depth] = targets[:, row, column, depth]
    finally:
        model.init_cache()


def test_coefficient_attention_uses_history_and_dictionary_vector():
    model, aux, _, _ = make_case()
    history = torch.randn(2, 12, requires_grad=True)
    dictionary_vectors = aux.dictionary.t()[torch.tensor([1, 3])].clone().requires_grad_()
    observed = []
    hook = model.coeff_micro_transformer.register_forward_pre_hook(
        lambda module, inputs: observed.append(inputs[0].detach().clone())
    )
    try:
        logits = model.coefficient_logits(history, dictionary_vectors, depth_index=3)
    finally:
        hook.remove()
    expected_inputs = torch.stack((history, model.coeff_atom_proj(dictionary_vectors)), dim=1)
    torch.testing.assert_close(observed[0], expected_inputs + model.coeff_micro_pos)
    logits.square().sum().backward()
    for gradient in (history.grad, dictionary_vectors.grad,
                     model.coeff_atom_proj.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.coeff_micro_transformer.parameters())
