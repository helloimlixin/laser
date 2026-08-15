import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "train_foundationvision_var_laser_stage2.py"
SPEC = importlib.util.spec_from_file_location("foundationvision_var_laser", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_factorized_objective_is_joint_token_nll():
    torch.manual_seed(0)
    atom_logits = torch.randn(2, 5, 7)
    coeff_logits = torch.randn(2, 5, 11)
    atom_targets = torch.randint(0, 7, (2, 5))
    coeff_targets = torch.randint(0, 11, (2, 5))

    loss, atom_nll, coeff_nll = MODULE.exact_joint_objective(
        atom_logits, coeff_logits, atom_targets, coeff_targets
    )
    joint_log_probs = (
        atom_logits.log_softmax(-1).unsqueeze(-1)
        + coeff_logits.log_softmax(-1).unsqueeze(-2)
    )
    expected = -joint_log_probs[
        torch.arange(2)[:, None], torch.arange(5)[None, :],
        atom_targets, coeff_targets,
    ].mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(loss, (atom_nll + coeff_nll).mean())


def test_official_backbone_has_two_complete_8x8_stages(monkeypatch):
    monkeypatch.setattr(MODULE.fv_dist, "get_device", lambda: torch.device("cpu"))
    model = MODULE.FoundationVisionLaserVAR(
        input_dim=8,
        num_atoms=13,
        coeff_vocab_size=17,
        num_classes=1,
        depth=2,
        embed_dim=32,
        num_heads=2,
        patch_nums=(8, 8),
        flash_if_available=False,
        fused_if_available=False,
    )
    assert model.L == 128
    assert model.begin_ends == [(0, 64), (64, 128)]
    assert model.first_l == 64
    allowed = torch.isfinite(model.attn_bias_for_masking[0, 0])
    assert allowed[:64, :64].all()
    assert not allowed[:64, 64:].any()
    assert allowed[64:, :].all()


def test_official_forward_adapter_shapes(monkeypatch):
    monkeypatch.setattr(MODULE.fv_dist, "get_device", lambda: torch.device("cpu"))
    model = MODULE.FoundationVisionLaserVAR(
        input_dim=8,
        num_atoms=13,
        coeff_vocab_size=17,
        num_classes=1,
        depth=2,
        embed_dim=32,
        num_heads=2,
        patch_nums=(2, 2),
        flash_if_available=False,
        fused_if_available=False,
    )
    labels = torch.zeros(3, dtype=torch.long)
    previous = torch.randn(3, 4, 8)
    target_atom_vectors = torch.randn(3, 8, 8)
    output = model(labels, previous, target_atom_vectors)
    assert output["atom_logits"].shape == (3, 8, 13)
    assert output["coeff_logits"].shape == (3, 8, 17)


def test_compound_batch_is_level_major():
    class Aux:
        coeff_scales = torch.tensor([2.0, 3.0])
        coeff_bins = torch.tensor([-1.0, 0.0, 1.0])
        dictionary = torch.arange(32.0).reshape(4, 8)

        def compound_coeff_ids(self, coeffs, stochastic, hard):
            return coeffs.long(), None

        def compound_embeddings(self, atoms, coeff_ids):
            vectors = self.dictionary.t()[atoms]
            scales = self.coeff_scales.view(1, 1, 1, 2)
            return vectors * (self.coeff_bins[coeff_ids] * scales)[..., None]

    atoms = torch.tensor([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]])
    coeffs = torch.tensor([[[[0, 1], [2, 0]], [[1, 2], [0, 1]]]])
    previous, atom_targets, coeff_targets, atom_vectors = MODULE.compound_batch(
        Aux(), atoms, coeffs
    )
    assert atom_targets.tolist() == [[0, 2, 4, 6, 1, 3, 5, 7]]
    assert coeff_targets.tolist() == [[0, 2, 1, 0, 1, 0, 2, 1]]
    assert previous.shape == (1, 4, 4)
    torch.testing.assert_close(atom_vectors, Aux.dictionary.t()[atom_targets])
