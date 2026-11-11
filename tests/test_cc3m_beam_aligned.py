from types import SimpleNamespace

import pytest
import torch

from src.models.laser_joint_tokenizer import JointAtomCoefficientRQ
from src.training import cc3m_joint
from src.training.beam_aligned_teacher import aligned_targets, verify_conditionals


def test_shared_prefix_conditionals_recover_the_sequence_posterior():
    assert verify_conditionals()['passed']


def test_cc3m_dispatch_uses_beam_teacher_and_same_sampled_prefixes(tmp_path):
    torch.manual_seed(7)
    dictionary = torch.randn(3, 5)
    atom_ids = torch.arange(9).remainder(5).repeat(4, 1)
    coefficients = torch.tensor([0., -.8, -.2, .3, .9, -.5, .1, .6, 1.1]).repeat(4, 1)
    book = JointAtomCoefficientRQ(dictionary, atom_ids, coefficients, encoding_beam_width=16)
    aux = SimpleNamespace(physical_book=book)  # Deliberately no native quantizer.
    latent = torch.randn(2, 1, 1, 3)
    cfg = dict(run_id='beam-test', tokenizer_backend='native_beam_pair', image_teacher='beam_aligned',
        local_checkpoints=str(tmp_path/'checkpoints'), execution=dict(exact_beam_cost=False),
        beam_teacher=dict(beam_width=16, sequence_temperature=2., max_relative_candidate_error=.01))
    cc3m_joint._TEACHER_AUDITED.clear()
    torch.manual_seed(41)
    expected_targets, expected_codes = aligned_targets(book, latent, 2.)
    torch.manual_seed(41)
    targets, codes = cc3m_joint.image_targets(aux, latent, cfg)
    torch.testing.assert_close(targets, expected_targets, rtol=0, atol=0)
    assert torch.equal(codes, expected_codes)
    assert not targets.requires_grad
    assert (tmp_path/'live/teacher-audit-rank0.json').exists()


def test_cc3m_rejects_an_unvalidated_beam_policy():
    cfg = dict(image_teacher='beam_aligned', beam_teacher=dict(beam_width=4, max_relative_candidate_error=.01))
    with pytest.raises(ValueError, match='beam16'):
        cc3m_joint.image_targets(None, torch.empty(1), cfg)
