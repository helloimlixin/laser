from unittest.mock import patch
import torch

from src.church_original_scratch import scratch_prior, initialization_audit, FullBatchEpochStream
from src.church_ffhq_recipe import early_decay_lr
from tests.test_church_epoch50_loop import pair


def test_factory_cannot_read_or_load_a_checkpoint():
    with patch('torch.load', side_effect=AssertionError('Checkpoint read forbidden')), \
         patch('torch.nn.Module.load_state_dict', side_effect=AssertionError('State loading forbidden')):
        with torch.device('meta'):
            control, looped = scratch_prior('control'), scratch_prior('looped')
    assert control.config.embed_dim == 1024
    assert len(control.body_transformer.blocks) == 24
    assert len(control.head_transformer.blocks) == len(looped.head_transformer.blocks) == 4
    assert sum(p.numel() for p in control.parameters()) == 404738048
    assert sum(p.numel() for p in looped.parameters()) == 404738050
    assert control.pair_autoregressive and looped.pair_autoregressive


def test_initialization_audit_ignores_only_added_gates():
    control, looped, _, _, _ = pair()
    a, b = initialization_audit(control, 601), initialization_audit(looped, 601)
    assert a == b
    assert a['kind'] == 'random' and a['stage2_checkpoint_loaded'] is None
    next(p for p in looped.classifier.parameters() if p.ndim == 2).data.add_(.1)
    assert initialization_audit(looped, 601)['common_tensor_sha256'] != a['common_tensor_sha256']


def test_original_full_batches_drop_remainder_and_resume_exactly():
    a, b = FullBatchEpochStream(11, 17), FullBatchEpochStream(11, 99)
    first, progress, end = a.next(4)
    assert len(first) == 4 and not end
    second, progress, end = a.next(4)
    assert len(second) == 4 and end and progress == 1
    assert a.position == a.size
    b.load_state_dict(a.state_dict())
    x, xp, xe = a.next(4)
    y, yp, ye = b.next(4)
    assert torch.equal(x, y) and xp == yp and xe == ye
    assert a.epoch == b.epoch == 1


def test_scratch_lr_has_training_scale_and_early_decay():
    args = (100, 5e-4, 5e-5, 1e-6, 1., 10.)
    assert early_decay_lr(1, *args) == 5e-4
    assert abs(early_decay_lr(10, *args) - 5e-5) < 1e-12
    assert early_decay_lr(100, *args) == 1e-6
