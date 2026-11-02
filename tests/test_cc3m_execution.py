import copy
import pytest
import torch
from unittest.mock import patch
from src.training.cc3m_joint import configure_execution, execution_options, attention_options
from src.training.rq_sdpa_backend import sdpa_attention
from tests.test_rq_sdpa_backend import attention


def test_runtime_overrides_preserve_training_config_and_can_be_reset():
    cfg=dict(encoder_batch=64,target_chunk=1024,execution=dict(attention_batch_limit=2048))
    before=copy.deepcopy(cfg)
    try:
        configure_execution(dict(short_sequence_math=True,target_chunk=4096,ce_chunk_size=4096))
        assert attention_options(cfg)==dict(batch_limit=2048,short_sequence_math=True)
        assert execution_options(cfg)['target_chunk']==4096
        assert cfg==before
        configure_execution({})
        assert execution_options(cfg)['target_chunk']==1024
    finally:configure_execution({})


@pytest.mark.parametrize('overrides',[dict(lr=.1),dict(short_sequence_math=1),dict(encoder_batch=True),dict(ce_chunk_size=0),dict(target_chunk=-1)])
def test_runtime_override_rejects_changed_recipe_and_invalid_controls(overrides):
    with pytest.raises(ValueError):configure_execution(overrides)


@pytest.mark.parametrize('length',[5,8,9])
def test_short_math_matches_native_outputs_and_all_gradients(length):
    torch.manual_seed(923);native=attention().train();fast=copy.deepcopy(native)
    x=torch.randn(3,length,12,requires_grad=True);y=x.detach().clone().requires_grad_()
    expected=native(x);expected.square().mean().backward()
    with sdpa_attention(fast,batch_limit=1,short_sequence_math=True):
        actual=fast(y);actual.square().mean().backward()
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(y.grad,x.grad,rtol=1e-5,atol=1e-6)
    for a,b in zip(native.parameters(),fast.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-5,atol=1e-6)
    assert '_sdpa_short_sequence_math' not in fast.__dict__


def test_short_math_is_training_only_and_restores_nested_context():
    from src.training import rq_sdpa_backend as backend
    model=attention();x=torch.randn(2,5,12)
    with patch.object(backend,'sdpa_kernel',wraps=backend.sdpa_kernel) as selector:
        with sdpa_attention(model,short_sequence_math=True):
            model(x)
            model(x,caching=True)
            assert selector.call_count==0
            model.train();model(x)
            assert selector.call_count==1
            with sdpa_attention(model,short_sequence_math=False):
                model(x)
                assert selector.call_count==1
            model(x)
            assert selector.call_count==2
    assert '_sdpa_short_sequence_math' not in model.__dict__


def test_target_chunking_preserves_full_conditional_distribution():
    from tests.test_laser_joint_vocabulary import case
    q=case();x=torch.randn(3,2,2,7)
    expected,codes=q.get_soft_codes(x,[.2]*4,stochastic=False,chunk_size=3)
    actual,larger_codes=q.get_soft_codes(x,[.2]*4,stochastic=False,chunk_size=8192)
    torch.testing.assert_close(actual,expected)
    assert torch.equal(larger_codes,codes)


def test_large_ce_chunks_preserve_bfloat16_loss_and_gradient():
    from src.scaled_atom_training import soft_cross_entropy
    torch.manual_seed(35)
    x=torch.randn(3,5,11,dtype=torch.bfloat16,requires_grad=True)
    reference=x.detach().clone().requires_grad_();targets=torch.rand_like(x,dtype=torch.float32).softmax(-1)*.9999
    actual=soft_cross_entropy(x,targets,chunk_size=8192)
    expected=-(reference.float().log_softmax(-1)*targets).sum(-1).mean()
    actual.backward();expected.backward()
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(x.grad,reference.grad,atol=3e-5,rtol=1e-2)
