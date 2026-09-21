"""Reject recovery that would silently turn the control into another experiment."""
import copy
import pytest

from src.training.church_released_control import validate_control_resume


def valid_payload():
    config = {'experiment': {'accumulation_steps': 8, 'batch_size': 128, 'total_batch_size': 2048}}
    cache = {'checkpoint_sha256': 'released', 'cache_sha256': 'fixed-pixels'}
    protocol = {'generated_samples': 50000, 'reference_sha256': 'full-church'}
    payload = dict(state_dict={}, optimizer={}, scheduler={}, scaler={},
        rng_states=[dict(torch=0,cuda=0,numpy=0,python=0) for _ in range(2)],
        epoch=0,batch_in_epoch=16,step=2,attempts=2,skipped_amp_updates=0,
        tokenizer=copy.deepcopy(cache),config=copy.deepcopy(config),fid_protocol=copy.deepcopy(protocol),
        run_id='control',initial_weights_sha256='fresh',ranked_fid_checkpoints=[],pending_evaluation_epoch=None)
    return payload,dict(config=config,cache=cache,protocol=protocol,run_id='control',loader_batches=494)


def test_mid_epoch_and_pending_evaluation_resume():
    payload,kwargs = valid_payload()
    assert validate_control_resume(payload,**kwargs) == (0,16)
    payload.update(epoch=1,batch_in_epoch=0,pending_evaluation_epoch=1)
    assert validate_control_resume(payload,**kwargs) == (1,0)


@pytest.mark.parametrize('field,value',[
    ('tokenizer',{'checkpoint_sha256':'fine-tuned-instead'}),
    ('fid_protocol',{'reference_sha256':'other-population'}),
    ('config',{'experiment':{'batch_size':256}}),
    ('run_id','different-run'),('batch_in_epoch',15),('attempts',3),
    ('pending_evaluation_epoch',1),('rng_states',[dict(torch=0)]),
])
def test_changed_experiment_or_incomplete_cursor_is_rejected(field,value):
    payload,kwargs = valid_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        validate_control_resume(payload,**kwargs)


def test_weights_only_checkpoint_cannot_resume():
    _,kwargs = valid_payload()
    with pytest.raises(ValueError,match='Incomplete'):
        validate_control_resume({'state_dict':{}},**kwargs)
