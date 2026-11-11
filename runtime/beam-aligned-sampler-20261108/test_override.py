"""Contracts for the generation-only, fixed-weight calibration override."""
import copy
import importlib.util
from pathlib import Path
import pytest
import hashlib
import json
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('calibrated_sampling',Path(__file__).with_name('calibrated_sampling.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def evidence():
    sampling=dict(temperature=.95,top_k=[1024,1024,512,256],top_p=[.95,.95,.9,.85])
    base=dict(temperature=1.,top_k=1024,top_p=.95)
    common=dict(model_state_sha256='model',checkpoint_sha256='checkpoint',seed=42,generated_images=50000,
        weights_unmodified=True,training_updates_added=0,reference_images=1281167,reference_split='train',
        reference_sha256='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260',
        rejection_sampling=False,source_step=5000,sampling_batch_per_rank=256)
    rows=[dict(common,policy='baseline',sampling=base,fid=40.,inception_score=25.),
          dict(common,policy='winner',sampling=sampling,fid=38.,inception_score=26.)]
    return dict(passed=True,improved=True,weights_unmodified=True,training_updates_added=0,generated_images=50000,
        reference_images=1281167,reference_split='train',fid=38.,baseline_fid=40.,minimum_fid_improvement=.5,
        checkpoint_sha256='checkpoint',source_step=5000,selected_policy='winner',sampling=sampling,
        source_sampling=base,baseline_inception_score=25.,confirmation=rows)


def test_depth_settings_are_preserved_and_evidence_is_immutable():
    value=evidence();before=copy.deepcopy(value)
    assert module.validate(value)==value['sampling']
    assert value==before


@pytest.mark.parametrize('field,replacement',[
    ('training_updates_added',1),('generated_images',2000),('reference_split','val'),
    ('reference_images',50000),('weights_unmodified',False),('improved',False)])
def test_insufficient_or_incomparable_evidence_cannot_change_sampler(field,replacement):
    value=evidence();value[field]=replacement
    with pytest.raises(AssertionError):module.validate(value)


@pytest.mark.parametrize('field,replacement',[
    ('checkpoint_sha256','other'),('model_state_sha256','other'),('seed',43),
    ('source_step',5001),('generated_images',2000),('rejection_sampling',True)])
def test_each_confirmation_requires_the_same_weights_seed_and_protocol(field,replacement):
    value=evidence();value['confirmation'][1][field]=replacement
    with pytest.raises(AssertionError):module.validate(value)


def test_fid_gain_cannot_be_claimed_by_falsifying_baseline():
    value=evidence();value['baseline_fid']=50.
    with pytest.raises(AssertionError):module.validate(value)


def test_large_inception_score_regression_is_not_applied_automatically():
    value=evidence();value['confirmation'][1]['inception_score']=20.
    with pytest.raises(AssertionError):module.validate(value)


def replicated():
    value=evidence();value['minimum_fid_improvement']=0.
    repeat=copy.deepcopy(value['confirmation'])
    for row in repeat:row['seed']=43
    repeat[1]['fid']=39.
    value['repeat_confirmation']=repeat
    return value


def test_two_independent_full_confirmations_allow_a_replicated_gain():
    assert module.validate(replicated())==evidence()['sampling']


@pytest.mark.parametrize('field,replacement',[
    ('seed',42),('fid',41.),('inception_score',20.),('model_state_sha256','other')])
def test_unreplicated_or_incomparable_gains_are_rejected(field,replacement):
    value=replicated()
    if field=='seed':
        for row in value['repeat_confirmation']:row['seed']=replacement
    else:value['repeat_confirmation'][1][field]=replacement
    with pytest.raises(AssertionError):module.validate(value)


def test_loading_the_override_changes_generation_only_and_preserves_training_yaml(tmp_path):
    value=replicated();inputs=tmp_path/'inputs';inputs.mkdir()
    original=b'decoder: native_joint_integer\ntargets: {sequence_temperature: 2.0}\n'
    (inputs/'recipe.yaml').write_bytes(original)
    value.update(source_run='test',codebook_sha256='book',training_recipe_sha256=hashlib.sha256(original).hexdigest())
    (inputs/'generation-sampling-override.json').write_text(json.dumps(value))
    profile=SimpleNamespace(base=tmp_path,run_id='test',codebook_sha256='book',generation_batch=64)
    assert module.settings(profile)==value['sampling']
    assert profile.generation_batch==256
    assert (inputs/'recipe.yaml').read_bytes()==original
    module.verify_resume(profile,{'global_step':5000})
    with pytest.raises(AssertionError):module.verify_resume(profile,{'global_step':4999})
