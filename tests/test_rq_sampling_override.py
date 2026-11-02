import copy

import pytest

from src.training.rq_sampling_override import validated_sampling_recipe


def example():
    source=dict(atom_temperature=1.,atom_top_k=256,atom_top_p=.95,
        coefficient_temperature=1.,coefficient_top_k=256,coefficient_top_p=.95)
    target=dict(source,atom_top_k=0)
    recipe=dict(run_id='test-run',sampling=source,optimization=dict(warmup_epochs=0),
        tokenizer=dict(coefficient_bins=512))
    evidence=dict(passed=True,weights_unmodified=True,training_updates_added=0,
        generated_images=50000,reference_images=1281167,reference_split='train',
        source_run='test-run',source_sampling=source,sampling=target,fid=80.,baseline_fid=112.)
    return recipe,evidence


def test_sampler_change_preserves_training_recipe_and_input_records():
    recipe,evidence=example()
    original=copy.deepcopy(recipe)
    result=validated_sampling_recipe(recipe,recipe,evidence)
    assert result['sampling']['atom_top_k']==0
    assert recipe==original
    assert {k:v for k,v in result.items() if k!='sampling'}=={k:v for k,v in recipe.items() if k!='sampling'}
    # A later checkpoint already saved with the approved sampler is resumable.
    assert validated_sampling_recipe(recipe,result,evidence)==result


@pytest.mark.parametrize('key,value',[
    ('generated_images',1000),('reference_split','val'),('reference_images',50000),
    ('weights_unmodified',False),('training_updates_added',1),('source_run','other-run'),
    ('fid',113.),('passed',False),
])
def test_unmatched_or_failed_validation_cannot_change_sampler(key,value):
    recipe,evidence=example()
    evidence[key]=value
    with pytest.raises(ValueError):validated_sampling_recipe(recipe,recipe,evidence)


def test_sampler_override_cannot_change_optimizer_or_tokenizer():
    recipe,evidence=example()
    for field,value in [('optimization',dict(warmup_epochs=2)),('tokenizer',dict(coefficient_bins=2048))]:
        saved=copy.deepcopy(recipe);saved[field]=value
        with pytest.raises(ValueError):validated_sampling_recipe(recipe,saved,evidence)


def test_evidence_for_multiple_changes_cannot_be_used_for_one_change():
    recipe,evidence=example()
    evidence['sampling']=dict(evidence['sampling'],atom_top_p=.9)
    with pytest.raises(ValueError):validated_sampling_recipe(recipe,recipe,evidence)


def joint_example():
    recipe, evidence = example()
    source = dict(temperature=1., top_k=512, top_p=.95)
    recipe.update(sampling=source, loss=dict(type='soft_target_cross_entropy'),
                  tokenizer=dict(vocab_size=20481, codebook_sha256='fixed-codebook'))
    evidence.update(source_sampling=source, sampling=dict(source, top_k=2048))
    return recipe, evidence


def test_joint_integer_change_keeps_training_and_accepts_later_checkpoints():
    recipe, evidence = joint_example()
    original = copy.deepcopy(recipe)
    result = validated_sampling_recipe(recipe, recipe, evidence)
    assert result == dict(recipe, sampling=dict(temperature=1., top_k=2048, top_p=.95))
    assert recipe == original
    assert validated_sampling_recipe(result, recipe, evidence) == result
    assert validated_sampling_recipe(result, result, evidence) == result


@pytest.mark.parametrize('key,value', [('temperature', .9), ('top_p', .9),
                                      ('top_k', 1024), ('top_k', None)])
def test_joint_override_rejects_other_sampling_changes(key, value):
    recipe, evidence = joint_example()
    evidence['sampling'][key] = value
    with pytest.raises(ValueError):
        validated_sampling_recipe(recipe, recipe, evidence)


@pytest.mark.parametrize('field,value', [
    ('loss', dict(type='soft_target_cross_entropy', depth_weights=[1.5, 1.25, 1., .75, .5])),
    ('tokenizer', dict(vocab_size=20481, codebook_sha256='another-codebook')),
    ('optimization', dict(warmup_epochs=2)),
])
def test_joint_override_rejects_training_or_codebook_changes(field, value):
    recipe, evidence = joint_example()
    changed = copy.deepcopy(recipe)
    changed[field] = value
    with pytest.raises(ValueError):
        validated_sampling_recipe(changed, recipe, evidence)
