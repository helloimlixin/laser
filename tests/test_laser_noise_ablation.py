import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('laser_noise_ablation',
    Path(__file__).resolve().parents[1]/'scripts/tools/run_laser_joint20k_noise_ablation.py')
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def recipe():
    return dict(run_id='control', name='control', version='1', epochs=100,
        lr_schedule_epochs=100, global_batch=2048, microbatch=64,
        training_rng=dict(policy='independent_rank_v1', base_seed=261001),
        targets=dict(temperature=[.1]*5, chunk_size=128, calibration={'policy':'old'}),
        sampling=dict(temperature=1., top_k=2048, top_p=.95),
        tokenizer=dict(depth=5, codebook_sha256='frozen'))


def test_trial_rejects_sampler_model_batch_or_rng_changes():
    control = recipe()
    for key, value in [('sampling',dict(temperature=1.,top_k=1024,top_p=.95)),
                       ('tokenizer',dict(depth=4,codebook_sha256='frozen')),
                       ('global_batch',1024),('lr_schedule_epochs',4),
                       ('training_rng',dict(policy='independent_rank_v1',base_seed=123))]:
        candidate = copy.deepcopy(control)
        candidate[key] = value
        with pytest.raises(ValueError):
            ablation.assert_temperature_only(control, candidate, [.1]*5)


def test_trial_can_change_only_temperatures_and_record_provenance():
    control = recipe()
    candidate = copy.deepcopy(control)
    candidate.update(run_id='trial',name='trial',version='2',lineage={'control':'control'},
                     optimization={'probe_initial_weight_sha256':'known'})
    candidate['targets'].update(temperature=[.2]*5, calibration={'hypothesis':'test'})
    ablation.assert_temperature_only(control,candidate,[.2]*5)


def test_restore_early_does_not_change_later_depths():
    current, earlier = [.01,.02,.03,.04,.05], [.1,.2,.3,.4,.5]
    variants = ablation.temperature_variants(current, earlier)
    assert variants == {'earlier':earlier,'restore_early':[.1,.2,.03,.04,.05]}
    assert current == [.01,.02,.03,.04,.05]


@pytest.mark.parametrize('bad', [[0.]*5,[float('nan')]*5,[.1]*4])
def test_invalid_temperature_rejected(bad):
    with pytest.raises(ValueError):
        ablation.temperature_variants([.1]*5,bad)


def evaluation(epoch, fid):
    return {'train/global_step':epoch*626,'eval/fid':fid,'generated_images':50000,
            'real_images':1281167,'real_split':'train','rejection_sampling':False}


def test_comparison_uses_matched_epoch_and_positive_improvement_means_lower_fid():
    rows = ablation.comparison_rows({'current':{2:evaluation(2,70.),4:evaluation(4,55.)},
        'earlier':{2:evaluation(2,68.)},'restore_early':{4:evaluation(4,54.)}})
    assert rows[1]['improvement_over_current'] == {'earlier':2.}
    assert rows[2]['improvement_over_current'] == {'restore_early':1.}
    assert 'earlier' not in rows[2]['fid']


@pytest.mark.parametrize('key,value', [('generated_images',10000),('real_split','val'),
    ('train/global_step',500),('rejection_sampling',True),('eval/fid',float('nan'))])
def test_incompatible_evaluations_rejected(key,value):
    row = evaluation(2,70.)
    row[key] = value
    with pytest.raises(ValueError):
        ablation.comparison_rows({'current':{2:row}})
