import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('hard_target_trial',
    Path(__file__).resolve().parents[1]/'scripts/tools/run_joint20k_hard_target_test.py')
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def recipes():
    control = dict(run_id='control', name='control', version='1', epochs=100,
        lr_schedule_epochs=100, global_batch=2048, microbatch=64,
        training_rng=dict(policy='independent_rank_v1', base_seed=261001),
        targets=dict(temperature=[.1]*5, chunk_size=128, stochastic_prefixes=True,
                     calibration={'policy':'old'}), objective='native_soft_target_cross_entropy',
        sampling=dict(temperature=1., top_k=2048, top_p=.95),
        tokenizer=dict(depth=5, codebook_sha256='frozen'))
    candidate = copy.deepcopy(control)
    candidate.update(run_id='trial', objective='hard_target_cross_entropy')
    candidate['targets'].update(policy='hard_deterministic', stochastic_prefixes=False)
    candidate['targets'].pop('calibration')
    return control, candidate


def test_only_target_policy_can_change():
    control, candidate = recipes()
    trial.assert_hard_target_only(control, candidate)
    candidate['optimization'] = {'probe_initial_weight_sha256':'same'}
    trial.assert_hard_target_only(control, candidate)


@pytest.mark.parametrize('key,value', [
    ('sampling',dict(temperature=1.,top_k=1024,top_p=.95)),
    ('tokenizer',dict(depth=4,codebook_sha256='frozen')),
    ('global_batch',1024), ('lr_schedule_epochs',4),
    ('training_rng',dict(policy='independent_rank_v1',base_seed=123)),
    ('objective','native_soft_target_cross_entropy')])
def test_trial_rejects_other_changes(key, value):
    control, candidate = recipes()
    candidate[key] = value
    with pytest.raises(ValueError):
        trial.assert_hard_target_only(control, candidate)
