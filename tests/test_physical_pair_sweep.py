import copy
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from scripts.tools.sweep_physical_pair_sampling import (
    completed_result, confirmation_names, digest, load_plan)
from src import physical_pair_sampling as helper


PLAN = Path(__file__).resolve().parents[1] / 'outputs/sampling-broad-sweep-20261006/sweep-plan.json'


def test_plan_covers_distinct_effective_laws_and_independent_confirmation_seed():
    plan, settings = load_plan(PLAN, helper)
    assert len(settings) == 24
    assert {p.candidate_atoms for p in settings.values()} >= {16,32,64}
    assert plan['seed'] != plan['confirmation_seed']


def test_dead_coefficient_nucleus_axis_is_rejected(tmp_path):
    plan = json.loads(PLAN.read_text())
    duplicate = copy.deepcopy(plan['policies'][0])
    duplicate['name'] = 'dead-axis'
    duplicate['policy']['coefficient_top_p'] = .7
    plan['policies'].append(duplicate)
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Duplicate effective'):
        load_plan(path, helper)


def test_resume_requires_matching_protocol_seed_and_untampered_grid(tmp_path):
    policy = helper.PairSamplingPolicy()
    protocol = {'checkpoint':'fixed77'}
    result = dict(protocol=protocol, policy=asdict(policy), seed=12,
                  fid=15., inception_score=75., inception_score_std=1.)
    path = tmp_path / 'official-test.json'
    path.write_text(json.dumps(result))
    assert completed_result(tmp_path,'test',policy,12,protocol) is None
    grid = tmp_path / 'official-test-samples.png'
    grid.write_bytes(b'grid')
    (tmp_path / 'completed-test.json').write_text(json.dumps(
        dict(result_sha256=digest(path),grid_sha256=digest(grid))))
    assert completed_result(tmp_path,'test',policy,12,protocol)['fid'] == 15.
    with pytest.raises(ValueError,match='incompatible'):
        completed_result(tmp_path,'test',policy,13,protocol)
    grid.write_bytes(b'changed')
    with pytest.raises(ValueError,match='digest changed'):
        completed_result(tmp_path,'test',policy,12,protocol)


def test_confirmation_retains_controls_and_independent_metric_leader():
    settings = dict(anchor=None, **{'native-control':None,'fid-one':None,'fid-two':None,'is-leader':None})
    results = {k:dict(fid=fid,inception_score=score) for k,fid,score in [
        ('anchor',15,75),('native-control',16,70),('fid-one',13,73),('fid-two',14,74),('is-leader',17,80)]}
    assert confirmation_names(results,settings) == ['fid-one','fid-two','is-leader','anchor','native-control']
