import copy
import pytest

from src.training.laser_identity_pilot_analysis import analyze,paired_bootstrap


def report(atom=4.,coeff=2.):
    return dict(probe_sha256_by_rank=['same'],rows=[dict(index=i,atom_ce=[atom+i*.01]*4,
        coefficient_ce=[coeff]*4,coefficient_entropy=[1.]*4,sign_accuracy=[.9]*4,atom_top1=[.1]*4) for i in range(32)])


def test_gate_accepts_consistent_support_and_joint_improvement():
    result=dict(hashes_by_rank=[dict(order='same',targets='same')])
    actual=analyze(report(5.),report(5.),report(4.),report(3.9),result,result)
    assert actual['qualifies_for_fid'] and actual['matched_initial_predictions']
    assert actual['atom_candidate_minus_control']['interval95'][1]<0


@pytest.mark.parametrize('candidate',[report(4.1),report(3.9,2.2)])
def test_gate_rejects_worse_support_or_joint_prediction(candidate):
    result=dict(hashes_by_rank=['same'])
    assert not analyze(report(5.),report(5.),report(4.),candidate,result,result)['qualifies_for_fid']


def test_matching_evidence_is_required_and_bootstrap_uses_images():
    result=dict(hashes_by_rank=['same'])
    with pytest.raises(ValueError):analyze(report(),report(),report(),report(),result,dict(hashes_by_rank=['different']))
    altered=copy.deepcopy(report());altered['rows'][0]['index']=999
    with pytest.raises(ValueError):analyze(report(),report(),report(),altered,result,result)
    summary=paired_bootstrap([-.1]*32,draws=256)
    assert summary['images']==32 and summary['interval95'][1]<0
