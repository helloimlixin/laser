"""Paired held-out gate for the single-factor atom-identity continuation."""
import numpy as np


def paired_bootstrap(delta, *, seed=261016782, draws=10000):
    delta=np.asarray(delta,dtype=np.float64)
    if delta.ndim!=1 or len(delta)<2 or not np.isfinite(delta).all():
        raise ValueError('Expected finite paired per-image differences')
    rng=np.random.default_rng(seed)
    # Bound temporary memory independently of the probe size.
    means=[]
    for first in range(0,draws,256):
        index=rng.integers(0,len(delta),size=(min(256,draws-first),len(delta)))
        means.extend(delta[index].mean(-1).tolist())
    return dict(mean=float(delta.mean()),interval95=np.quantile(means,[.025,.975]).tolist(),
                images=len(delta),resampling_unit='image',bootstrap_draws=draws,seed=seed)


def analyze(initial_control,initial_candidate,control,candidate,control_result,candidate_result):
    if control_result['hashes_by_rank']!=candidate_result['hashes_by_rank']:
        raise ValueError('Training images, labels, targets, coefficient prefixes or order differed')
    reports=[initial_control,initial_candidate,control,candidate]
    indices=[[row['index'] for row in report['rows']] for report in reports]
    if any(values!=indices[0] for values in indices[1:]):
        raise ValueError('Held-out image populations differed')
    if any(report['probe_sha256_by_rank']!=reports[0]['probe_sha256_by_rank'] for report in reports[1:]):
        raise ValueError('Held-out target caches differed')
    def values(report,key):return np.asarray([row[key] for row in report['rows']],dtype=np.float64)
    for key in ['atom_ce','coefficient_ce','coefficient_entropy','sign_accuracy','atom_top1']:
        np.testing.assert_allclose(values(initial_control,key),values(initial_candidate,key),rtol=0,atol=1.e-6)
    a_control=values(control,'atom_ce').mean(-1);a_candidate=values(candidate,'atom_ce').mean(-1)
    a_initial=values(initial_control,'atom_ce').mean(-1)
    c_control=values(control,'coefficient_ce').mean(-1);c_candidate=values(candidate,'coefficient_ce').mean(-1)
    atom=paired_bootstrap(a_candidate-a_control)
    joint=paired_bootstrap(a_candidate+c_candidate-a_control-c_control)
    conditions=dict(candidate_atom_ce_below_frozen_source=bool(a_candidate.mean()<a_initial.mean()),
        candidate_atom_ce_upper95_below_control=bool(atom['interval95'][1]<0),
        candidate_joint_pair_ce_no_worse_than_control=bool(joint['mean']<=0))
    return dict(passed=True,qualifies_for_fid=all(conditions.values()),conditions=conditions,
        images=len(a_candidate),source_step=1252,tested_step=1380,additional_updates_per_branch=128,
        atom_candidate_minus_control=atom,joint_pair_candidate_minus_control=joint,
        frozen_source_atom_ce=float(a_initial.mean()),control_atom_ce=float(a_control.mean()),
        candidate_atom_ce=float(a_candidate.mean()),
        sign_accuracy_delta_per_depth=(values(candidate,'sign_accuracy')-values(control,'sign_accuracy')).mean(0).tolist(),
        matched_training_hashes=True,matched_initial_predictions=True,automatic_adoption=False,
        scope='Paired prediction losses for identical tokenizations and held-out images; generation FID is a separate gate.')
