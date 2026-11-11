"""Read only measured generation overrides; keep the training recipe immutable."""
import hashlib
import json
import math
from pathlib import Path


def normalize(sampling):
    assert set(sampling)=={'temperature','top_k','top_p'}
    t=float(sampling['temperature']);assert math.isfinite(t) and t>0
    k=sampling['top_k'];k=[k] if isinstance(k,int) else list(k)
    p=sampling['top_p'];p=[p] if isinstance(p,(int,float)) else list(p)
    assert len(k) in (1,4) and all(isinstance(x,int) and not isinstance(x,bool) and 1<=x<=16384 for x in k)
    assert len(p) in (1,4) and all(math.isfinite(float(x)) and 0<float(x)<=1 for x in p)
    return dict(temperature=t,top_k=k,top_p=[float(x) for x in p])


def validate(evidence):
    assert evidence['passed'] and evidence['improved'] and evidence['weights_unmodified']
    assert evidence['training_updates_added']==0 and evidence['generated_images']==50000
    assert evidence['reference_images']==1281167 and evidence['reference_split']=='train'
    assert evidence['fid']<=evidence['baseline_fid']-evidence['minimum_fid_improvement']
    repeated=evidence.get('repeat_confirmation')
    if not repeated:assert evidence['minimum_fid_improvement']>=max(.5,.01*evidence['baseline_fid'])
    rows=evidence['confirmation']
    assert len({r['model_state_sha256'] for r in rows})==1
    assert len({r['checkpoint_sha256'] for r in rows})==1
    assert len({r['seed'] for r in rows})==1
    assert len({r['sampling_batch_per_rank'] for r in rows})==1
    for r in rows:
        assert r['generated_images']==50000 and r['weights_unmodified'] and r['training_updates_added']==0
        assert r['reference_images']==1281167 and r['reference_split']=='train'
        assert r['checkpoint_sha256']==evidence['checkpoint_sha256']
        assert r['reference_sha256']=='3f9c92d15755e76ec312964a819e3e19b9cf3aadc618a7ae99f5e8aa96501260'
        assert not r['rejection_sampling'] and r['source_step']==evidence['source_step']
    winner=next(r for r in rows if r['policy']==evidence['selected_policy'])
    baseline=next(r for r in rows if r['policy']=='baseline')
    assert baseline['fid']==evidence['baseline_fid'] and baseline['sampling']==evidence['source_sampling']
    assert baseline['inception_score']==evidence['baseline_inception_score']
    assert winner['sampling']==evidence['sampling'] and winner['fid']==evidence['fid']
    assert winner['inception_score']>=.95*evidence['baseline_inception_score']
    if repeated:
        assert len(repeated)==2 and len({r['seed'] for r in repeated})==1
        assert repeated[0]['seed']!=rows[0]['seed']
        for r in repeated:
            assert r['checkpoint_sha256']==evidence['checkpoint_sha256'] and r['model_state_sha256']==rows[0]['model_state_sha256']
            assert r['generated_images']==50000 and r['reference_images']==1281167 and r['reference_split']=='train'
            assert r['reference_sha256']==rows[0]['reference_sha256'] and r['sampling_batch_per_rank']==rows[0]['sampling_batch_per_rank']
            assert r['weights_unmodified'] and r['training_updates_added']==0 and not r['rejection_sampling']
            assert r['source_step']==evidence['source_step']
        repeat_base=next(r for r in repeated if r['policy']=='baseline')
        repeat_winner=next(r for r in repeated if r['policy']==evidence['selected_policy'])
        assert repeat_base['sampling']==baseline['sampling'] and repeat_winner['sampling']==winner['sampling']
        delta1=baseline['fid']-winner['fid'];delta2=repeat_base['fid']-repeat_winner['fid']
        assert delta1>0 and delta2>0 and (delta1+delta2)/2>=.3
        assert repeat_winner['inception_score']>=.95*repeat_base['inception_score']
    return normalize(evidence['sampling'])


def settings(profile):
    if not hasattr(profile,'_calibrated_sampling_cache'):
        path=profile.base/'inputs/generation-sampling-override.json'
        value=None
        if path.exists():
            evidence=json.loads(path.read_text());value=validate(evidence)
            assert evidence['source_run']==profile.run_id
            assert evidence['training_recipe_sha256']==hashlib.sha256((profile.base/'inputs/recipe.yaml').read_bytes()).hexdigest()
            assert evidence['codebook_sha256']==profile.codebook_sha256
            winner=next(r for r in evidence['confirmation'] if r['policy']==evidence['selected_policy'])
            batch=int(winner['sampling_batch_per_rank']);assert 1<=batch<=256
            profile.generation_batch=batch
            profile._calibrated_sampling_evidence=evidence
        profile._calibrated_sampling_cache=value
    return profile._calibrated_sampling_cache


def verify_resume(profile,payload):
    if settings(profile) is not None:
        assert payload['global_step']>=profile._calibrated_sampling_evidence['source_step']


def log_settings(run,profile):
    sampling=settings(profile)
    if sampling is None:return
    evidence=profile._calibrated_sampling_evidence
    run.config.update(dict(active_generation_sampling=sampling,native_sampling=sampling,
        sampler_temperature=sampling['temperature'],sampler_top_k=sampling['top_k'][0],sampler_top_p=sampling['top_p'][0],
        sampler_top_k_by_depth=sampling['top_k']*4 if len(sampling['top_k'])==1 else sampling['top_k'],
        sampler_top_p_by_depth=sampling['top_p']*4 if len(sampling['top_p'])==1 else sampling['top_p'],
        active_generation_batch_per_rank=profile.generation_batch,
        sampler_calibration_step=evidence['source_step']),allow_val_change=True)
    run.summary.update({'sampling/calibration_step':evidence['source_step'],
        'sampling/calibration_baseline_fid':evidence['baseline_fid'],'sampling/calibration_selected_fid':evidence['fid'],
        'sampling/calibration_weights_frozen':True,'sampling/training_recipe_unchanged':True})
    if evidence.get('repeat_confirmation'):
        run.summary.update({'sampling/calibration_fid_gain_by_seed':evidence['fid_improvement_by_seed'],
            'sampling/calibration_mean_fid_gain':evidence['mean_fid_improvement'],
            'sampling/calibration_independent_confirmation_seeds':2})

