"""Verify and retain the sampler audit, without changing live training."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

LOCAL=Path('/tmp/laser-rq24576-sampler-refine-epoch80-20261109')


def write(path,value):
    temporary=path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    plan=json.loads((LOCAL/'plan.json').read_text())
    out=Path(plan['output']);report=json.loads((out/'report.json').read_text())
    assert json.loads((out/'status.json').read_text())['phase']=='completed'
    assert hashlib.sha256((Path(plan['base'])/'inputs/recipe.yaml').read_bytes()).hexdigest()==plan['active_recipe_sha256']
    proofs=[json.loads((out/f'verification-rank-{rank}.json').read_text()) for rank in range(8)]
    assert all(p['passed'] and p['model_weights_unmodified'] and p['tokenizer_unchanged'] and p['active_recipe_unchanged'] for p in proofs)
    assert len({p['tokenizer_digest'] for p in proofs})==1
    for summary in [report['current'],report['challenger']]:
        rows=summary['results']
        assert sorted(row['seed'] for row in rows)==sorted(plan['generation_seeds'])
        assert all(row['generated_images']==50000 and row['checkpoint_sha256']==plan['checkpoint_sha256'] and row['sampling']==summary['sampling'] for row in rows)
        for values,mean_key,std_key in [([row['fid'] for row in rows],'fid_mean','fid_seed_std'),([row['inception_score'] for row in rows],'inception_score_mean','inception_score_seed_std')]:
            assert math.isclose(statistics.mean(values),summary[mean_key],rel_tol=1e-12,abs_tol=1e-12)
            assert math.isclose(statistics.stdev(values),summary[std_key],rel_tol=1e-12,abs_tol=1e-12)
    all_cases=report['primary_confirmation']+report['current']['results']+report['challenger']['results']
    rng_by_seed={}
    for case in all_cases:
        hashes=case.get('initial_cuda_rng_sha256_by_rank')
        if hashes is None:
            assert case['policy']=='current__s261001' and case['reused_verified_result']
            continue
        assert len(hashes)==8 and len(set(hashes))==8
        if case['seed'] in rng_by_seed:assert hashes==rng_by_seed[case['seed']]
        rng_by_seed[case['seed']]=hashes
    assert sorted(rng_by_seed)==sorted(plan['generation_seeds'])
    assert len({value for hashes in rng_by_seed.values() for value in hashes})==24
    write(out/'seed-routing-final-verification.json',dict(passed=True,three_distinct_generation_seed_groups=True,all_eight_ranks_paired_between_samplers=True,distinct_initial_cuda_rng_states=24))
    selected=report['challenger'] if report['improvement_confirmed_on_fresh_seeds'] else report['current']
    summary=dict(source_run=plan['source_run'],epoch=plan['source_epoch'],global_step=plan['source_step'],checkpoint_sha256=plan['checkpoint_sha256'],inference_checkpoint=plan['retained_inference_checkpoint'],sampling=selected['sampling'],generation_seeds=plan['generation_seeds'],generated_images_per_seed=50000,reference_images=plan['reference_images'],reference_split='train',current=report['current'],challenger=report['challenger'],paired_seed_comparisons=report['paired_seed_comparisons'],mean_fid_improvement=report['mean_fid_improvement'],fresh_mean_fid_improvement=report['fresh_mean_fid_improvement'],improvement_confirmed_on_fresh_seeds=report['improvement_confirmed_on_fresh_seeds'],best_observed_single_seed=report['best_observed_single_seed'],uncertainty=report['limits'],audit_url='https://forge.coreweave.com/wandb/helloimlixin-rutgers/laser/runs/'+out.name,active_training_sampler_changed=False,training_seed_changed=False,time=time.time())
    training_out=Path('/workspace/Projects/laser/outputs/imagenet-rq24576-native-scratch100-8h100-20261108')
    write(training_out/'sampler-refinement-epoch80-summary.json',summary)
    write(training_out/'recommended-generation-sampling.json',dict(sampling=selected['sampling'],validated_epoch=80,checkpoint_sha256=plan['checkpoint_sha256'],generation_seeds=plan['generation_seeds'],fid_mean=selected['fid_mean'],fid_seed_std=selected['fid_seed_std'],inception_score_mean=selected['inception_score_mean'],inception_score_seed_std=selected['inception_score_seed_std'],source_audit=summary['audit_url'],active_training_sampler_changed=False))
    manifest_path=training_out/'best-tuned-evaluation.json'
    manifest=json.loads(manifest_path.read_text())
    best_seed=min(selected['results'],key=lambda row:row['fid'])
    if best_seed['fid']<manifest['best']['fid']:
        manifest['best']=dict(epoch=80,source_step=50080,fid=best_seed['fid'],
            inception_score=best_seed['inception_score'],sampling=selected['sampling'],
            generation_seed=best_seed['seed'],fid_mean=selected['fid_mean'],
            fid_seed_std=selected['fid_seed_std'],checkpoint_sha256=plan['checkpoint_sha256'],
            inference_checkpoint=plan['retained_inference_checkpoint'],
            evaluation_url=summary['audit_url'],report=str(out/'report.json'),
            single_seed_score_is_descriptive=True)
    manifest['sampler_refinement_summary']=str(training_out/'sampler-refinement-epoch80-summary.json')
    manifest['selection']='Lowest observed FID50k; sampler recommendation uses two fresh matched seeds. Seed means and variation are recorded separately.'
    manifest['time']=time.time()
    write(manifest_path,manifest)
    training=json.loads((training_out/'status.json').read_text())
    write(out/'completion-verification.json',dict(passed=True,frozen_weights_verified_all_eight_ranks=True,seed_routing_verified=True,seed_means_and_sample_standard_deviations_verified=True,training_recipe_unchanged=True,training_progress=training['progress'],time=time.time()))
    os.environ['WANDB_API_KEY']=Path('/root/.config/laser/wandb.key').read_text().strip()
    os.environ['WANDB_BASE_URL']='https://api.wandb.ai'
    import wandb
    run=wandb.Api(timeout=30).run('helloimlixin-rutgers/laser/'+out.name)
    assert run.state=='finished'
    assert abs(run.summary['recommended_fid_mean']-report['recommended_fid_mean'])<1e-10
    print(json.dumps(dict(passed=True,policy=selected['policy'],sampling=selected['sampling'],fid_mean=selected['fid_mean'],fid_seed_std=selected['fid_seed_std'],inception_score_mean=selected['inception_score_mean'],inception_score_seed_std=selected['inception_score_seed_std'],fresh_mean_fid_improvement=report['fresh_mean_fid_improvement'],training_epoch=training['progress']['train/epoch'],training_images_per_second=training['progress']['train/images_per_second'],wandb_url=run.url)),flush=True)


if __name__=='__main__':main()
