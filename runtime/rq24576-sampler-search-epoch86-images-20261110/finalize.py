"""Publish validated search evidence and keep single-seed records distinct."""
import hashlib
import json
import os
from pathlib import Path
import time

LOCAL=Path('/tmp/laser-rq24576-sampler-search-epoch86-images-20261110')


def read(p): return json.loads(Path(p).read_text())


def write(path,value):
    path=Path(path)
    pending=path.with_name(path.name+f'.{os.getpid()}.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def main():
    plan=read(LOCAL/'plan.json');out=Path(plan['output'])
    parent=Path('/workspace/Projects/laser/outputs')/plan['source_run'].split('/')[-1]
    evidence=dict(status='running',source_epoch=86,checkpoint_sha256=plan['checkpoint_sha256'],
        baseline_fid=13.05785712569508,baseline_is=100.89720153808594,
        configurations=len(plan['screen_policies']),images_per_config=64,
        report=str(out/'report.json'),url=read(LOCAL/'wandb.json')['url'],
        training_recipe_unchanged=True)
    write(parent/'epoch86-sampler-search.json',evidence)
    while True:
        status=read(LOCAL/'status.json')
        if status['phase']=='failed':
            evidence.update(status='failed',error=status.get('error'));write(parent/'epoch86-sampler-search.json',evidence)
            raise RuntimeError(status.get('error'))
        if status['phase']=='completed': break
        time.sleep(15)
    report=read(out/'report.json');images=read(out/'image-logging.json')
    assert read(out/'wandb-media-verification.json')['passed']
    for rank in range(plan['world_size']):
        assert read(out/f'verification-rank-{rank}.json')['passed']
        assert read(out/f'adapter-verification-rank-{rank}.json')['passed']
    assert len([k for k in images['panels'] if k.endswith('__s261001')])==24
    assert images['panel_count']==28 and images['total_images']==1792
    for item in images['panels'].values():
        assert hashlib.sha256(Path(item['file']).read_bytes()).hexdigest()==item['sha256']
    rows=[read(LOCAL/'confirm/current__s261001.json')]+report['primary_confirmation']+report['current']['results'][1:]+report['challenger']['results'][1:]
    # Every fresh matched seed must begin from identical per-rank CUDA RNG states.
    for seed in plan['replication_seeds']:
        pair=[r for r in rows if r['seed']==seed]
        assert len(pair)==2
        assert pair[0]['initial_cuda_rng_sha256_by_rank']==pair[1]['initial_cuda_rng_sha256_by_rank']
    assert hashlib.sha256((Path(plan['base'])/'inputs/recipe.yaml').read_bytes()).hexdigest()==plan['active_recipe_sha256']
    manifest=read(parent/'best-tuned-evaluation.json')
    best=report['best_observed_single_seed']
    record=dict(epoch=86,source_step=plan['source_step'],fid=best['fid'],inception_score=best['inception_score'],
        is_split_std=best['inception_score_std'],sampling=best['sampling'],generation_seed=best['seed'],
        checkpoint_sha256=plan['checkpoint_sha256'],inference_checkpoint=plan['retained_inference_checkpoint'],
        evaluation_url=evidence['url'],report=str(out/'report.json'),single_seed_score_is_descriptive=True,
        seed_statistics_available=best['base_policy'] in ['current',report['challenger']['policy']])
    manifest['evaluations']=[r for r in manifest['evaluations'] if r.get('report')!=record['report']]+[record]
    if record['fid']<manifest['best']['fid']: manifest['best']=record
    manifest['epoch86_sampling_search']=dict(report=str(out/'report.json'),url=evidence['url'],
        joint_improvement_confirmed=report['improvement_confirmed_on_fresh_seeds'],
        current=report['current'],challenger=report['challenger'],images_logged=True)
    write(parent/'best-tuned-evaluation.json',manifest)
    recommendation=read(parent/'recommended-generation-sampling.json')
    recommendation['epoch86_sampling_search']=dict(report=str(out/'report.json'),url=evidence['url'],
        joint_improvement_confirmed=report['improvement_confirmed_on_fresh_seeds'],
        current=report['current'],challenger=report['challenger'],images_logged=True)
    selected=report['challenger'] if report['improvement_confirmed_on_fresh_seeds'] else report['current']
    recommendation.update(sampling=selected['sampling'],validated_epoch=86,checkpoint_sha256=plan['checkpoint_sha256'],
        generation_seeds=plan['generation_seeds'],fid_mean=selected['fid_mean'],fid_seed_std=selected['fid_seed_std'],
        inception_score_mean=selected['inception_score_mean'],inception_score_seed_std=selected['inception_score_seed_std'],
        source_audit=evidence['url'],active_training_sampler_changed=False)
    recommendation['latest_model_evaluation']=record
    write(parent/'recommended-generation-sampling.json',recommendation)
    evidence.update(status='completed',joint_improvement_confirmed=report['improvement_confirmed_on_fresh_seeds'],
        recommended_sampling=selected['sampling'],fid_mean=selected['fid_mean'],inception_score_mean=selected['inception_score_mean'],
        image_panels=images['panel_count'],images_logged=images['total_images'],verification_passed=True)
    write(parent/'epoch86-sampler-search.json',evidence)
    write(out/'finalization.json',evidence)
    print(json.dumps(evidence),flush=True)


if __name__=='__main__': main()
