"""Validate epoch88 results and publish records without promoting seed luck."""
import hashlib
import json
import os
from pathlib import Path
import time

LOCAL=Path('/tmp/laser-rq24576-sampler-search-epoch88-images-20261110')


def read(p):return json.loads(Path(p).read_text())


def write(p,v):
 p=Path(p);t=p.with_name(p.name+f'.{os.getpid()}.pending')
 t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def main():
 plan=read(LOCAL/'plan.json');out=Path(plan['output'])
 parent=Path('/workspace/Projects/laser/outputs')/plan['source_run'].split('/')[-1]
 while not (LOCAL/'wandb.json').exists():time.sleep(2)
 state=dict(status='running',source_epoch=88,source_step=plan['source_step'],checkpoint_sha256=plan['checkpoint_sha256'],
  configurations=len(plan['primary_policies']),primary_images_per_config=50000,previews_per_config=64,
  url=read(LOCAL/'wandb.json')['url'],report=str(out/'report.json'),training_recipe_unchanged=True)
 write(parent/'epoch88-sampler-search.json',state)
 while True:
  if (LOCAL/'status.json').exists():
   status=read(LOCAL/'status.json')
   if status['phase']=='failed':
    state.update(status='failed',error=status.get('error'));write(parent/'epoch88-sampler-search.json',state)
    raise RuntimeError(status.get('error'))
   if status['phase']=='completed':break
  time.sleep(15)
 report=read(out/'report.json');images=read(out/'image-logging.json')
 assert read(out/'wandb-media-verification.json')['passed']
 for rank in range(plan['world_size']):
  assert read(out/f'verification-rank-{rank}.json')['passed']
  assert read(out/f'adapter-verification-rank-{rank}.json')['passed']
 assert images['panel_count']==12 and images['total_images']==768
 for row in images['panels'].values():assert hashlib.sha256(Path(row['file']).read_bytes()).hexdigest()==row['sha256']
 for seed in plan['replication_seeds']:
  a=next(r for r in report['current']['results'] if r['seed']==seed)
  b=next(r for r in report['challenger']['results'] if r['seed']==seed)
  assert a['initial_cuda_rng_sha256_by_rank']==b['initial_cuda_rng_sha256_by_rank']
 assert hashlib.sha256((Path(plan['base'])/'inputs/recipe.yaml').read_bytes()).hexdigest()==plan['active_recipe_sha256']
 # Retention is independent of GPU evaluation and may finish earlier or later.
 deadline=time.monotonic()+600
 while not (out/'checkpoint-retention.json').exists():
  assert time.monotonic()<deadline,'Durable inference snapshot retention did not finish'
  time.sleep(10)
 retention=read(out/'checkpoint-retention.json');assert retention['verified_bytes']==retention['bytes']
 assert retention['source_checkpoint_sha256']==plan['checkpoint_sha256']
 manifest=read(parent/'best-tuned-evaluation.json')
 rows=[report['best_observed_single_seed']]
 def record(best):
  return dict(epoch=88,source_step=plan['source_step'],fid=best['fid'],inception_score=best['inception_score'],
   is_split_std=best['inception_score_std'],sampling=best['sampling'],generation_seed=best['seed'],
   checkpoint_sha256=plan['checkpoint_sha256'],inference_checkpoint=plan['retained_inference_checkpoint'],
   evaluation_url=state['url'],report=str(out/'report.json'),single_seed_score_is_descriptive=True,
   seed_statistics_available=best['base_policy'] in ['current',report['challenger']['policy']])
 best=record(rows[0]);manifest['evaluations']=[r for r in manifest['evaluations'] if r.get('report')!=best['report']]+[best]
 if best['fid']<manifest['best']['fid']:manifest['best']=best
 search=dict(report=str(out/'report.json'),url=state['url'],joint_improvement_confirmed=report['improvement_confirmed_on_fresh_seeds'],
  current=report['current'],challenger=report['challenger'],images_logged=True,epoch86_comparison=report['epoch88_vs_epoch86_current_sampler'])
 manifest['epoch88_sampling_search']=search;write(parent/'best-tuned-evaluation.json',manifest)
 recommendation=read(parent/'recommended-generation-sampling.json')
 recommendation['epoch88_sampling_search']=search
 selected=report['challenger'] if report['improvement_confirmed_on_fresh_seeds'] else report['current']
 prior=report['epoch86_reference']['current']
 paired=[]
 for row in selected['results']:
  old=next(r for r in prior['results'] if r['seed']==row['seed'])
  paired.append(dict(seed=row['seed'],fid_improvement=old['fid']-row['fid'],is_improvement=row['inception_score']-old['inception_score']))
 fresh=[r for r in paired if r['seed'] in plan['replication_seeds']]
 # A newer checkpoint is recommended only when both metrics improve on each fresh seed.
 promoted=(all(r['fid_improvement']>0 and r['is_improvement']>0 for r in fresh)
  and sum(r['fid_improvement'] for r in fresh)/2>=.1 and sum(r['is_improvement'] for r in fresh)/2>=.5
  and selected['fid_mean']<prior['fid_mean'] and selected['inception_score_mean']>prior['inception_score_mean'])
 if promoted:
  recommendation.update(sampling=selected['sampling'],validated_epoch=88,checkpoint_sha256=plan['checkpoint_sha256'],
   generation_seeds=plan['generation_seeds'],fid_mean=selected['fid_mean'],fid_seed_std=selected['fid_seed_std'],
   inception_score_mean=selected['inception_score_mean'],inception_score_seed_std=selected['inception_score_seed_std'],
   source_audit=state['url'],active_training_sampler_changed=False)
 recommendation['latest_model_evaluation']=best
 recommendation['epoch88_sampling_search']['promoted_over_epoch86']=promoted
 recommendation['epoch88_sampling_search']['paired_with_epoch86']=paired
 write(parent/'recommended-generation-sampling.json',recommendation)
 state.update(status='completed',joint_sampler_improvement_confirmed=report['improvement_confirmed_on_fresh_seeds'],
  recommended_over_epoch86=promoted,best_single=best,epoch88_selected_summary={k:v for k,v in selected.items() if k!='results'},
  paired_with_epoch86=paired,image_panels=images['panel_count'],images_logged=images['total_images'],verification_passed=True)
 write(parent/'epoch88-sampler-search.json',state);write(out/'finalization.json',state)
 print(json.dumps(state),flush=True)


if __name__=='__main__':main()
