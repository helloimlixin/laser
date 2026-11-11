"""Publish the expanded sampler study with separate FID and joint-objective records."""
import hashlib
import json
import os
from pathlib import Path
import time

LOCAL=Path('/tmp/laser-rq24576-sampler-expanded-epoch94-images-20261110')


def read(p):return json.loads(Path(p).read_text())


def write(p,v):
 p=Path(p);temporary=p.with_name(p.name+f'.{os.getpid()}.pending')
 temporary.write_text(json.dumps(v,indent=2)+'\n');temporary.replace(p)


def main():
 plan=read(LOCAL/'plan.json');out=Path(plan['output'])
 parent=Path('/workspace/Projects/laser/outputs')/plan['source_run'].split('/')[-1]
 while not (LOCAL/'wandb.json').exists():
  if (LOCAL/'status.json').exists() and read(LOCAL/'status.json')['phase']=='failed':raise RuntimeError('Expanded sampler study failed to initialize')
  time.sleep(2)
 state=dict(status='running',source_epoch=94,source_step=plan['source_step'],new_configurations=12,total_configurations=18,
  url=read(LOCAL/'wandb.json')['url'],report=str(out/'report.json'),checkpoint_sha256=plan['checkpoint_sha256'])
 write(parent/'epoch94-sampler-expanded.json',state)
 while True:
  if (LOCAL/'status.json').exists():
   status=read(LOCAL/'status.json')
   if status['phase']=='failed':
    state.update(status='failed',error=status.get('error'));write(parent/'epoch94-sampler-expanded.json',state)
    raise RuntimeError(status.get('error'))
   if status['phase']=='completed':break
  time.sleep(15)
 report=read(out/'report.json');images=read(out/'image-logging.json')
 assert read(out/'wandb-media-verification.json')['passed']
 for rank in range(8):
  for stem in ['verification','adapter-verification','cache-source-verification']:
   assert read(out/f'{stem}-rank-{rank}.json')['passed']
 rows=report['all_completed_evaluations']
 assert images['panel_count']==len({(r['base_policy'],r['seed']) for r in rows})
 assert images['total_images']==64*images['panel_count']
 for item in images['panels'].values():assert hashlib.sha256(Path(item['file']).read_bytes()).hexdigest()==item['sha256']
 baseline={r['seed']:r for r in report['current']['results']}
 for group in report['confirmed_challenger_summaries']:
  for row in group['results']:
   assert row['initial_cuda_rng_sha256_by_rank']==baseline[row['seed']]['initial_cuda_rng_sha256_by_rank']
 assert hashlib.sha256((Path(plan['base'])/'inputs/recipe.yaml').read_bytes()).hexdigest()==plan['active_recipe_sha256']
 retention=read(out/'checkpoint-retention.json');assert retention['verified_bytes']==retention['bytes']
 assert retention['source_checkpoint_sha256']==plan['checkpoint_sha256'] and Path(retention['path']).exists()
 best=report['best_observed_single_seed']
 record=dict(epoch=94,source_step=plan['source_step'],fid=best['fid'],inception_score=best['inception_score'],
  is_split_std=best['inception_score_std'],sampling=best['sampling'],generation_seed=best['seed'],
  checkpoint_sha256=plan['checkpoint_sha256'],inference_checkpoint=plan['retained_inference_checkpoint'],
  evaluation_url=state['url'],report=str(out/'report.json'),single_seed_score_is_descriptive=True,
  seed_statistics_available=len({r['seed'] for r in rows if r['base_policy']==best['base_policy']})==3)
 manifest=read(parent/'best-tuned-evaluation.json')
 manifest['evaluations']=[r for r in manifest['evaluations'] if r.get('report')!=record['report']]+[record]
 if record['fid']<=manifest['best']['fid']:manifest['best']=record
 evidence=dict(report=str(out/'report.json'),url=state['url'],joint_improvement_confirmed=report['improvement_confirmed_on_fresh_seeds'],
  current=report['current'],confirmed_challengers=report['confirmed_challenger_summaries'],decisions=report['challenger_decisions'],images_logged=True)
 manifest['epoch94_expanded_sampling_search']=evidence;write(parent/'best-tuned-evaluation.json',manifest)
 recommendation=read(parent/'recommended-generation-sampling.json');recommendation['epoch94_expanded_sampling_search']=evidence
 selected=report['challenger'] if report['improvement_confirmed_on_fresh_seeds'] else report['current']
 previous=read(Path(plan['predecessor_output'])/'report.json')
 if recommendation.get('validated_epoch')==94:
  old=previous['challenger'] if recommendation['sampling']==previous['challenger']['sampling'] else previous['current']
 else:old=report['epoch86_reference']['current']
 old_rows={r['seed']:r for r in old['results']}
 pairs=[dict(seed=r['seed'],fid_improvement=old_rows[r['seed']]['fid']-r['fid'],
  is_improvement=r['inception_score']-old_rows[r['seed']]['inception_score']) for r in selected['results']]
 fresh=[r for r in pairs if r['seed'] in plan['replication_seeds']]
 promoted=(report['improvement_confirmed_on_fresh_seeds'] and all(r['fid_improvement']>0 and r['is_improvement']>0 for r in fresh)
  and sum(r['fid_improvement'] for r in fresh)/2>=.1 and sum(r['is_improvement'] for r in fresh)/2>=.5
  and selected['fid_mean']<recommendation['fid_mean'] and selected['inception_score_mean']>recommendation['inception_score_mean'])
 if promoted:
  recommendation.update(sampling=selected['sampling'],validated_epoch=94,checkpoint_sha256=plan['checkpoint_sha256'],
   generation_seeds=plan['generation_seeds'],fid_mean=selected['fid_mean'],fid_seed_std=selected['fid_seed_std'],
   inception_score_mean=selected['inception_score_mean'],inception_score_seed_std=selected['inception_score_seed_std'],
   source_audit=state['url'],active_training_sampler_changed=False)
 recommendation['latest_model_evaluation']=record
 recommendation['epoch94_expanded_sampling_search']['promoted_over_previous_recommendation']=bool(promoted)
 write(parent/'recommended-generation-sampling.json',recommendation)
 state.update(status='completed',joint_sampler_improvement_confirmed=report['improvement_confirmed_on_fresh_seeds'],
  promoted_over_previous_recommendation=bool(promoted),best_single=record,image_panels=images['panel_count'],
  images_logged=images['total_images'],verification_passed=True,confirmed_challengers=report['confirmed_challenger_summaries'])
 write(parent/'epoch94-sampler-expanded.json',state);write(out/'finalization.json',state)
 print(json.dumps(state),flush=True)


if __name__=='__main__':main()
