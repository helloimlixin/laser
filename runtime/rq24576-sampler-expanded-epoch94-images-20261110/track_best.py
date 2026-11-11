"""Keep the measured best record current while the epoch94 search is running."""
import json
import os
from pathlib import Path
import time

LOCAL=Path('/tmp/laser-rq24576-sampler-expanded-epoch94-images-20261110')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
 path=Path(path);pending=path.with_name(path.name+f'.{os.getpid()}.pending')
 pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)


def refresh():
 if not (LOCAL/'wandb.json').exists():return
 plan=read(LOCAL/'plan.json');out=Path(plan['output'])
 rows=[read(f) for f in (out/'confirm').glob('*.json')]
 if not rows:return
 for row in rows:
  assert row['checkpoint_sha256']==plan['checkpoint_sha256'] and row['generated_images']==50000
 best=min(rows,key=lambda r:r['fid']);policy=best['base_policy']
 available={r['seed'] for r in rows if r['base_policy']==policy}
 record=dict(epoch=94,source_step=plan['source_step'],fid=best['fid'],inception_score=best['inception_score'],
  is_split_std=best['inception_score_std'],sampling=best['sampling'],generation_seed=best['seed'],
  checkpoint_sha256=plan['checkpoint_sha256'],inference_checkpoint=plan['retained_inference_checkpoint'],
  evaluation_url=read(LOCAL/'wandb.json')['url'],report=str(out/'report.json'),
  completed_evaluation=str(out/'confirm'/(best['policy']+'.json')),
  single_seed_score_is_descriptive=True,seed_statistics_available=len(available)==3)
 parent=Path('/workspace/Projects/laser/outputs')/plan['source_run'].split('/')[-1]
 target=parent/'best-tuned-evaluation.json';manifest=read(target)
 if record['fid']<=manifest['best']['fid'] and manifest['best']!=record:
  manifest['best']=record
  manifest['evaluations']=[r for r in manifest['evaluations'] if r.get('report')!=record['report']]+[record]
  write(target,manifest)
  print(json.dumps(dict(event='best_record_updated',fid=record['fid'],policy=policy,seed=best['seed'])),flush=True)
 progress=dict(best_observed_single_seed=record,completed_full50k_evaluations=len(rows),
  fresh_seed_validation_complete=(out/'finalization.json').exists())
 target=out/'current-best.json'
 if not target.exists() or read(target)!=progress:write(target,progress)


def main():
 while True:
  refresh()
  plan=read(LOCAL/'plan.json');out=Path(plan['output'])
  if (out/'finalization.json').exists():break
  phase=read(LOCAL/'status.json')['phase'] if (LOCAL/'status.json').exists() else 'initializing'
  if phase in ['failed','superseded_by_epoch94']:break
  time.sleep(10)


if __name__=='__main__':main()
