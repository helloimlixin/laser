"""Publish confirmed sampler winners and verify every official result online."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def record(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','support','key-file'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    deadline=time.monotonic()+18*3600
    while time.monotonic()<deadline:
        status=json.loads((args.output/'sweep-status.json').read_text())
        if status['state']=='completed':break
        if status['state'] in ('failed','cancelled'):
            record(args.output/'online-completion-proof.json',dict(complete=False,sweep_status=status,time=time.time()))
            return
        time.sleep(20)
    else:raise RuntimeError('Timed out waiting for official sampling sweep')
    sys.path.insert(0,str(args.support))
    from verified_wandb_checkpoint_upload import VerifiedCloudUpload
    os.environ['WANDB_API_KEY']=args.key_file.read_text().strip()
    import wandb
    run_path='helloimlixin-rutgers/laser/'+args.run_id
    run=wandb.Api(timeout=120).run(run_path)
    results=json.loads((args.output/'official-comparison-results.json').read_text())
    selection=json.loads((args.output/'official-sampler-selection.json').read_text())
    if selection['provisional']:
        raise RuntimeError('Sweep finished without independent-seed confirmation')
    best_fid=min(results,key=lambda k:results[k]['fid'])
    best_is=max(results,key=lambda k:results[k]['inception_score'])
    confirmed=selection['confirmed_best_fid_policy']
    record(args.output/'official-confirmed-sampler-selection.json',dict(
        confirmed_policy=confirmed,two_seed_mean_fid=selection['two_seed_mean_fid'][confirmed],
        confirmation_candidates=selection['two_seed_mean_fid'],raw_best_fid_evaluation=best_fid,
        raw_best_fid=results[best_fid]['fid'],raw_best_is_evaluation=best_is,
        raw_best_inception_score=results[best_is]['inception_score'],
        source_epoch=77,source_global_step=48202,completed_evaluations=len(results),
        confirmation_seed=selection['confirmation_seed'],exploration_seed=selection['exploration_seed'],
        policy=results[confirmed]['policy'],all_evaluations_official50k=True))
    for kind,name in [('best-fid',best_fid),('best-is',best_is),('confirmed-best-fid',confirmed)]:
        shutil.copyfile(args.output/f'official-{name}-samples.png',args.output/f'{kind}-samples.png')
    paths=[args.output/name for name in ('official-confirmed-sampler-selection.json',
           'best-fid-samples.png','best-is-samples.png','confirmed-best-fid-samples.png','last-samples.png')]
    VerifiedCloudUpload(run_path,args.output/'confirmed-selection-cloud-receipt.json')(paths,epoch=77)
    verified=[]
    for name in results:
        for filename in (f'official-{name}.json',f'official-{name}-samples.png',f'completed-{name}.json'):
            path=args.output/filename
            with path.open('rb') as stream:
                md5=base64.b64encode(hashlib.file_digest(stream,'md5').digest()).decode()
            remote=run.file(filename)
            if remote.size!=path.stat().st_size or remote.md5!=md5:
                raise RuntimeError('Official result verification failed: '+filename)
            verified.append(dict(name=filename,bytes=remote.size,md5=remote.md5))
    proof=args.output/'online-completion-proof.json'
    record(proof,dict(complete=True,official_evaluations=len(results),verified_files=verified,
        confirmed_policy=confirmed,raw_best_fid_evaluation=best_fid,raw_best_is_evaluation=best_is,time=time.time()))
    VerifiedCloudUpload(run_path,args.output/'completion-proof-cloud-receipt.json')([proof],epoch=77)
    run.summary.update({'selection/confirmed_best_fid_policy':confirmed,
        'selection/raw_best_fid_evaluation':best_fid,'selection/raw_best_is_evaluation':best_is,
        'execution/completion_verified':True,'evaluation/official_results_pending':False})


if __name__=='__main__':main()
