#!/usr/bin/env python3
"""Pin W&B run files once and stage the same verified checkpoint on every node."""
import argparse,base64,hashlib,json,os,shutil,time
from pathlib import Path
from church_compound_support import sha,atomic_json,validate_recovered_checkpoint

RUN_ID='church-laser-rfid421-ft3-compound-scratch90-20260918'

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['resolve','stage','cache-finish','cache-check','finish'])
    p.add_argument('--base',type=Path,required=True);p.add_argument('--selection',type=Path);p.add_argument('--local',type=Path)
    args=p.parse_args();base=args.base
    preflight=json.loads((base/'preflight.json').read_text())
    if args.action in ('cache-finish','cache-check'):
        import torch
        cache=base/'prepared/compound-cache.pt';report=json.loads(cache.with_suffix('.validation.json').read_text())
        assert report['passed'] and report['atom_exact_fraction']==1. and report['items']==126227
        payload=torch.load(cache,map_location='cpu',weights_only=True,mmap=True);meta=payload['meta']
        assert meta['format']=='laser_compound_pairs_v1' and meta['shape']==[8,8,4]
        assert meta['coeff_scales']==preflight['coeff_scales'] and meta['items']==126227
        assert meta['num_atoms']==16384 and meta['coeff_vocab_size']==2048 and meta['coeff_max']==3.
        assert meta['tokenizer_sha256']==preflight['tokenizer']['export_sha256']
        assert payload['atoms'].shape==payload['coeffs'].shape==(126227,8,8,4)
        assert payload['atoms'].dtype==torch.int16 and payload['coeffs'].dtype==torch.float16
        assert report['duplicate_atom_within_support_fraction']==0.
        assert json.loads((base/'prepared/data-validation.json').read_text())['passed']
        # Paths may move across Amarel's filesystem caches; hashes define identity.
        assert sha(base/'prepared/tokenizer.pt')==preflight['tokenizer']['export_sha256']
        if args.action=='cache-finish':
            record=dict(passed=True,cache_sha256=sha(cache),tokenizer_sha256=preflight['tokenizer']['export_sha256'],
                        source_run=RUN_ID,items=126227,shape=[8,8,4],coeff_scales=meta['coeff_scales'],
                        stored_values='OMP supports and continuous normalized coefficients, not sampled bin ids',
                        training_targets='recomputed stochastic soft targets each visit',validation=report)
            atomic_json(base/'prepared/cache-ready.json',record)
        else:
            record=json.loads((base/'prepared/cache-ready.json').read_text())
            assert record['cache_sha256']==sha(cache)
            assert record['tokenizer_sha256']==preflight['tokenizer']['export_sha256']
        print(json.dumps(record),flush=True);return
    import wandb
    api=wandb.Api(timeout=120);run=api.run('helloimlixin-rutgers/laser/'+RUN_ID)
    if args.action=='finish':
        import torch
        last=args.local/'checkpoints/last.pt'
        payload=torch.load(last,map_location='cpu',weights_only=False,mmap=True)
        validated=validate_recovered_checkpoint(payload,last)
        sources={'last.pt':last}
        for index,(_,old) in enumerate(payload['best_fid'],1):
            sources[f'best-fid-{index:02d}.pt']=args.local/'checkpoints'/Path(old).name
        expected={}
        for name,path in sources.items():
            with path.open('rb') as stream:
                expected[name]=base64.b64encode(hashlib.file_digest(stream,'md5').digest()).decode()
        for attempt in range(18):
            current=wandb.Api(timeout=120).run('helloimlixin-rutgers/laser/'+RUN_ID)
            actual={name:current.file(name).md5 for name in expected}
            if actual==expected:break
            time.sleep(10)
        else:raise RuntimeError('W&B has not confirmed the final latest/best checkpoint bytes')
        record=dict(online_uploads_verified=True,files=expected,**validated)
        atomic_json(base/'train'/f"uploads-{os.environ['SLURM_JOB_ID']}.json",record)
        current.summary.update({'amarel/last_verified_checkpoint_step':validated['step'],
            'amarel/last_completed_slurm_job':os.environ['SLURM_JOB_ID'],
            'amarel/target_complete':validated['step']>=88740})
        print(json.dumps(record),flush=True);return
    if args.action=='resolve':
        if run.state=='running' and os.environ.get('CHURCH_HANDOFF_FROM'):
            for _ in range(12):
                time.sleep(5)
                run=wandb.Api(timeout=120).run('helloimlixin-rutgers/laser/'+RUN_ID)
                if run.state!='running':break
        if run.state=='running':raise RuntimeError('Another trainer reports this W&B run active')
        names=['last.pt','best-fid-01.pt']
        files=[dict(name=name,md5=run.file(name).md5,size=run.file(name).size) for name in names]
        minimum=max(preflight['step'],int(os.environ.get('CHURCH_MIN_STEP','0')))
        record=dict(run_id=RUN_ID,files=files,minimum_step=minimum)
        atomic_json(args.selection,record);print(json.dumps(record),flush=True);return
    selection=json.loads(args.selection.read_text());local=args.local
    inputs=local/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    for row in selection['files']:
        f=run.file(row['name']);assert f.md5==row['md5'] and f.size==row['size']
        path=Path(f.download(root=str(inputs),exist_ok=True,replace=True).name)
        with path.open('rb') as stream:digest=base64.b64encode(hashlib.file_digest(stream,'md5').digest()).decode()
        assert digest==row['md5']
    import torch
    payload=torch.load(inputs/'last.pt',map_location='cpu',weights_only=False,mmap=True)
    validated=validate_recovered_checkpoint(payload,inputs/'last.pt')
    assert validated['step']>=selection['minimum_step'], 'Online checkpoint predates the verified handoff'
    best=payload['best_fid']
    assert len(best)==1, 'This continuation retains exactly one best-FID checkpoint'
    best_payload=torch.load(inputs/'best-fid-01.pt',map_location='cpu',weights_only=False,mmap=True)
    assert abs(float(best_payload['fid'])-float(best[0][0]))<1e-9, 'Online latest/best slots are inconsistent'
    ckpts=local/'checkpoints';ckpts.mkdir(exist_ok=True)
    for _,old in best:
        target=ckpts/Path(old).name
        if not target.exists():os.link(inputs/'best-fid-01.pt',target)
    if not (ckpts/'last.pt').exists():os.link(inputs/'last.pt',ckpts/'last.pt')
    selection=dict(selection,validated_checkpoint=validated,best_fid=best)
    atomic_json(local/'source-selection.json',selection)
    print(json.dumps(dict(phase='staged',node=os.uname().nodename,**selection)),flush=True)

if __name__=='__main__':main()
